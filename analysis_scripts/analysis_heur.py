import itertools
import os

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from analysis_scripts.utils import list_params, load_results, friedman_posthoc, palette, renaming, plot_order, fig_size, \
    plot_performance_profile

import scikit_posthocs as sp
import pickle

root_dir = "../results/"
n_clusters = 10
plots_dir = os.path.join(root_dir, "plots","heuristic")

import numpy as np
import pandas as pd

def has_cvf(transform):
    return "DirectOptimization" not in transform and transform != "Wachter"

def calculate_hypervolume_2d(points, ref_point):
    """
    Computes the Hypervolume for 2D minimization problems.

    Args:
        points (np.array): shape (N, 2) array of sorted non-dominated points.
        ref_point (tuple): (x_ref, y_ref) reference point (usually > 1.0 after norm).

    Returns:
        float: The hypervolume (area covered).
    """
    if len(points) == 0:
        return 0.0

    # Ensure points are sorted by x (first objective)
    # Note: Since they are Pareto optimal, if x is increasing, y must be decreasing.
    points = points[points[:, 0].argsort()]

    # Reference coordinates
    ref_x, ref_y = ref_point

    hv = 0.0

    # Calculate area of rectangles formed by the points and the reference point
    # We slice vertically: from current x to next x, the height is (ref_y - current_y)
    for i in range(len(points)):
        current_x = points[i, 0]
        current_y = points[i, 1]

        # The width extends to the next point's x, or to the ref_x for the last point
        if i < len(points) - 1:
            next_x = points[i + 1, 0]
        else:
            next_x = ref_x

        # Skip if point is outside reference bounds (shouldn't happen with proper scaling)
        if current_x > ref_x or current_y > ref_y:
            continue

        width = next_x - current_x
        height = ref_y - current_y

        hv += width * height

    return hv


def get_problem_bounds(results_clean, datasets, transforms, label_clusters, wass_str, lip_str):
    """
    Pre-computes the global Min and Max for every (dataset, label_cluster, cvf)
    across ALL transforms. This is crucial for fair normalization.

    Returns:
        bounds: dict[(dataset, label_cluster, cvf)] -> {min_w, max_w, min_l, max_l}
    """
    bounds = {}

    for dataset in datasets:
        for label_cluster_i in label_clusters:
            # We need to scan all CVFs (1-10)
            for cvf in range(1, 11):
                all_wass = []
                all_lip = []

                for transform in transforms:
                    res = results_clean[dataset][transform][label_cluster_i]
                    if res is None or len(res) == 0:
                        continue

                    if not has_cvf(transform):
                        # DirectOptimization is valid for all CVFs, use it for bounds
                        data = res
                    else:
                        data = res[res["CVF"] == cvf]

                    if len(data) > 0:
                        all_wass.append(data[wass_str].values)
                        all_lip.append(data[lip_str].values)

                # If no data found for this specific fold across any method
                if not all_wass:
                    bounds[(dataset, label_cluster_i, cvf)] = None
                    continue

                # Concatenate to find global min/max for this problem instance
                flat_wass = np.concatenate(all_wass)
                flat_lip = np.concatenate(all_lip)

                bounds[(dataset, label_cluster_i, cvf)] = {
                    'min_w': np.min(flat_wass),
                    'max_w': np.max(flat_wass),
                    'min_l': np.min(flat_lip),
                    'max_l': np.max(flat_lip)
                }

    return bounds

def remove_self_dominated_sorted(df, wass_str, lip_str):
    """
    O(n log n) Pareto front extraction using sorting.
    For minimization: sort by Wasserstein ascending, keep points with improving Lipschitz.

    Args:
        df: DataFrame with solutions
        wass_str: Column name for Wasserstein distance
        lip_str: Column name for Lipschitz continuity

    Returns:
        DataFrame with only non-dominated solutions
    """
    if len(df) == 0:
        return df

    # Sort by Wasserstein ascending (primary key)
    df_sorted = df.sort_values(wass_str).reset_index(drop=True)

    pareto_indices = []
    min_lip_so_far = float('inf')

    for i in range(len(df_sorted)):
        current_lip = df_sorted.iloc[i][lip_str]

        # Keep this point if it improves on Lipschitz
        # (Wasserstein is already guaranteed to be >= all previous due to sorting)
        if current_lip < min_lip_so_far:
            pareto_indices.append(i)
            min_lip_so_far = current_lip

    return df_sorted.iloc[pareto_indices].reset_index(drop=True)


def compute_domination_count_sorted(df_subset, df_reference, wass_str, lip_str):
    """
    Count how many points in df_subset are dominated by df_reference.
    Uses efficient checking: a point (w1, l1) is dominated if there exists (w2, l2)
    where w2 < w1 AND l2 < l1.

    Returns:
        Number of non-dominated points in df_subset
    """
    if len(df_subset) == 0 or len(df_reference) == 0:
        return len(df_subset)

    # Convert to numpy for faster access
    wass_subset = df_subset[wass_str].values
    lip_subset = df_subset[lip_str].values
    wass_ref = df_reference[wass_str].values
    lip_ref = df_reference[lip_str].values

    non_dominated_count = 0

    # For each point in subset, check if it's dominated by any point in reference
    for i in range(len(df_subset)):
        w_i = wass_subset[i]
        l_i = lip_subset[i]

        # Check if dominated by any reference point
        # Dominated if: (wass_ref < w_i) AND (lip_ref < l_i) for ANY reference point
        dominated = np.any((wass_ref < w_i) & (lip_ref < l_i))

        if not dominated:
            non_dominated_count += 1

    return non_dominated_count


def clean_results_per_cvf(results_crude, datasets, transforms, label_clusters,
                          wass_str, lip_str):
    """
    Stage 1: Clean each experiment's Pareto front per CVF.
    This removes self-dominated solutions within each (dataset, transform, label_cluster, CVF).

    Returns:
        results_clean: Same structure as results_crude but with only Pareto-optimal points
    """
    results_clean = {}

    for dataset in datasets:
        results_clean[dataset] = {}

        for transform in transforms:
            results_clean[dataset][transform] = {}

            for label_cluster_i in label_clusters:
                df = results_crude[dataset][transform][label_cluster_i]

                if df is None or len(df) == 0:
                    results_clean[dataset][transform][label_cluster_i] = None
                    continue

                if not has_cvf(transform):
                    # No CVF column for DirectOptimization
                    df_clean = remove_self_dominated_sorted(df, wass_str, lip_str)
                    results_clean[dataset][transform][label_cluster_i] = df_clean
                else:
                    # Process each CVF fold separately
                    df_clean = pd.DataFrame()
                    for cvf in range(1, 11):
                        df_cvf = df[df["CVF"] == cvf]
                        if len(df_cvf) > 0:
                            df_cvf_pareto = remove_self_dominated_sorted(df_cvf, wass_str, lip_str)
                            df_clean = pd.concat([df_clean, df_cvf_pareto], ignore_index=True)

                    results_clean[dataset][transform][label_cluster_i] = df_clean

    return results_clean


def compute_joint_pareto_fronts(results_clean, datasets, transforms, label_clusters,
                                wass_str, lip_str):
    """
    Stage 2: Compute joint Pareto front across all transforms.
    For each (dataset, label_cluster, CVF), combine all transforms' solutions
    and extract the global Pareto front.

    Returns:
        all_solutions: dict[dataset][(label_cluster, cvf)] -> DataFrame of joint Pareto front
    """
    all_solutions = {}

    for dataset in datasets:
        all_solutions[dataset] = {}

        for label_cluster_i in label_clusters:
            for cvf in range(1, 11):
                # Aggregate solutions from all transforms for this (dataset, label_cluster, CVF)
                combined_solutions = pd.DataFrame()

                for transform in transforms:
                    res = results_clean[dataset][transform][label_cluster_i]

                    if res is None or len(res) == 0:
                        continue

                    if not has_cvf(transform):
                        # DirectOptimization has no CVF, use all its solutions
                        df_cvf = res[[wass_str, lip_str]]
                    else:
                        # Filter by CVF and drop the CVF column
                        df_cvf = res[res["CVF"] == cvf][[wass_str, lip_str]]

                    combined_solutions = pd.concat([combined_solutions, df_cvf], ignore_index=True)

                # Compute global Pareto front for this combination
                if len(combined_solutions) > 0:
                    pareto_front = remove_self_dominated_sorted(combined_solutions, wass_str, lip_str)
                    all_solutions[dataset][(label_cluster_i, cvf)] = pareto_front
                else:
                    all_solutions[dataset][(label_cluster_i, cvf)] = pd.DataFrame()

    return all_solutions


def compute_metrics_per_transform(results_clean, all_solutions, datasets, transforms,
                                  label_clusters, wass_str, lip_str, is_test=False):
    """
    Stage 3: Compute metrics for each transform.
    Includes Hypervolume (HV) with min-max normalization.
    """
    results = {}

    # Pre-calculate bounds for normalization
    print(f"  Pre-calculating normalization bounds for {'test' if is_test else 'train'}...")
    bounds_map = get_problem_bounds(results_clean, datasets, transforms, label_clusters, wass_str, lip_str)

    # Reference point for HV (after normalization to [0,1], we use 1.1 to capture boundary points)
    HV_REF_POINT = (1.001, 1.001)

    for dataset in datasets:
        results[dataset] = {}

        for transform in transforms:
            results[dataset][transform] = {}

            for label_cluster_i in label_clusters:
                res = results_clean[dataset][transform][label_cluster_i]

                # Initialize lists for per-CVF metrics
                min_wass_per_cvf = []
                min_lip_per_cvf = []
                dom_percent_per_cvf = []
                hv_per_cvf = []

                if res is None or len(res) == 0:
                    # No results: worst possible metrics
                    df_metrics = pd.Series({
                        f"Min {wass_str}": np.inf,
                        f"Min {lip_str}": np.inf,
                        "Domination percent" + (" test" if is_test else ""): 1.0,
                        "Hypervolume" + (" test" if is_test else ""): 0.0
                    })
                    results[dataset][transform][label_cluster_i] = df_metrics
                    continue

                # Process each CVF fold
                for cvf in range(1, 11):
                    # Get normalization bounds for this specific problem instance
                    bounds = bounds_map.get((dataset, label_cluster_i, cvf))

                    if not has_cvf(transform):
                        res_cvf = res  # No CVF column
                    else:
                        res_cvf = res[res["CVF"] == cvf]

                    # -- Handle Empty or Invalid Cases --
                    if len(res_cvf) == 0 or bounds is None:
                        min_wass_per_cvf.append(np.inf)
                        min_lip_per_cvf.append(np.inf)
                        dom_percent_per_cvf.append(1.0)  # 100% dominated (worst)
                        hv_per_cvf.append(0.0)  # 0 volume (worst)

                        if not has_cvf(transform):
                            break
                        continue

                    # 1. Standard Scalar Metrics
                    min_wass_per_cvf.append(res_cvf[wass_str].min())
                    min_lip_per_cvf.append(res_cvf[lip_str].min())

                    # 2. Domination Percentage
                    joint_pareto = all_solutions[dataset].get((label_cluster_i, cvf), pd.DataFrame())
                    if len(joint_pareto) == 0:
                        dom_percent_per_cvf.append(0.0)
                    else:
                        non_dom_count = compute_domination_count_sorted(
                            res_cvf[[wass_str, lip_str]], joint_pareto, wass_str, lip_str
                        )
                        domination_pct = 1.0 - (non_dom_count / len(res_cvf))
                        dom_percent_per_cvf.append(domination_pct)

                    # 3. Hypervolume (Normalized)
                    # Extract raw values
                    points = res_cvf[[wass_str, lip_str]].values.copy()

                    # Prevent division by zero if max == min (e.g., only one solution exists across all methods)
                    denom_w = bounds['max_w'] - bounds['min_w']
                    denom_l = bounds['max_l'] - bounds['min_l']

                    if denom_w < 1e-9: denom_w = 1.0
                    if denom_l < 1e-9: denom_l = 1.0

                    # Apply Min-Max Normalization: (x - min) / (max - min)
                    points[:, 0] = (points[:, 0] - bounds['min_w']) / denom_w
                    points[:, 1] = (points[:, 1] - bounds['min_l']) / denom_l

                    # Compute HV
                    hv = calculate_hypervolume_2d(points, HV_REF_POINT)
                    hv_per_cvf.append(hv)

                    # DirectOptimization: process only once
                    if not has_cvf(transform):
                        min_wass_per_cvf = [min_wass_per_cvf[0]] * 10
                        min_lip_per_cvf = [min_lip_per_cvf[0]] * 10
                        dom_percent_per_cvf = [dom_percent_per_cvf[0]] * 10
                        hv_per_cvf = [hv_per_cvf[0]] * 10
                        break

                # Aggregate metrics across CVFs (mean)
                suffix = " test" if is_test else ""
                df_metrics = pd.Series({
                    f"Min {wass_str}": np.mean(min_wass_per_cvf),
                    f"Min {lip_str}": np.mean(min_lip_per_cvf),
                    f"Domination percent{suffix}": np.mean(dom_percent_per_cvf),
                    f"Hypervolume{suffix}": np.mean(hv_per_cvf)
                })
                results[dataset][transform][label_cluster_i] = df_metrics

    return results


def create_summary_dataframe(results, datasets, transforms, label_clusters, wass_str, lip_str):
    """
    Updated summary creation to include Hypervolume column.
    """
    # Detect if this is test or train based on keys in the first valid result
    # (A bit hacky but works with the existing flow)
    sample_key = list(results[datasets[0]][transforms[0]].keys())[0]
    keys = results[datasets[0]][transforms[0]][sample_key].index.tolist()

    # Create DF with dynamic columns
    df_joint = pd.DataFrame(columns=keys + ["Dataset", "Transform", "Label_Cluster"])

    for dataset in datasets:
        for transform in transforms:
            for label_cluster_i in label_clusters:
                df_row = results[dataset][transform][label_cluster_i].to_frame().T
                df_row["Dataset"] = dataset
                df_row["Transform"] = transform
                df_row["Label_Cluster"] = str(label_cluster_i)
                df_joint = pd.concat([df_joint, df_row], ignore_index=True)

    return df_joint

def optimized_pipeline(results_crude, datasets, transforms, label_clusters,
                       wass_str, lip_str):
    """
    Complete optimized pipeline for computing domination metrics.

    Pipeline:
    1. Clean each experiment's Pareto front (per CVF)
    2. Compute joint Pareto fronts (all transforms combined, per CVF)
    3. Compute metrics per transform (min values + domination %, averaged across CVFs)
    4. Create summary DataFrame

    Args:
        results_crude: dict[dataset][transform][label_cluster] -> DataFrame
        datasets: list of dataset names
        transforms: list of transform names
        label_clusters: list of label cluster identifiers
        wass_str: column name for Wasserstein distance
        lip_str: column name for Lipschitz continuity

    Returns:
        df_summary: DataFrame with aggregated metrics
        results_clean: Cleaned Pareto fronts (for pickling)
        all_solutions: Joint Pareto fronts (for pickling)
    """
    print(f"Stage 1: Cleaning Pareto fronts per CVF...")
    results_clean = clean_results_per_cvf(
        results_crude, datasets, transforms, label_clusters, wass_str, lip_str
    )

    print(f"Stage 2: Computing joint Pareto fronts...")
    all_solutions = compute_joint_pareto_fronts(
        results_clean, datasets, transforms, label_clusters, wass_str, lip_str
    )

    print(f"Stage 3: Computing metrics per transform...")
    results = compute_metrics_per_transform(
        results_clean, all_solutions, datasets, transforms, label_clusters, wass_str, lip_str
    )

    print(f"Stage 4: Creating summary DataFrame...")
    df_summary = create_summary_dataframe(
        results, datasets, transforms, label_clusters, wass_str, lip_str
    )

    return df_summary, results_clean, all_solutions


# Complete integration with checkpoint logic (replaces your main processing code):
if __name__ == "__main__":

    datasets, transforms, label_clusters = list_params(root_dir, n_clusters=n_clusters, exp_type="heuristic")

    # Define checkpoint file paths
    pickle_file_train = os.path.join(root_dir, "results_crude_train.pkl")
    pickle_file_test = os.path.join(root_dir, "results_crude_test.pkl")
    all_solutions_file_train = os.path.join(root_dir, "all_solutions_train.pkl")
    all_solutions_file_test = os.path.join(root_dir, "all_solutions_test.pkl")
    summary_file_train = os.path.join(root_dir, "analysis_heuristic_summary.csv")
    summary_file_test = os.path.join(root_dir, "analysis_heuristic_summary test.csv")

    # File for new long format
    summary_file_long = os.path.join(root_dir, "analysis_heuristic_summary_long.csv")

    # CHECKPOINT 1: Check if cleaned results exist
    if not (os.path.exists(pickle_file_train) and os.path.exists(pickle_file_test)):
        print("Creating cleaned Pareto fronts (Stage 1)...")

        # Separate train and test data
        results_crude_train = {}
        results_crude_test = {}
        for dataset in datasets:
            results_crude_train[dataset] = {}
            results_crude_test[dataset] = {}
            for transform in transforms:
                results_crude_train[dataset][transform] = {}
                results_crude_test[dataset][transform] = {}
                for label_cluster_i in label_clusters:
                    df = pd.read_csv(os.path.join(
                        root_dir, dataset, str(n_clusters), "heuristic", transform,
                        f"label_{label_cluster_i[0]}_cluster_{label_cluster_i[1]}.csv"
                    ))
                    if df is not None and len(df) > 0:
                        if not has_cvf(transform):
                            df_train = df[["Wasserstein", "Lipschitz"]]
                            df_test = df[["Wasserstein", "Lipschitz"]]
                            # Rename columns to match expected names
                            df_test = df_test.rename(columns={"Wasserstein": "Wasserstein test", "Lipschitz": "Lipschitz test"})
                        else:
                            df_train = df[["CVF", "Wasserstein", "Lipschitz"]]
                            df_test = df[["CVF", "Wasserstein test", "Lipschitz test"]]
                        results_crude_train[dataset][transform][label_cluster_i] = df_train
                        results_crude_test[dataset][transform][label_cluster_i] = df_test
                    else:
                        results_crude_train[dataset][transform][label_cluster_i] = None
                        results_crude_test[dataset][transform][label_cluster_i] = None

        # Clean Pareto fronts for train and test
        print("  Cleaning train data...")
        results_clean_train = clean_results_per_cvf(
            results_crude_train, datasets, transforms, label_clusters,
            "Wasserstein", "Lipschitz"
        )

        print("  Cleaning test data...")
        results_clean_test = clean_results_per_cvf(
            results_crude_test, datasets, transforms, label_clusters,
            "Wasserstein test", "Lipschitz test"
        )

        # Save checkpoints
        print("  Saving cleaned results...")
        with open(pickle_file_train, "wb") as f:
            pickle.dump(results_clean_train, f)
        with open(pickle_file_test, "wb") as f:
            pickle.dump(results_clean_test, f)
    else:
        print("Loading cleaned Pareto fronts from checkpoint...")
        results_clean_train = pickle.load(open(pickle_file_train, "rb"))
        results_clean_test = pickle.load(open(pickle_file_test, "rb"))

    # CHECKPOINT 2: Check if joint solutions exist
    if not (os.path.exists(all_solutions_file_train) and os.path.exists(all_solutions_file_test)):
        print("Creating joint Pareto fronts (Stage 2)...")

        print("  Computing joint fronts for train data...")
        all_solutions_train = compute_joint_pareto_fronts(
            results_clean_train, datasets, transforms, label_clusters,
            "Wasserstein", "Lipschitz"
        )

        print("  Computing joint fronts for test data...")
        all_solutions_test = compute_joint_pareto_fronts(
            results_clean_test, datasets, transforms, label_clusters,
            "Wasserstein test", "Lipschitz test"
        )

        # Save checkpoints
        print("  Saving joint Pareto fronts...")
        with open(all_solutions_file_train, "wb") as f:
            pickle.dump(all_solutions_train, f)
        with open(all_solutions_file_test, "wb") as f:
            pickle.dump(all_solutions_test, f)
    else:
        print("Loading joint Pareto fronts from checkpoint...")
        all_solutions_train = pickle.load(open(all_solutions_file_train, "rb"))
        all_solutions_test = pickle.load(open(all_solutions_file_test, "rb"))

    # CHECKPOINT 3: Check if summary files exist
    if not (os.path.exists(summary_file_train) and os.path.exists(summary_file_test)):
        print("Computing metrics and creating summaries (Stage 3)...")

        # Process train data
        print("  Processing train data...")
        results_train = compute_metrics_per_transform(
            results_clean_train, all_solutions_train, datasets, transforms,
            label_clusters, "Wasserstein", "Lipschitz"
        )
        df_joint_train = create_summary_dataframe(
            results_train, datasets, transforms, label_clusters,
            "Wasserstein", "Lipschitz"
        )

        # Process test data
        print("  Processing test data...")
        results_test = compute_metrics_per_transform(
            results_clean_test, all_solutions_test, datasets, transforms,
            label_clusters, "Wasserstein test", "Lipschitz test", is_test=True
        )
        df_joint_test = create_summary_dataframe(
            results_test, datasets, transforms, label_clusters,
            "Wasserstein test", "Lipschitz test"
        )

        # Save summaries
        print("  Saving summary files...")
        print("Train summary:")
        print(df_joint_train)
        df_joint_train.to_csv(summary_file_train, index=False)

        print("Test summary:")
        print(df_joint_test)
        df_joint_test.to_csv(summary_file_test, index=False)
    else:
        print("Loading summaries from existing files...")
        df_joint_train = pd.read_csv(summary_file_train)
        df_joint_test = pd.read_csv(summary_file_test)

    print("\\nProcessing complete!")
    print(f"Train summary: {summary_file_train}")
    print(f"Test summary: {summary_file_test}")

    # Exlude transforms with word "proxy" in their name for the plots
    df_joint_train = df_joint_train[~df_joint_train["Transform"].str.contains("proxy")]
    df_joint_test = df_joint_test[~df_joint_test["Transform"].str.contains("proxy")]

    # --- Merge, Melt and Save Long Format ---
    print("Creating merged long-format summary...")

    # 1. Merge Train and Test summaries
    # We join on the identifier columns.
    df_merged = pd.merge(
        df_joint_train,
        df_joint_test,
        on=["Dataset", "Transform", "Label_Cluster"],
        how="outer"
    )

    # 2. Melt to long format
    # This takes all metric columns (Wasserstein, Hypervolume, etc.) and stacks them
    df_long = df_merged.melt(
        id_vars=["Dataset", "Transform", "Label_Cluster"],
        var_name="metric",
        value_name="value"
    )

    # 3. Clean up column names and sort
    df_long.columns = [c.lower() for c in df_long.columns]  # 'Dataset' -> 'dataset'

    # Reorder columns to user specification: metric, value, dataset, transform, label_cluster
    df_long = df_long[["metric", "value", "dataset", "transform", "label_cluster"]]

    # Save
    df_long.to_csv(summary_file_long, index=False)

    print("\\nProcessing complete!")
    print(f"Long-format summary saved to: {summary_file_long}")
    print(df_long.head())

    # Add the time metric
    df_time = pd.DataFrame(columns=df_long.columns)
    for dataset, transform in itertools.product(datasets, transforms):
        time_file = os.path.join(
            root_dir, dataset, str(n_clusters), "heuristic", transform,
            "exec_times.csv"
        )
        if os.path.exists(time_file):
            exec_time_i = pd.read_csv(time_file)
        # Set multiindex with the first two columns
            exec_time_i.set_index(["label", "cluster"], inplace=True)
        for label_cluster in label_clusters:
            if label_cluster in exec_time_i.index:
                time_val = exec_time_i.loc[label_cluster, "exec_time"]
                df_time = pd.concat([df_time, pd.DataFrame({
                    "metric": ["Time"],
                    "value": [time_val],
                    "dataset": [dataset],
                    "transform": [transform],
                    "label_cluster": [str(label_cluster)]
                })], ignore_index=True)

    # Append time data to long dataframe
    df_long = pd.concat([df_long, df_time], ignore_index=True)
    df_long = df_long.reset_index(drop=True)

    # Rename transforms
    df_long["transform"] = df_long["transform"].replace(renaming)

    # Delete rows with "Any affine" transform
    df_long = df_long[df_long["transform"] != "Any affine"]

    actual_plot_order = [i for i in plot_order if i in df_long["transform"].unique()]

    # Create a two performance profile plots: one for time, one for wasserstein test
    fs = fig_size * 1.2
    for metric in df_long["metric"].unique():
        fig = plt.figure(figsize=fs)
        ax = fig.gca()
        # Plot performance profile
        if "Hypervolume" in metric:
            plot_performance_profile(df_long, metric, ax=ax, palette=palette, max_x=10000, verbose=True, title=None,
                                     maximize=True)
        else:
            plot_performance_profile(df_long, metric, ax=ax, palette=palette, max_x=100, verbose=True, title=metric)
        fig.tight_layout()
        plot_path = os.path.join(plots_dir, f"performance_profile_{metric.replace(' ', '_')}.pdf")
        fig.savefig(plot_path, bbox_inches="tight", pad_inches=0)
        fig.show()

    # Now critical difference plots for each metric
    for metric in df_long["metric"].unique():
        fig = plt.figure(figsize=fs)
        ax = fig.gca()
        subset = df_long[(df_long["metric"] == metric)]
        if subset.empty:
            continue

        if "Hypervolume" in metric:
            # For Hypervolume, higher is better, so we invert values for ranking
            subset = subset.copy()
            max_value = subset["value"].max()
            subset["value"] = max_value - subset["value"]

        # pivot: rows = dataset–label_cluster, cols = transforms
        pivot = subset.pivot_table(
            index=["dataset", "label_cluster"],
            columns="transform",
            values="value"
        )

        if pivot.shape[1] < 2:
            continue  # skip if fewer than two transforms

        friedman_res = friedman_posthoc(pivot, correct="bergmann")

        # --- Plot CD diagram ---
        print(friedman_res["summary_ranks"])
        print(palette)
        sp.critical_difference_diagram(
            friedman_res["summary_ranks"],
            friedman_res["p_adjusted"],
            label_fmt_left="{label}",
            label_fmt_right="{label}",
            color_palette=palette,
            ax=ax,
        )
        #ax.set_title(f"{metric}")
        fig.tight_layout()
        plot_path = os.path.join(plots_dir, f"critical_difference_{metric.replace(' ', '_')}.pdf")
        fig.savefig(plot_path, bbox_inches="tight")
        fig.show()


