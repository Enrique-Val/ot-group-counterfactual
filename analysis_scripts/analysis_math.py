import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from itertools import product

import numpy as np
import pandas as pd

import scikit_posthocs as sp

import matplotlib.pyplot as plt
import seaborn as sns
from torch.distributions.constraints import independent

from analysis_scripts.utils import list_params, load_results, friedman_posthoc, palette, renaming, plot_order, fig_size, \
    plot_performance_profile

root_dir = "../results"
n_clusters = 10

def run_friedman_by_K(all_df, correct="bergmann", palette = None, store_independent = True):
    """
    For each metric and K, aggregate values across datasets and label clusters,
    then run Friedman + CD diagram.
    """
    fs = fig_size*1.2
    results = {}
    for metric in all_df["metric"].unique():
        if "test" not in metric:
            continue  # Only test metrics
        fig, axes = plt.subplots(figsize=(12, 8), nrows=3, ncols=2)
        for i,K_val in enumerate(sorted(all_df["K"].unique())):
            ax = axes[i // 2, i % 2]
            if store_independent :
                fig = plt.figure(figsize=(5, 3))
                ax = fig.gca()
            subset = all_df[(all_df["metric"] == metric) & (all_df["K"] == K_val)]
            if "Validity" in metric:
                # Invert validity so that higher is better
                subset["value"] = 1.0 - subset["value"]
            if subset.empty:
                continue

            # pivot: rows = dataset–label_cluster, cols = transforms
            pivot = subset.pivot_table(
                index=["dataset", "label_cluster"],
                columns="transform",
                values="value",
                aggfunc="mean"
            )

            if pivot.shape[1] < 2:
                continue  # skip if fewer than two transforms

            friedman_res = friedman_posthoc(pivot, correct=correct)
            results[(metric, K_val)] = friedman_res

            # --- Plot CD diagram ---
            sp.critical_difference_diagram(
                friedman_res["summary_ranks"],
                friedman_res["p_adjusted"],
                label_fmt_left="{label}",
                label_fmt_right="{label}",
                color_palette=palette,
                ax=ax,
            )
            if store_independent :
                fig.tight_layout()
                plot_path = os.path.join(plots_dir, "cdd", f"friedman_cd_{metric.replace(' ','_')}_K{K_val}.pdf")
                fig.savefig(plot_path, bbox_inches="tight")
            ax.set_title(f"{metric}  (K={K_val})")
        # The last subplot will be used for all the K's at the same time
        ax = axes[2, 1]
        if store_independent :
            fig = plt.figure(figsize=fs)
            ax = fig.gca()
        subset = all_df[all_df["metric"] == metric]
        # pivot: rows = dataset–label_cluster–K, cols = transforms
        pivot = subset.pivot_table(
            index=["dataset", "label_cluster", "K"],
            columns="transform",
            values="value",
            aggfunc="mean"
        )
        if pivot.shape[1] < 2:
            continue  # skip if fewer than two transforms>
        friedman_res = friedman_posthoc(pivot, correct=correct)
        results[(metric, "all_K")] = friedman_res
        sp.critical_difference_diagram(
            friedman_res["summary_ranks"],
            friedman_res["p_adjusted"],
            label_fmt_left="{label}",
            label_fmt_right="{label}",
            color_palette=palette,
            ax=ax,
        )
        if store_independent :
            fig.tight_layout()
            plot_path = os.path.join(plots_dir, "cdd", f"friedman_cd_{metric.replace(' ','_')}_all_K.pdf")
            fig.savefig(plot_path, bbox_inches="tight")
        else :
            ax.set_title(f"{metric}  (all K)")
            fig.tight_layout()
            plot_path = os.path.join(plots_dir, f"friedman_cd_{metric.replace(' ','_')}.pdf")
            fig.savefig(plot_path, bbox_inches="tight")
            fig.show()

    return results

if __name__ == "__main__":
    # Create a plots subdir
    plots_dir = os.path.join(root_dir, "plots", "math_opt")
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)

    datasets, transforms, label_clusters = list_params(root_dir, n_clusters=n_clusters, exp_type="math_opt")
    datasets = datasets[1:]  # Exclude synthetic
    print(datasets)
    print(transforms)
    print(label_clusters)
    results = load_results(root_dir, datasets, transforms, label_clusters, n_clusters=n_clusters, exp_type="math_opt")
    # Iterate and set first column as index
    for dataset in datasets:
        for transform in transforms:
            for label_cluster_i in label_clusters:
                df = results[dataset][transform][label_cluster_i]
                if df is None:
                    raise ValueError(f"Missing data for dataset {dataset}, transform {transform}, label_cluster {label_cluster_i}")
                df.set_index(df.columns[0], inplace=True)
                # Create two new metrics: Distortion and Distortion test
                df["Distortion"] = 1 -np.min([df["Empirical lower bound"],(1/ df["Empirical upper bound"])], axis=0)
                df["Distortion test"] = 1 -np.min([df["Empirical lower bound test"], (1/ df["Empirical upper bound test"])], axis=0)
                # Remove last two Ks
                df = df.iloc[:-2, :]
                results[dataset][transform][label_cluster_i] = df


    # 2. GENERATE RAW & NORMALIZED DATA
    records_raw = []
    records_norm = []

    # Compute median metrics for Wachter
    all_Wachter_series = []
    for dataset, label_cluster_i in product(datasets, label_clusters):
        df = results[dataset]["Wachter"][label_cluster_i].iloc[0]  # First row, they are all the same
        all_Wachter_series.append(df)
    median_Wachter = pd.concat(all_Wachter_series, axis=1).median(axis=1)

    # If the ws distance for a method is NaN or too large, we consider the entire experiment broken
    WASS_THRESH = 1000
    broken_count = pd.Series(data=0, index=transforms)

    for dataset in datasets:
        for label_cluster_i in label_clusters:
            # Get Baseline Data
            Ks = results[dataset]["Wachter"][label_cluster_i].index
            base_df = results[dataset]["Wachter"][label_cluster_i].iloc[0]  # First rwo, they are all the same

            # Mask for K values where a method failed
            bad_ws_mask = pd.Series(data=False, index=Ks)

            # Count instances being NaN or too large
            for transform in transforms:
                df = results[dataset][transform][label_cluster_i]
                base_ws = df["Wasserstein test"]
                bad_ws_mask_t = (base_ws.isna()) | (base_ws > WASS_THRESH)
                n_broken = bad_ws_mask_t.sum()
                broken_count[transform] += n_broken
                bad_ws_mask = bad_ws_mask | bad_ws_mask_t  # Union of bad masks

            for transform in transforms:
                df = results[dataset][transform][label_cluster_i]
                # Create a copy for normalization
                df_norm = df.copy()

                # Change nans to infinite for time metric (or 0 if column is Empirical lower bound, Empirical lower bound test or Validity test)
                for col in df.columns:
                    if "Empirical lower bound" in col or "Validity test" in col or "Distortion" in col:
                        df[col] = df[col].fillna(0)
                    else:
                        df[col] = df[col].fillna(np.inf)

                # --- PATH A: RAW DATA (Includes Baseline) ---
                # We save everything, including DirectOptimization
                df_long = df.melt(ignore_index=False, var_name="metric", value_name="value")
                df_long["dataset"] = dataset
                df_long["transform"] = transform
                df_long["label_cluster"] = str(label_cluster_i)
                df_long["K"] = df_long.index
                records_raw.append(df_long.reset_index(drop=True))

                # --- PATH B: NORMALIZED DATA (Excludes Baseline) ---
                # IMPORTANT: We remove experiments where ANY method fails. We have a counter for that above.
                if transform == "Wachter":
                    continue  # Skip baseline (Ratio=1.0 is redundant information)

                # Direct Vectorized Division (Fast & Clean)
                # Since indices match exactly, pandas divides row-by-row automatically
                # remove inter-subject variability while maintaining interpretability
                for metric in df.columns:
                    continue
                    df_norm[metric] = df[metric] / base_df[metric] * median_Wachter[metric]

                # Drop rows with bad ws
                df_norm = df_norm[~bad_ws_mask]

                # Melt and Store
                df_norm_long = df_norm.melt(ignore_index=False, var_name="metric", value_name="value")
                df_norm_long["dataset"] = dataset
                df_norm_long["transform"] = transform
                df_norm_long["label_cluster"] = str(label_cluster_i)
                df_norm_long["K"] = df_norm_long.index
                records_norm.append(df_norm_long.reset_index(drop=True))


    print(f"Total of {broken_count} instances where baseline Wasserstein was NaN or > {WASS_THRESH}")
    print(f"Total instances processed: {len(records_norm)}")

    # 3. CONCATENATE
    all_df_raw = pd.concat(records_raw, ignore_index=True)
    all_df_norm = pd.concat(records_norm, ignore_index=True)


    # Rename transforms
    all_df_raw["transform"] = all_df_raw["transform"].replace(renaming)
    all_df_norm["transform"] = all_df_norm["transform"].replace(renaming)

    actual_plot_order = [i for i in plot_order if i in all_df_raw["transform"].unique()]

    # make one boxplot per metric
    '''for metric in all_df_norm["metric"].unique():
        fig = plt.figure(figsize=fig_size)
        ax = fig.gca()
        ax.grid(True)
        subset = all_df_norm[all_df_norm["metric"] == metric]
        print("Medians for metric ", metric)
        print(subset.groupby("transform")["value"].median())
        print("Total median" , subset["value"].median())
        print("-----")
        sns.boxplot(data=subset, x="transform", y="value", color=None, showfliers = False,
                    hue= "transform",
                    palette=palette,
                    order = actual_plot_order[1:],
                    ax = ax)
        # Log scale for time
        ax.set_yscale("log")
        sns.despine()
        #ax.set_title(f"{metric}")
        n = len(subset[subset["transform"] == plot_order[0]])
        # Write the number of samples per box at the top of the plot
        ax.text(0.5, 1 * np.quantile(subset["value"],0.9), "n = " + str(n), horizontalalignment='center',
                      verticalalignment='center', fontsize=10)
        ax.set_xlabel("Transform")
        ax.set_ylabel(metric)
        #ax.set_xlabel("")
        #ax.set_ylabel("")
        # Save plot
        plot_path = os.path.join(plots_dir, f"boxplot_{metric.replace(' ','_')}.pdf")
        fig.tight_layout()
        fig.savefig(plot_path, bbox_inches="tight")
        fig.show()'''

    # Filter for just the Wasserstein metrics
    wass_df = all_df_norm[all_df_norm["metric"].str.contains("Wasserstein")]

    # Filter out for Group (exclude these)
    wass_df = wass_df[~wass_df["transform"].str.contains("Group")]

    plt.figure(figsize=(12, 6))
    sns.boxplot(
        data=wass_df,
        x="transform",
        y="value",
        hue="metric",  # This splits the box into Train (Blue) and Test (Orange)
        palette="muted",
        showfliers=False,
        order=actual_plot_order[3:],  # Exclude baseline
    )
    plt.axhline(1.0, color='red', linestyle='--', label="Baseline Reference")
    plt.yscale("log")
    plt.ylabel("Normalized Cost (Ratio to Baseline)")
    plt.title("Generalization Gap: Train vs. Test Performance")
    plt.legend(title="Metric")
    plt.tight_layout()
    plt.show()

    # Create a two performance profile plots: one for time, one for wasserstein test
    fs = fig_size*1.2
    fig = plt.figure(figsize=fs)
    ax = fig.gca()
    # Plot Time Profile
    plot_performance_profile(all_df_raw, "Wasserstein test", ax=ax, palette=palette, max_x=7, verbose=True)
    fig.tight_layout()
    plot_path = os.path.join(plots_dir, f"performance_profile_Wasserstein_test.pdf")
    fig.savefig(plot_path, bbox_inches="tight")
    fig.show()

    fig = plt.figure(figsize=fs)
    ax = fig.gca()
    # Plot Time Profile
    plot_performance_profile(all_df_raw, "Exec time", ax=ax, palette=palette, max_x=1e10, verbose = True)
    fig.tight_layout()
    plot_path = os.path.join(plots_dir, f"performance_profile_Time.pdf")
    fig.savefig(plot_path, bbox_inches="tight")
    fig.show()

    LOWER_BOUND = "Empirical lower bound test"
    UPPER_BOUND = "Empirical upper bound test"
    bound_metrics = [LOWER_BOUND, UPPER_BOUND]

    # Lineplot. K is the x-axis, transform is hue, value is y-axis
    for metric in all_df_norm["metric"].unique():
        fig = plt.figure(figsize=fig_size*1.2)
        ax = fig.gca()
        ax.grid(True)
        if "test" not in metric or "bound" in metric:
            continue  # Only plot test metrics
        if "Wasserstein" in metric or "Time" in metric or True :
            subset = all_df_norm[all_df_norm["metric"] == metric]
            # Filter out Wachter
            subset = subset[subset["transform"] != "Wachter"]
            # Horizontal line at y=1, named Wachter
            ax.axhline(median_Wachter[metric], color=palette["Wachter"], linestyle='--', label="Wachter")
            lp_plot_order = actual_plot_order[1:]  # Exclude Wachter
        else :
            subset = all_df_raw[all_df_raw["metric"] == metric]
            lp_plot_order = actual_plot_order
        sns.lineplot(data=subset, x="K", y="value", hue="transform",
                     palette=palette,
                     hue_order=lp_plot_order,
                     marker="o",
                     ax=ax,
                     estimator="median")
        if metric == "Time":
            ax.set_yscale("log")
        sns.despine()
        ax.set_xlabel("K parameter")
        ax.set_ylabel(metric[:-5])
        ax.legend(title="Method", loc="upper right")
        # Save plot
        fig.tight_layout()
        plot_path = os.path.join(plots_dir, f"lineplot_{metric.replace(' ','_')}.pdf")
        fig.savefig(plot_path, bbox_inches="tight")
        fig.show()

    # ==========================================
    # PART B: Separate Block for Combined Bounds
    # ==========================================
    #raise NotImplementedError("Combined Bounds Plot is not ready yet.")
    fig = plt.figure(figsize=fig_size * 1.2)
    ax = fig.gca()
    ax.grid(True)

    # 1. Filter for BOTH metrics at once
    subset = all_df_norm[all_df_norm["metric"].isin(bound_metrics)]

    # 2. Filter out Wachter rows
    subset = subset[subset["transform"] != "Wachter"]

    # 3. Add Horizontal lines for Wachter (one for each bound if they differ)
    # We use different linestyles so you can tell which baseline belongs to which bound
    line_styles = ['-', '--']
    for i, m in enumerate(bound_metrics):
        if m in median_Wachter :
            if i == 0:
                ax.axhline(median_Wachter[m],
                           color=palette["Wachter"],
                           linestyle=line_styles[0],
                           label=f"Wachter",
                           alpha=0.7)
            else:
                ax.axhline(median_Wachter[m],
                           color=palette["Wachter"],
                           linestyle=line_styles[1],
                           alpha=0.7)

    lp_plot_order = actual_plot_order[1:]

    # 4. Plot with style="metric"
    # This distinguishes the Upper/Lower bounds using line styles (solid vs dashed)
    # while 'hue' still handles the transform types.
    sns.lineplot(data=subset, x="K", y="value", hue="transform",
                 palette=palette,
                 hue_order=lp_plot_order,
                 style="metric",  # <--- Key change: separates the bounds visually
                 marker="o",
                 ax=ax,
                 estimator="median",
                 ci=None)

    sns.despine()
    ax.set_xlabel("K parameter")
    ax.set_ylabel("Distortion")

    # Remove from legend the transform header and the metric info
    handles, labels = ax.get_legend_handles_labels()
    new_handles = []
    new_labels = []
    for handle, label in zip(handles, labels):
        if label in plot_order:
            new_handles.append(handle)
            new_labels.append(label)
    ax.legend(new_handles, new_labels, title="Method", loc="upper right")

    # Save Combined Plot
    fig.tight_layout()
    plot_path = os.path.join(plots_dir, "lineplot_Combined_Bounds.pdf")
    fig.savefig(plot_path, bbox_inches="tight")
    fig.show()
    plt.close(fig)

    # Run Friedman + post-hoc tests
    friedman_results = run_friedman_by_K(all_df_raw, correct="bergmann", palette=palette)