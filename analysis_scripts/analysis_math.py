import os

import numpy as np
import pandas as pd

import scikit_posthocs as sp

import matplotlib.pyplot as plt
import seaborn as sns

from analysis_scripts.utils import list_params, load_results, friedman_posthoc, palette, renaming, plot_order, fig_size, \
    plot_performance_profile

root_dir = "../results"
n_clusters = 10

def run_friedman_by_K(all_df, correct="bergmann", palette = None):
    """
    For each metric and K, aggregate values across datasets and label clusters,
    then run Friedman + CD diagram.
    """
    results = {}
    for metric in all_df["metric"].unique():
        fig, axes = plt.subplots(figsize=(12, 8), nrows=3, ncols=2)
        for i,K_val in enumerate(sorted(all_df["K"].unique())):
            subset = all_df[(all_df["metric"] == metric) & (all_df["K"] == K_val)]
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
            ax = axes[i // 2, i % 2]
            sp.critical_difference_diagram(
                friedman_res["summary_ranks"],
                friedman_res["p_adjusted"],
                label_fmt_left="{label}",
                label_fmt_right="{label}",
                color_palette=palette,
                ax=ax,
            )
            ax.set_title(f"{metric}  (K={K_val})")
        # The last subplot will be used for all the K's at the same time
        ax = axes[2, 1]
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
        ax.set_title(f"{metric}  (all K)")
        fig.tight_layout()
        fig.show()

    return results

if __name__ == "__main__":
    # Create a plots subdir
    plots_dir = os.path.join(root_dir, "plots", "math_opt")
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)

    datasets, transforms, label_clusters = list_params(root_dir, n_clusters=n_clusters, exp_type="math_opt")
    datasets = datasets[:-1]
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
                results[dataset][transform][label_cluster_i] = df

    # 2. GENERATE RAW & NORMALIZED DATA
    records_raw = []
    records_norm = []
    records_anchored = []

    baseline_broken_count = 0

    for dataset in datasets:
        for label_cluster_i in label_clusters:
            # Get Baseline Data
            base_df = results[dataset]["DirectOptimization"][label_cluster_i]
            base_time = base_df["Time"]
            base_ws = base_df["Wasserstein test"]

            WASS_THRESH = 1000

            bad_ws_mask = (base_ws.isna()) | (base_ws > WASS_THRESH)
            baseline_broken_count += bad_ws_mask.sum()

            for transform in transforms:
                df = results[dataset][transform][label_cluster_i]

                # --- PATH A: RAW DATA (Includes Baseline) ---
                # We save everything, including DirectOptimization
                df_long = df.melt(ignore_index=False, var_name="metric", value_name="value")
                df_long["dataset"] = dataset
                df_long["transform"] = transform
                df_long["label_cluster"] = str(label_cluster_i)
                df_long["K"] = df_long.index
                records_raw.append(df_long.reset_index(drop=True))

                # --- PATH B: ANCHORED DATA (New Logic) ---
                # 1. Get the Anchor Value safely (Baseline at largest K)
                anchor_idx = base_ws.index.max()
                anchor_val = base_ws.loc[anchor_idx]
                anchor_time = base_time.loc[anchor_idx]

                # 2. Check Anchor Validity (The "Gatekeeper")
                # If the anchor is NaN or > Threshold, we DROP this entire anchored experiment
                is_anchor_valid = not (pd.isna(anchor_val) or anchor_val > WASS_THRESH)

                if is_anchor_valid:
                    df_anchor = df.copy()

                    # Normalize everything by the SCALAR anchor value
                    df_anchor["Wasserstein test"] = df["Wasserstein test"] / anchor_val
                    df_anchor["Time"] = df["Time"] / anchor_time

                    # We apply the bad_ws_mask from the baseline to the anchored data as well
                    df_anchor.loc[bad_ws_mask, "Wasserstein test"] = np.nan
                    df_anchor.loc[bad_ws_mask, "Wasserstein"] = np.nan

                    df_anchor_long = df_anchor.melt(ignore_index=False, var_name="metric", value_name="value")
                    df_anchor_long["dataset"] = dataset
                    df_anchor_long["transform"] = transform
                    df_anchor_long["label_cluster"] = str(label_cluster_i)
                    df_anchor_long["K"] = df_anchor_long.index
                    records_anchored.append(df_anchor_long.reset_index(drop=True))

                # --- PATH C: NORMALIZED DATA (Excludes Baseline) ---
                if transform == "DirectOptimization":
                    continue  # Skip baseline (Ratio=1.0 is redundant information)

                # Create a copy for normalization
                df_norm = df.copy()

                # Direct Vectorized Division (Fast & Clean)
                # Since indices match exactly, pandas divides row-by-row automatically
                df_norm["Wasserstein test"] = df["Wasserstein test"] / base_ws
                df_norm["Time"] = df["Time"] / base_time

                df_norm.loc[bad_ws_mask, "Wasserstein test"] = np.nan
                df_norm.loc[bad_ws_mask, "Wasserstein"] = np.nan

                # Melt and Store
                df_norm_long = df_norm.melt(ignore_index=False, var_name="metric", value_name="value")
                df_norm_long["dataset"] = dataset
                df_norm_long["transform"] = transform
                df_norm_long["label_cluster"] = str(label_cluster_i)
                df_norm_long["K"] = df_norm_long.index
                records_norm.append(df_norm_long.reset_index(drop=True))


    print(f"Total of {baseline_broken_count} instances where baseline Wasserstein was NaN or > {WASS_THRESH}")
    print(f"Total instances processed: {len(records_norm)}")

    # 3. CONCATENATE
    all_df_raw = pd.concat(records_raw, ignore_index=True)
    all_df_norm = pd.concat(records_norm, ignore_index=True)
    all_df_anchored = pd.concat(records_anchored, ignore_index=True)


    # Rename transforms
    all_df_raw["transform"] = all_df_raw["transform"].replace(renaming)
    all_df_norm["transform"] = all_df_norm["transform"].replace(renaming)
    all_df_anchored["transform"] = all_df_anchored["transform"].replace(renaming)

    actual_plot_order = [i for i in plot_order if i in all_df_raw["transform"].unique()]

    # make one boxplot per metric
    for metric in all_df_norm["metric"].unique():
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
        '''ax.text(0.5, 1 * np.quantile(subset["value"],0.9), "n = " + str(n), horizontalalignment='center',
                      verticalalignment='center', fontsize=10)'''
        ax.set_xlabel("Transform")
        ax.set_ylabel(metric)
        #ax.set_xlabel("")
        #ax.set_ylabel("")
        # Save plot
        plot_path = os.path.join(plots_dir, f"boxplot_{metric.replace(' ','_')}.pdf")
        fig.tight_layout()
        fig.savefig(plot_path, bbox_inches="tight")
        fig.show()

    # Filter for just the Wasserstein metrics
    wass_df = all_df_norm[all_df_norm["metric"].str.contains("Wasserstein")]

    plt.figure(figsize=(12, 6))
    sns.boxplot(
        data=wass_df,
        x="transform",
        y="value",
        hue="metric",  # This splits the box into Train (Blue) and Test (Orange)
        palette="muted",
        showfliers=False
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
    plot_performance_profile(all_df_raw, "Wasserstein test", ax=ax, palette=palette, max_x=5, verbose=True)
    fig.tight_layout()
    plot_path = os.path.join(plots_dir, f"performance_profile_Wasserstein_test.pdf")
    fig.savefig(plot_path, bbox_inches="tight")
    fig.show()

    fig = plt.figure(figsize=fs)
    ax = fig.gca()
    # Plot Time Profile
    plot_performance_profile(all_df_raw, "Time", ax=ax, palette=palette, max_x=100)
    fig.tight_layout()
    plot_path = os.path.join(plots_dir, f"performance_profile_Time.pdf")
    fig.savefig(plot_path, bbox_inches="tight")
    fig.show()

    # Lineplot. K is the x-axis, transform is hue, value is y-axis
    for metric in all_df_anchored["metric"].unique()[1:]:
        fig = plt.figure(figsize=fig_size*1.2)
        ax = fig.gca()
        ax.grid(True)
        subset = all_df_anchored[all_df_anchored["metric"] == metric]
        sns.lineplot(data=subset, x="K", y="value", hue="transform",
                     palette=palette,
                     hue_order=actual_plot_order,
                     marker="o",
                     ax=ax,
                     estimator="median")
        if metric == "Time":
            ax.set_yscale("log")
        sns.despine()
        ax.set_xlabel("")
        ax.set_ylabel(metric)
        # Save plot
        fig.tight_layout()
        plot_path = os.path.join(plots_dir, f"lineplot_{metric.replace(' ','_')}.pdf")
        fig.savefig(plot_path, bbox_inches="tight")
        fig.show()

    # Run Friedman + post-hoc tests
    friedman_results = run_friedman_by_K(all_df_raw, correct="bergmann", palette=palette)