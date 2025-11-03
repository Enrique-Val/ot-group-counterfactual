import os
import re

import pandas as pd

import scikit_posthocs as sp

import matplotlib.pyplot as plt
import seaborn as sns

from results.utils import list_params, load_results, friedman_posthoc

root_dir = "../results/"
n_clusters = 5

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
    datasets, transforms, label_clusters = list_params(root_dir, n_clusters=n_clusters, exp_type="math_opt")
    print(datasets)
    print(transforms)
    print(label_clusters)
    results = load_results(root_dir, datasets, transforms, label_clusters)
    # Iterate and set first column as index
    for dataset in datasets:
        for transform in transforms:
            for label_cluster_i in label_clusters:
                df = results[dataset][transform][label_cluster_i]
                if df is not None:
                    df.set_index(df.columns[0], inplace=True)
                    results[dataset][transform][label_cluster_i] = df

    #Normalize W2 and time per dataset. Take as norm the time of DirectOptimization.
    for dataset in datasets:
        for label_cluster_i in label_clusters:
            # Get DirectOptimization time
            direct_opt_df = results[dataset]["DirectOptimization"][label_cluster_i]
            print(dataset, label_cluster_i)
            direct_opt_time = direct_opt_df["exec_time"]
            direct_opt_ws = direct_opt_df["Wasserstein"]
            for transform in transforms:
                df = results[dataset][transform][label_cluster_i]
                if df is not None and len(df) > 0:
                    df["Wasserstein"] = df["Wasserstein"] / direct_opt_ws
                    df["exec_time"] = df["exec_time"] / direct_opt_time
                    results[dataset][transform][label_cluster_i] = df

    # Iterate (within dataset -1) over all transforms and print the results for label_cluster 0,0
    dataset = datasets[1]
    for label_cluster_i in label_clusters:
        print(results[dataset][transforms[0]][label_cluster_i])


    records = []

    for dataset, dval in results.items():
        for transform, tval in dval.items():
            for label_cluster, df in tval.items():
                print(dataset, transform, label_cluster)
                df_long = df.melt(ignore_index=False, var_name="metric", value_name="value")
                df_long["dataset"] = dataset
                df_long["transform"] = transform
                df_long["label_cluster"] = str(label_cluster)
                df_long["K"] = df_long.index
                records.append(df_long.reset_index(drop=True))

    all_df = pd.concat(records, ignore_index=True)


    renaming = {"DiagonalAffine" : "Diagonal Affine", "GaussianCommutativeTransform" : "Gaussian Commutative",
                "GaussianTransform" : "Gaussian", "PSDAffine" : "PSD Affine", "GMMForwardTransform" : "GMM",
                "DirectOptimization" : "Direct Optimization"}

    plot_order = ["PSD Affine", "Diagonal Affine", "Gaussian", "Gaussian Commutative", "GMM"]


    # Drop DirectOptimization rows
    all_df_no_do = all_df[all_df["transform"] != "DirectOptimization"].reset_index(drop=True)


    # Rename transforms
    all_df["transform"] = all_df["transform"].replace(renaming)
    all_df_no_do["transform"] = all_df_no_do["transform"].replace(renaming)

    palette = sns.color_palette("husl", len(all_df["transform"].unique()))


    # make one boxplot per metric
    for metric in all_df_no_do["metric"].unique():
        fig = plt.figure(figsize=[12,8])
        ax = fig.gca()
        ax.grid(True)
        subset = all_df_no_do[all_df_no_do["metric"] == metric]
        sns.boxplot(data=subset, x="transform", y="value", color=None, showfliers = False,
                    hue= "transform",
                    palette=palette,
                    order = plot_order)
        sns.despine()
        ax.set_title(f"{metric}")
        n = len(subset[subset["transform"] == plot_order[0]])
        # Write the number of samples per box at the top of the plot
        ax.text(0.5, 0.05, "n = " + str(n), horizontalalignment='center',
                      verticalalignment='center', fontsize=10)
        fig.suptitle("")
        ax.set_xlabel("Transform")
        ax.set_ylabel(metric)
        fig.show()

    # make one boxplot per metric
    for metric in all_df_no_do["metric"].unique():
        fig, axes = plt.subplots(figsize=[12,8], nrows=3, ncols=2)
        for i,K_val in enumerate(all_df_no_do["K"].unique()):
            ax = axes [i // 2, i % 2]
            ax.grid(True)
            subset = all_df_no_do[(all_df_no_do["metric"] == metric) & (all_df_no_do["K"] == K_val)]

            sns.boxplot(data=subset, x="transform", y="value", color=None, showfliers=False,
                        hue="transform",
                        palette=palette,
                        order=plot_order, ax = ax)
            sns.despine()
            n = len(subset[subset["transform"] == plot_order[0]])
            # Write the number of samples per box at the top of the plot
            ax.text(0.5, 0.05, "n = " + str(n), horizontalalignment='center',
                    verticalalignment='center', fontsize=10)
            ax.set_title(f"{metric} for K={K_val}")
            ax.set_xlabel("Transform")
            ax.set_ylabel(metric)
        print()
        fig.show()

    # Run Friedman + post-hoc tests
    friedman_results = run_friedman_by_K(all_df, correct="bergmann", palette=palette)