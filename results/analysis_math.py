import os
import re

import pandas as pd

import matplotlib.pyplot as plt
#import seaborn as sns

from results.utils import list_params, load_results

root_dir = "../results/"
n_clusters = 5

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

    # aggregate over dataset and label_cluster
    agg_df = all_df.groupby(["transform", "metric"])["value"].apply(list).reset_index()

    # make one boxplot per metric
    for metric in agg_df["metric"].unique():
        fig = plt.figure(figsize=[12,8])
        ax = fig.gca()
        subset = all_df[all_df["metric"] == metric]
        subset.boxplot(column="value", by="transform", grid=False, ax =ax)
        #ax.title(f"{metric}")
        fig.suptitle("")
        #ax.xlabel("Transform")
        #ax.ylabel(metric)
        fig.show()