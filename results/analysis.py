import os
import re

import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

root_dir = "../results/"

if __name__ == "__main__":
    datasets, transforms, label_cluster = list_params(root_dir)
    print(datasets)
    print(transforms)
    print(label_cluster)
    results = load_results(root_dir, datasets, transforms, label_cluster)

    # Iterate (within dataset -1) over all transforms and print the results for label_cluster 0,0
    dataset = datasets[1]
    for label_cluster_i in label_cluster:
        label, cluster = label_cluster_i
        to_plot = {}
        for transform in transforms:
            res = results[dataset][transform][(label, cluster)]
            res = res[["Wasserstein", "Lipschitz"]]
            # Delete all dominated points (Pareto front)
            if res is not None and len(res) > 0:
                pareto_mask = []
                for i, row_i in res.iterrows():
                    dominated = False
                    for j, row_j in res.iterrows():
                        if i != j:
                            if (row_j["Wasserstein"] <= row_i["Wasserstein"] and
                                row_j["Lipschitz"] <= row_i["Lipschitz"] and
                                (row_j["Wasserstein"] < row_i["Wasserstein"] or
                                 row_j["Lipschitz"] < row_i["Lipschitz"])):
                                dominated = True
                                break
                    pareto_mask.append(not dominated)
                res = res[pareto_mask]
            to_plot[transform] = res

        # Jointly plot all results
        plt.figure(figsize=(6, 4))
        for transform, res in to_plot.items():
            if res is not None:
                sns.scatterplot(data=res, x="Wasserstein", y="Lipschitz", label=transform)
        plt.xlabel("Wasserstein Distance")
        plt.ylabel("1-Lipschitz Metric")
        plt.title(f"Pareto Fronts for Dataset {dataset}, Label {label}, Cluster {cluster}")
        plt.legend()
        plt.grid(True)
        plt.show()

        # Create a df with all the solutions and a column for the transform
