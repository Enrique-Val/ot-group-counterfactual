import itertools
import os
import re

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from results.utils import list_params, load_results, friedman_posthoc

import scikit_posthocs as sp

root_dir = "../results/"

def remove_self_dominated(df1, df2, wass_str, lip_str):
    pareto_mask = []
    for i in range(df1.shape[0]):
        row_i = df1.iloc[i]
        dominated = False
        for j in range(df2.shape[0]):
            row_j = df2.iloc[j]
            if (row_j[wass_str] < row_i[wass_str] and
                row_j[lip_str] < row_i[lip_str]):
                #print("Row", i, "is dominated by row", j)
                dominated = True
                break
        pareto_mask.append(not dominated)
    return df1[pareto_mask].reset_index(drop=True)

if __name__ == "__main__":
    datasets, transforms, label_clusters = list_params(root_dir, n_clusters=5, exp_type="heuristic")
    print(datasets)
    print(transforms)
    print(label_clusters)
    results_crude = load_results(root_dir, datasets, transforms, label_clusters, exp_type='heuristic')
    #transforms = transforms[:2]
    #label_clusters = label_clusters[:2]

    transforms = [t for t in transforms if not "proxy" in t]

    # Check if the summary files already exist
    summary_file_train = os.path.join(root_dir, "analysis_heuristic_summary.csv")
    summary_file_test = os.path.join(root_dir, "analysis_heuristic_summary test.csv")
    if not (os.path.exists(summary_file_train) and os.path.exists(summary_file_test)):
        # First, divide each df of results crude into train and test
        results_crude_train = {}
        results_crude_test = {}
        for dataset in datasets:
            results_crude_train[dataset] = {}
            results_crude_test[dataset] = {}
            for transform in transforms:
                results_crude_train[dataset][transform] = {}
                results_crude_test[dataset][transform] = {}
                for label_cluster_i in label_clusters:
                    df = results_crude[dataset][transform][label_cluster_i]
                    if df is not None and len(df) > 0:
                        if transform == "DirectOptimization" :
                            df_train = df[["Wasserstein", "Lipschitz"]]
                            df_test = df[["Wasserstein test", "Lipschitz test"]]
                        else :
                            df_train = df[["CVF", "Wasserstein", "Lipschitz"]]
                            df_test = df[["CVF","Wasserstein test", "Lipschitz test"]]
                        results_crude_train[dataset][transform][label_cluster_i] = df_train
                        results_crude_test[dataset][transform][label_cluster_i] = df_test
                    else:
                        results_crude_train[dataset][transform][label_cluster_i] = None
                        results_crude_test[dataset][transform][label_cluster_i] = None

        # First, remove self-dominated solutions and collect all non-dominated solutions across transforms, datasets, label_clusters and CVF
        for str_i, results_crude_i in zip(["", " test"], [results_crude_train, results_crude_test]):
            for dataset in datasets:
                for transform in transforms:
                    for label_cluster_i in label_clusters:
                        df = results_crude_i[dataset][transform][label_cluster_i]
                        if df is not None and len(df) > 0:
                            if transform == "DirectOptimization" :
                                # This solutions has no self-domination, as it works with the empirical Lipschitz
                                pass
                            else :
                                # Divide df per CVF and remove self-dominated solutions per CVF
                                df_clean = pd.DataFrame()
                                for cvi in range(1,11):
                                    df_cvi = df[df["CVF"] == cvi]
                                    df_cvi_clean = remove_self_dominated(df_cvi, df_cvi, "Wasserstein" + str_i, "Lipschitz" + str_i)
                                    df_clean = pd.concat([df_clean, df_cvi_clean])
                                results_crude_i[dataset][transform][label_cluster_i] = df_clean.reset_index(drop=True)

        for str_i, results_crude_i in zip(["", " test"], [results_crude_train, results_crude_test]):
            metrics = ["Min " + i for i in results_crude_i[datasets[0]][transforms[0]][label_clusters[0]].columns[1:]]  + ["Domination percent" + str_i]
            wass_str = "Wasserstein" + str_i
            lip_str = "Lipschitz" + str_i

            # Prepare storage for results
            results ={}
            for dataset in datasets:
                results[dataset] = {}
                for transform in transforms:
                    results[dataset][transform] = {}
                    for label_cluster_i in label_clusters:
                        df = pd.Series(index = metrics)

            # All solutions contain the best individual across datasets, label_clusters and CVF, aggregating for all transforms
            all_solutions = {}
            for dataset in datasets:
                all_solutions[dataset] = {}
                for label_cluster_i,cvi in itertools.product(label_clusters, range(1,11)):
                    all_solutions_i = pd.DataFrame()
                    for transform in transforms:
                        if results_crude_i[dataset][transform][label_cluster_i] is not None :
                            df_cvi = results_crude_i[dataset][transform][label_cluster_i]
                            if not transform == "DirectOptimization" :
                                df_cvi = df_cvi[df_cvi["CVF"] == cvi]
                                # Drop CVF column
                                df_cvi = df_cvi.drop(columns=["CVF"])
                            all_solutions_i = pd.concat([all_solutions_i, df_cvi])
                    all_solutions_i = all_solutions_i.reset_index(drop=True)
                    all_solutions[dataset][(label_cluster_i, cvi)] = remove_self_dominated(all_solutions_i, all_solutions_i, wass_str, lip_str)

            # Now repeat iteration. For each transform, compute domination percentage by comparing to all solutions
            for dataset, transform, label_cluster_i in itertools.product(datasets, transforms, label_clusters):
                df = pd.Series(index = metrics)
                df[:] = np.nan
                res = results_crude_i[dataset][transform][label_cluster_i]
                if res is not None and len(res) > 0:
                    min_wass = []
                    min_lip = []
                    dp = []
                    for cvi in range(1, 11):
                        if not transform == "DirectOptimization" :
                            res_cvi = res[res["CVF"] == cvi]
                        else :
                            res_cvi = pd.DataFrame(columns=[wass_str, lip_str], data=results_crude_train[dataset][transform][label_cluster_i].to_numpy())
                        # Compute min metrics
                        min_wass.append(res_cvi[wass_str].min())
                        min_lip.append(res_cvi[lip_str].min())
                        # Compute domination percentage for proxy
                        all_solutions_i = all_solutions[dataset][(label_cluster_i, cvi)]

                        non_dom_global = remove_self_dominated(res_cvi, all_solutions_i, wass_str, lip_str)
                        domination_percent = 1- len(non_dom_global) / len(res_cvi)
                        dp.append(domination_percent)

                    # Set values in df
                    df[metrics[0]] = np.mean(min_wass)
                    df[metrics[1]] = np.mean(min_lip)
                    df[metrics[2]] = np.mean(dp)
                results[dataset][transform][label_cluster_i] = df

            # Iterate and put into a single df for further analysis if needed
            df_joint = pd.DataFrame(columns = metrics)
            for dataset, transform, label_cluster_i in itertools.product(datasets, transforms, label_clusters):
                df_i = results[dataset][transform][label_cluster_i]
                df_i = df_i.to_frame().T
                df_i["Dataset"] = dataset
                df_i["Transform"] = transform
                df_i["Label_Cluster"] = str(label_cluster_i)
                df_joint = pd.concat([df_joint, df_i], ignore_index=True)
            print("Joint results df:")
            print(df_joint)
            # Export to csv
            output_file = os.path.join(root_dir, f"analysis_heuristic_summary{str_i}.csv")
            df_joint.to_csv(output_file, index=False)

    # Load existing summary files
    summary_file_train = os.path.join(root_dir, "analysis_heuristic_summary.csv")
    summary_file_test = os.path.join(root_dir, "analysis_heuristic_summary test.csv")
    df_joint_train = pd.read_csv(summary_file_train)
    df_joint_test = pd.read_csv(summary_file_test)

    # Normalize with min wasserstein and min lipschitz of DirectOptimization per dataset and label_cluster
    for df_joint, str_i in zip([df_joint_train, df_joint_test], ["", " test"]):
        wass_str = "Min Wasserstein" + str_i
        for dataset, label_cluster_i in itertools.product(datasets, label_clusters):
            mask = (df_joint["Dataset"] == dataset) & (df_joint["Label_Cluster"] == str(label_cluster_i))
            direct_opt_mask = mask & (df_joint["Transform"] == "DirectOptimization")
            min_wass_direct = df_joint.loc[direct_opt_mask, wass_str].values[0]
            # Normalize
            df_joint.loc[mask, wass_str] = df_joint.loc[mask, wass_str] / min_wass_direct

    # Exlude transforms with word "proxy" in their name for the plots
    df_joint_train = df_joint_train[~df_joint_train["Transform"].str.contains("proxy")]
    df_joint_test = df_joint_test[~df_joint_test["Transform"].str.contains("proxy")]
    transforms = [t for t in transforms if not "proxy" in t]

    # Drop rows with NaN values
    df_joint_train = df_joint_train.dropna().reset_index(drop=True)
    df_joint_test = df_joint_test.dropna().reset_index(drop=True)

    palette = sns.color_palette("husl", n_colors=len(transforms))
    palette = {transform: palette[i] for i, transform in enumerate(transforms)}

    for df_joint,str_i in zip([df_joint_train, df_joint_test], ["", " test"]):
        #Boxplot, where each box is the transform. One plot per metric, aggregated over datasets and label_clusters
        metrics = ["Min Wasserstein", "Min Lipschitz", "Domination percent"]
        metrics = [m + str_i for m in metrics]
        for metric in metrics:
            fig = plt.figure(figsize=(15, 10))
            ax = fig.gca()
            ax.grid(True)
            if "Wasserstein" in metric:
                sns.boxplot(data=df_joint[df_joint["Transform"] != "DirectOptimization"], x="Transform", y=metric, hue="Transform",
                            palette=palette, showfliers=False, ax=ax)
            else :
                sns.boxplot(data=df_joint, x="Transform", y=metric, hue="Transform",
                            palette=palette, showfliers=False, ax = ax)
            #sns.despine()
            ax.set_title(f"{metric} across datasets and label-clusters")
            fig.show()

    # Same plots, but with Friedman post-hoc test results
    for df_joint, str_i in zip([df_joint_train, df_joint_test], ["", " test"]):
        metrics = ["Min Wasserstein", "Min Lipschitz", "Domination percent"]
        metrics = [m + str_i for m in metrics]
        for metric in metrics:
            # Prepare data for Friedman test
            data = pd.DataFrame()
            for transform in transforms:
                subset = df_joint[df_joint["Transform"] == transform]
                data[transform] = subset[metric].reset_index(drop=True)
            posthoc_results = friedman_posthoc(data, correct="bergmann")
            fig = plt.figure(figsize=(10, 6))
            ax = fig.gca()
            ax.grid(True)
            sp.critical_difference_diagram(
                posthoc_results["summary_ranks"],
                posthoc_results["p_adjusted"],
                label_fmt_left="{label}",
                label_fmt_right="{label}",
                color_palette=palette,
                ax=ax,
            )
            ax.set_title(f"Critical Difference Diagram for {metric}")
            fig.show()









    raise ValueError("Stop here")
    # Iterate (within dataset -1) over all transforms and print the results for label_cluster 0,0
    dataset = datasets[1]
    for label_cluster_i in label_clusters:
        label, cluster = label_cluster_i
        to_plot = {}
        for transform in transforms:
            res = results_crude[dataset][transform][(label, cluster)]
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
