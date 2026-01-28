import os
import platform
import re

import numpy as np
import pandas as pd
from pyomo.contrib.parmest.graphics import sns

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


def list_params(root_dir, n_clusters=5, exp_type = "math_opt"):
    datasets = [i for i in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, i)) and not i.startswith('_') and not i == "plots"]
    data_dir = os.path.join(root_dir, datasets[1], str(n_clusters), exp_type)
    transforms = []
    for transform in os.listdir(data_dir):
        if os.path.isdir(os.path.join(data_dir, transform)):
            transforms.append(transform)
    transform_dir = os.path.join(data_dir, transforms[0])
    label_cluster_list = []
    for transform in os.listdir(transform_dir):
        # Files have the shape label_{int}_cluster_{int}.csv
        # Use regular expression to extract label and cluster
        match = re.match(r"label_(\d+)_cluster_(\d+).csv", transform)
        if match:
            label = int(match.group(1))
            cluster = int(match.group(2))
            label_cluster_list.append((label, cluster))
    return datasets, transforms, label_cluster_list

def load_results(root_dir, datasets, transforms, label_cluster, n_clusters=5, exp_type = "math_opt") :
    results = {}
    for dataset in datasets:
        results[dataset] = {}
        data_dir = os.path.join(root_dir, dataset, str(n_clusters), exp_type)
        for transform in transforms:
            results[dataset][transform] = {}
            transform_dir = os.path.join(data_dir, transform)
            for label, cluster in label_cluster:
                file_path = os.path.join(transform_dir, f"label_{label}_cluster_{cluster}.csv")
                if os.path.exists(file_path):
                    results[dataset][transform][(label, cluster)] = pd.read_csv(file_path)
                else:
                    results[dataset][transform][(label, cluster)] = None
    return results

def friedman_posthoc(data, correct="bergmann", eps = 1e-5) -> dict[str, pd.DataFrame | pd.Series]:
    '''
    Perform the Friedman test and the Bermann-Hommel post-hoc test using the scmamp package in R

    Parameters
    ----------
    data : pandas.DataFrame
        A pandas DataFrame where each column is a different outcome to test and each row is a different instance.
    correct : str
        String indicating the correction method to use for the p-values. The possible values are: "shaffer", "bergmann",
         "holland", "finner", "rom" and "li"

    Returns
    -------
    dict
        A dictionary containing the summary statistics of the post-hoc test. The dictionary contains the following keys:
        - "summary": A pandas Series containing the summary statistics of the post-hoc test.
        - "p_values": A pandas DataFrame containing the p-values of the Friedman test.
        - "p_adjusted": A pandas DataFrame containing the adjusted p-values of the Bergmann-Hommel post-hoc test.
    '''

    from rpy2.robjects import pandas2ri, conversion
    import rpy2.robjects.packages as rpackages
    import rpy2.robjects as ro


    # Import the scmamp package from R
    if platform.system() == 'Windows':
        r_lib_path = os.path.expanduser('~/AppData/Local/R/win-library/4.3').replace("\\", "/")
    else:
        r_lib_path = os.path.expanduser('~/R/x86_64-pc-linux-gnu-library/4.4')
    scmamp = rpackages.importr('scmamp', lib_loc=r_lib_path)
    base = rpackages.importr("base")

    # Explicit conversion context
    with conversion.localconverter(ro.default_converter + pandas2ri.converter):
        r_data = ro.conversion.py2rpy(data)

    # Perform the post-hoc test in R using scmamp::postHocTest
    bh_posthoc_scmamp = scmamp.postHocTest(r_data, test="friedman", correct=correct)

    # Convert the rpy2 ListVector to a Python dictionary
    d = len(data.columns)

    def rmat_to_df(r_obj):
        # Convert possibly vector-like R object to full matrix
        mat = np.array(base.as_matrix(r_obj))
        mat = mat.reshape((d, d)) if mat.size == d * d else np.full((d, d), np.nan)
        return pd.DataFrame(mat, index=data.columns, columns=data.columns)

    bh_posthoc = {}
    summary = pd.Series(bh_posthoc_scmamp[0][0], index=data.columns)
    bh_posthoc["summary"] = summary
    bh_posthoc["summary_ranks"] = data.rank("columns").mean(axis=0)
    bh_posthoc["p_values"] = rmat_to_df(bh_posthoc_scmamp[1]).fillna(1.0)
    bh_posthoc["p_adjusted"] = rmat_to_df(bh_posthoc_scmamp[2]).fillna(1.0)
    # If there are values smaller than eps, set them to eps
    bh_posthoc["p_adjusted"] = bh_posthoc["p_adjusted"].clip(lower=eps)

    return bh_posthoc


def plot_performance_profile(df, metric, title=None, ax=None, palette=None, max_x=5, verbose=False, maximize = False):
    """
    Generates a Dolan-More Performance Profile.

    Args:
        df: DataFrame containing columns ['dataset', 'label_cluster', 'K', 'transform', 'value']
            (Must be the RAW dataframe, including DirectOptimization)
        metric: The specific metric to filter for (e.g., "Time" or "Wasserstein test")
        title: Optional title for the plot
        ax: Optional matplotlib axis to plot on
    """
    if verbose :
        print("Generating performance profile for metric:", metric)
    # Filter for the specific metric
    df_subset = df[df["metric"] == metric].copy()

    # Substitue nans for a large number (worse performance)
    # Compute ratio of nan values
    n_nans = df_subset["value"].isna().sum()
    total = len(df_subset)
    print(f"Substituting {n_nans} NaN values out of {total} ({(n_nans/total)*100:.2f}%) for metric {metric}")
    df_subset["value"] = df_subset["value"].fillna(df_subset["value"].max() * 100)

    # 1. Identify the BEST value for each problem instance
    # A "problem instance" is defined by (Dataset + Cluster + K) or just (Dataset + label_cluster)
    possible_group_cols = ["dataset", "label_cluster", "K"]
    group_cols = [c for c in possible_group_cols if c in df_subset.columns]

    problem_groups = df_subset.groupby(group_cols)

    if maximize:
        # Best = Max value
        best_values = df_subset.groupby(group_cols)["value"].transform("max")
    else:
        # Best = Min value
        best_values = df_subset.groupby(group_cols)["value"].transform("min")

        # --- 3. Calculate Performance Ratio (tau) ---
        # The best method always gets a ratio of 1.0. Worse methods get > 1.0.

    if maximize:
        # Formula: Best / Value
        # Example: Best=10, You=2. Ratio = 10/2 = 5 (You are 5x worse)
        # Protect against division by zero if value is 0
        df_subset["ratio"] = best_values / df_subset["value"].replace(0, 1e-10)
    else:
        # Formula: Value / Best
        # Example: Best=2, You=10. Ratio = 10/2 = 5 (You are 5x worse)
        df_subset["ratio"] = df_subset["value"] / best_values.replace(0, 1e-10)

    # 3. Plotting
    if ax is None:
        fig, ax = plt.subplots(figsize=fig_size)

    transforms = df_subset["transform"].unique()
    transforms = [t for t in plot_order if t in transforms]

    # --- STYLE DEFINITIONS ---
    # Cycle through these to distinguish lines beyond just color
    line_styles = ['-', '--', '-.', ':']
    # Distinct markers (Circle, Square, Triangle Up, Diamond, Triangle Down, X, Plus, Star)
    markers = ['o', 's', '^', 'D', 'v', 'X', 'P', '*']

    for i,t in enumerate(transforms):
        # Extract ratios for this specific transform
        t_ratios = df_subset[df_subset["transform"] == t]["ratio"]

        if len(t_ratios) == 0:
            continue

        # Sort ratios to calculate the Cumulative Distribution Function (CDF)
        sorted_ratios = np.sort(t_ratios)
        yvals = np.arange(1, len(sorted_ratios) + 1) / len(sorted_ratios)

        if verbose:
            print(f"Transform: {t}, Number of instances: {len(sorted_ratios)}")
            # Print every k (x,y) pairs
            k = max(1, len(sorted_ratios) // 10)
            for xi, yi in zip(sorted_ratios[::k], yvals[::k]):
                print(f"  Ratio: {xi:.4f}, CDF: {yi:.4f}")


        # --- DYNAMIC STYLING ---
        # 1. Get Color
        color = palette.get(t, None) if palette else None

        # 2. Get Linestyle (Cycle through the 4 types)
        ls = line_styles[i % len(line_styles)]

        # 3. Get Marker (Cycle through types)
        mk = markers[i % len(markers)]

        # 4. Calculate 'markevery'
        # This ensures we only see ~5 markers per line, preventing clutter
        n_points = len(sorted_ratios)
        mark_stride = max(1, n_points // 5)

        # Plot step function
        ax.step(
            sorted_ratios,
            yvals,
            where='post',
            label=t,
            linewidth=2,
            color=color,
            linestyle=ls,  # <--- Adds pattern
            marker=mk,  # <--- Adds symbol
            markevery=mark_stride,  # <--- Prevents clutter
            markersize=6,  # <--- Readable size
            alpha=0.9  # <--- Slight transparency for overlapping lines
        )
    # 4. Formatting
    ax.set_xscale("log")
    # Uncomment below to force integer ticks on x-axis
    #ax.xaxis.set_minor_formatter(mticker.ScalarFormatter())
    #ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.set_xlim(1, max_x)  # Limits x-axis to 100x the cost of the best method
    ax.set_xlabel(f"Performance ratio (relative to best method)")
    ax.set_ylabel(f"Fraction of problems solved")
    if title is not None:
        ax.set_title(title)
    ax.legend(title="Method", loc="lower right")
    ax.grid(True, which="both", linestyle="--", alpha=0.5)

    # Remove top and right spines for cleaner look
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    return ax


renaming = {"DiagonalAffine" : "Diag. \n affine", "GaussianCommutativeTransform" : "Comm. \n Gaussian",
                "GaussianTransform" : "Any \n Gaussian", "PSDAffine" : "PSD \n affine", "GMMForwardTransform" : "3-GMM",
                "DirectOptimization" : "Group w/ \n bi-Lipschitz", "FullAffine" : "Any \n affine",
             "GaussianScaleTransform" : "Scaled \n Gaussian",
            "DirectOptimization_nb" : "Group w/ \n Lipschitz", "Wachter" : "Independent" }

plot_order = ["Independent", "Group w/ \n Lipschitz", "Group w/ \n bi-Lipschitz", "PSD \n affine", "Diag. \n affine", "Any \n Gaussian", "Comm. \n Gaussian", "Scaled \n Gaussian" , "3-GMM"]

renaming_nb = {"DiagonalAffine" : "Diag. affine", "GaussianCommutativeTransform" : "Comm. Gaussian",
                "GaussianTransform" : "Any Gaussian", "PSDAffine" : "PSD affine", "GMMForwardTransform" : "3-GMM",
                "DirectOptimization" : "Group w/ bi-Lipschitz", "FullAffine" : "Any affine",
                "GaussianScaleTransform" : "Scaled Gaussian",
               "DirectOptimization_nb": "Group w/ Lipschitz", "Wachter": "Independent"}

plot_order_bn = ["Independent", "Group w/ Lipschitz", "Group w/ bi-Lipschitz", "PSD affine", "Diag. affine", "Any Gaussian", "Comm. Gaussian", "Scaled Gaussian" , "3-GMM"]

renaming = renaming_nb
plot_order = plot_order_bn

palette = sns.color_palette("deep", len(plot_order))
palette = {t: palette[i] for i, t in enumerate(plot_order)}

fig_size = np.array((6.75*0.7,3.5*0.7))