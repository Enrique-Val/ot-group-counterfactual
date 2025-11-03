import os
import platform
import re

import numpy as np
import pandas as pd


def list_params(root_dir, n_clusters=5, exp_type = "math_opt"):
    datasets = [i for i in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, i)) and not i.startswith('_')]
    data_dir = os.path.join(root_dir, datasets[0], str(n_clusters), exp_type)
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