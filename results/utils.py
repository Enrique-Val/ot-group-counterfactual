import os
import re

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