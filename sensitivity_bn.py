import argparse
import time

import numpy as np
import pandas as pd
import sklearn
from pymoo.algorithms.moo.nsga2 import NSGA2
import torch
import matplotlib.pyplot as plt
import os

from pymoo.termination.default import DefaultMultiObjectiveTermination
from sklearn.cluster import KMeans

from group_cfx.solver.pymoo_solver import PyMooSolver
from group_cfx.transforms.functional_transforms import FullAffine, DirectOptimization, \
    DiagonalAffine, PSDAffine
from group_cfx.transforms.probabilistic_transforms import GMMForwardTransform, ProbabilisticTransform
from group_cfx.transforms.gaussian_transforms import GaussianTransform, GaussianCommutativeTransform, \
    GaussianScaleTransform
from utils import synthetic_2d, get_openml_dataset, train_lg, \
    cross_experiment, cross_experiment_pymoo, get_transform

from sklearn_extra.cluster import KMedoids

import psutil, gc


import joblib

import cvxpy as cv

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

def print_mem(prefix=""):
    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss / 1e6  # MB
    print(f"{prefix} memory: {mem:.2f} MB")


if __name__ == "__main__":
    # Argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_id', type=int, default=-1, help='OpenML dataset ID')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    parser.add_argument('--output_dir', type=str, default='./results', help='Output directory')
    parser.add_argument('--device', type=str, default='cpu', help='Device to use (cpu or cuda)')
    parser.add_argument('--random_seed', type=int, default=0, help='Random seed for reproducibility')
    parser.add_argument('--n_clusters', type=int, default=10, help='Number of clusters for subgrouping')
    parser.add_argument('--transform', type=str, default='FullAffine', help='Type of transform to use',
                        choices=['FullAffine', 'FullAffine_proxy', 'PSDAffine', 'PSDAffine_proxy', 'DiagonalAffine',
                                 'DirectOptimization',
                                 'GaussianCommutativeTransform', 'GaussianTransform', 'GaussianTransform_proxy', 'GaussianScaleTransform',
                                 'GMMForwardTransform'])
    args = parser.parse_args()

    np.random.seed(args.random_seed)

    args.data_id = -1
    args.transform = 'DirectOptimization'
    args.verbose = True

    t0 = time.time()

    # Create output directory if it does not exist
    data_path = os.path.join(args.output_dir, f"data_{args.data_id}")
    models_path = os.path.join(data_path, "models")
    cluster_path = os.path.join(data_path, str(args.n_clusters))
    transform_path = os.path.join(cluster_path, "math_opt" if args.math_opt else "heuristic", args.transform)
    if not os.path.exists(transform_path):
        os.makedirs(transform_path)
    if not os.path.exists(models_path):
        os.makedirs(models_path)


    # ============================
    # Step 1: Load dataset
    # ============================
    if args.data_id == -1:
        X,y = synthetic_2d(noise_scale=0.05, random_state=args.random_seed, size=1000)
        x_lims = 0, 1
        y_lims = 0, 1
    else :
        X,y,label_dict = get_openml_dataset(args.data_id)
        x_lims = None, None
        y_lims = None, None
        # Standardize X
        X = (X - X.mean(axis=0)) / X.std(axis=0)
        # Min-max scale to 0-1
        #X = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
    d = X.shape[1]

    # Shuffle X and y
    X, y = sklearn.utils.shuffle(X, y, random_state=args.random_seed)
    # Divide into train and test
    split_idx = int(0.2 * X.shape[0])
    X_test, y_test = X[:split_idx], y[:split_idx]
    X, y = X[split_idx:], y[split_idx:]


    # ============================
    # Step 2: Torch classifier f and density estimator de
    # ============================

    device = "cpu"

    # Load model if it exists
    if os.path.exists(os.path.join(models_path, "lg.pkl")):
        f = joblib.load(os.path.join(models_path, "lg.pkl"))
        print("Loaded model from", os.path.join(models_path, "lg.pkl"))
        print("Best validation accuracy:", f.score(X, y))

    else :
        # Model was not trained
        raise Exception("No model found")

    # Compute max and mins per feature
    xl = np.min(X)
    xu = np.max(X)


    # ============================
    # Step 3: Subgroup samples
    # ============================

    unique_labels = np.unique(y)

    n_cluster = args.n_clusters
    y_orig = 0
    y_prime = unique_labels[unique_labels != y_orig][0]
    cluster_id = 0
    cluster_alg_dir = os.path.join(cluster_path, "label_"+str(y_orig)+".pkl")


    print("Y_orig =", y_orig, "y_prime =", y_prime)

    # Alternative: Find interesting groups to explain by applying clustering
    sub_data = X_test[y_test == y_orig]

    if not os.path.exists(cluster_alg_dir):
        raise ValueError("No clustering algorithm found")
    cluster_alg = joblib.load(cluster_alg_dir)
    cluster_labels = cluster_alg.predict(sub_data)

    X_sub = sub_data[cluster_labels == cluster_id]
    # Limit to 200 instances for testing
    X_sub = X_sub[:200]
    min_samples = 20
    if X_sub.shape[0] < min_samples:
        # Warn the user and introduce synthetic samples by jittering up to min_samples
        print(f"Warning: Cluster {cluster_id} has less than 20 samples ({X_sub.shape[0]} samples). Augmenting data by jittering.")
        n_needed = min_samples - X_sub.shape[0]
        jittered_samples = X_sub[np.random.choice(X_sub.shape[0], n_needed, replace=True)] + np.random.normal(0, 0.01, size=(n_needed, X_c.shape[1]))
        X_sub = np.vstack([X_sub, jittered_samples])

    # Confidence for y_prime
    y_prime_conf = 0.8

    # ============================
    # Step 5: Solve and analyse
    # ============================
    transform = get_transform(args.transform, X_sub, device = "cpu")


    solver = PyMooSolver(
        algorithm=NSGA2(pop_size=100, eliminate_duplicates=True, verbose =args.verbose, seed=args.random_seed),
        termination=DefaultMultiObjectiveTermination(),
        verbose=args.verbose,
        min_acc=y_prime_conf
    )

    X_prime, res_f, res_x = solver.solve(transform, f, X_sub, y_prime=y_prime,
                                         y_prime_confidence=y_prime_conf, seed=args.random_seed)

    # TODO
    # Train bn with res_x

    print("Total exec time:", time.time() - t0)