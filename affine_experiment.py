import argparse
import time

import numpy as np
import pandas as pd
import sklearn
from pymoo.algorithms.moo.nsga2 import NSGA2
import torch
import os

from pymoo.termination.default import DefaultMultiObjectiveTermination

from group_cfx.solver.pymoo_solver import PyMooSolver
from utils import synthetic_2d, get_openml_dataset, train_lg, \
    cross_experiment, cross_experiment_pymoo, get_transform, get_groups, train_gbt, train_mlp

from sklearn_extra.cluster import KMedoids


import joblib

import cvxpy as cv

import warnings

# Suppress the specific CPP backend warning from cvxpy
warnings.filterwarnings(
    "ignore",
    message="The problem includes expressions that don't support CPP backend.*"
)

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


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
                                 'DirectOptimization', 'Wachter', 'DirectOptimization_nb',
                                 'GaussianCommutativeTransform', 'GaussianTransform', 'GaussianTransform_proxy', 'GaussianScaleTransform',
                                 'GMMForwardTransform'])
    parser.add_argument('--math_opt', action='store_true', help='Use mathematical optimization')
    parser.add_argument('--only_train', action='store_true', help='Only train the classifier and density estimator')
    parser.add_argument('--model', type=str, default='lg', help='Model to use for the classifier (default: logistic regression)',
                        choices=['lg', 'gbt','mlp'])
    args = parser.parse_args()

    np.random.seed(args.random_seed)

    t0 = time.time()

    # Create output directory if it does not exist
    data_path = os.path.join(args.output_dir, f"data_{args.data_id}")
    models_path = os.path.join(data_path, "models")
    cluster_path = os.path.join(data_path,str(args.n_clusters))
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
        X,y,label_dict,_ = get_openml_dataset(args.data_id)
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
    if os.path.exists(os.path.join(models_path, args.model + ".pkl")) and not args.only_train:
        f = joblib.load(os.path.join(models_path, args.model + ".pkl"))
        print("Loaded model from", os.path.join(models_path, args.model + ".pkl"))
        print("Best validation accuracy:", f.score(X, y))

    else :
        # Train a logistic regression model
        scoring = "neg_log_loss"
        if args.model == "lg" :
            f, params, score = train_lg(X, y, scoring=scoring, random_state=args.random_seed, max_iter = 10000)
        elif args.model == "gbt" :
            f, params, score = train_gbt(X, y, scoring=scoring, random_state=args.random_seed)
        elif args.model == "mlp" :
            f, params, score = train_mlp(X, y, scoring=scoring, random_state=args.random_seed)
        else :
            raise ValueError("Unknown model")

        # Pickle model to file
        joblib.dump(f, os.path.join(models_path, args.model + ".pkl"))

        # Save the training results (param and score) to a text file
        with open(os.path.join(models_path, args.model + "_params.txt"), "w") as file :
            file.write(f"{params}\n")
        with open(os.path.join(models_path, args.model + "_"+str(scoring)+".txt"), "w") as file :
            file.write(f"{score}\n")

    # Compute max and mins per feature
    xl = np.min(X)
    xu = np.max(X)


    # ============================
    # Step 3: Subgroup samples
    # ============================

    unique_labels = np.unique(y)

    cluster_alg_dir_list = []

    for y_orig in unique_labels:
        # Check if the clustering algorithm is saved
        cluster_alg_dir = os.path.join(cluster_path, "label_" + str(y_orig) + ".pkl")
        cluster_alg_dir_list.append(cluster_alg_dir)
        if not os.path.exists(cluster_alg_dir) or args.only_train :
            # Alternative: Find interesting groups to explain by applying clustering
            sub_data = X_test[y_test == y_orig]

            # Use kmeans clustering (sklearn)
            # If there are more than 20k instances, train only with the first 20k and predict the rest
            if sub_data.shape[0] > 20000:
                cluster_alg = KMedoids(n_clusters=args.n_clusters, random_state=args.random_seed)
                cluster_alg.fit(sub_data[:20000])
            else:
                cluster_alg = KMedoids(n_clusters=args.n_clusters, random_state=args.random_seed)
                cluster_alg.fit(sub_data)
            # Save the cluster algorithm, pickling it
            joblib.dump(cluster_alg, cluster_alg_dir)

    if args.only_train :
        print("Only training the classifiers. Exiting.")
        exit(0)

    # Dataframe for the exec time (only if non linear, pyomo)
    if not args.math_opt :
        df_index = pd.MultiIndex.from_product([unique_labels, range(args.n_clusters)], names=['label', 'cluster'])
        df_time = pd.Series(index = df_index, name='exec_time')

    for y_orig,cluster_alg_dir in zip(unique_labels, cluster_alg_dir_list):
        # Load clustering algorithm
        cluster_alg = joblib.load(cluster_alg_dir)

        y_prime = unique_labels[unique_labels != y_orig][0]
        seggregated_data = X_test[y_test == y_orig]
        # Label instances
        cluster_labels = cluster_alg.predict(seggregated_data)
        # Get "sub" datasets for each cluster
        X_groups_list = get_groups(seggregated_data, cluster_alg)

        # Confidence for y_prime
        y_prime_conf = 0.8


        # ============================
        # Step 5: Solve and analyse
        # ============================
        for i, X_sub in enumerate(X_groups_list):
            transform = get_transform(args.transform, X_sub, xl = xl, xu=xu, device = "cpu")

            if args.math_opt :
                solver = cv.MOSEK if transform.is_cvx() else "gurobi"
                K_list = [1.01,1.5,2.0,3.5,5.0,7.5,10]
                # First, check if transform is Wachter. If so, it has no param K
                # Launch experiment only once and replicate for different K values
                if args.transform == "Wachter":
                    lenK = len(K_list)
                    wass, _, low_lip, _, up_lip, _, _, exec_time = cross_experiment(transform, X_sub, f, y_prime, y_prime_conf,
                                                 solver=solver, K=None)
                    print("Exec time label", y_orig, "cluster", i, "Wachter:", exec_time, "seconds")
                    # Create df and save to csv
                    df_results = pd.DataFrame({'K' : K_list, 'Wasserstein': [wass]*lenK, 'Wasserstein test': [wass]*lenK,
                                               'Empirical lower bound': [low_lip]*lenK,
                                               'Empirical lower bound test': [low_lip]*lenK,
                                               'Empirical upper bound': [up_lip]*lenK,
                                               'Empirical upper bound test': [up_lip]*lenK,
                                               'Validity test' : [1.0]*lenK,
                                               'Exec time': [exec_time]*lenK})
                    df_results.to_csv(os.path.join(transform_path, f'label_{y_orig}_cluster_{i}.csv'), index=False)
                else :
                    wass_list = []
                    wass_test_list = []
                    time_list = []
                    low_lip_list = []
                    low_lip_test_list = []
                    up_lip_list = []
                    up_lip_test_list = []
                    validity_list = []
                    prev_K = None
                    replicate = False
                    for K in K_list :
                        if replicate:
                            # Replicate previous results to save time
                            wass_list.append(wass_list[-1])
                            wass_test_list.append(wass_test_list[-1])
                            low_lip_list.append(low_lip_list[-1])
                            low_lip_test_list.append(low_lip_test_list[-1])
                            up_lip_list.append(up_lip_list[-1])
                            up_lip_test_list.append(up_lip_test_list[-1])
                            validity_list.append(validity_list[-1])
                            time_list.append(time_list[-1])
                            print("Replicated results for label", y_orig, "cluster", i, "K", K)
                            prev_K = K
                            continue

                        wass, wass_test, low_lip, low_lip_test, up_lip, up_lip_test, validity, exec_time = cross_experiment(transform, X_sub, f, y_prime, y_prime_conf, K=K,
                                                     solver=solver)
                        wass_list.append(wass)
                        low_lip_list.append(low_lip)
                        up_lip_list.append(up_lip)
                        validity_list.append(validity)
                        if wass_test is None or low_lip_test is None or up_lip_test is None:
                            wass_test_list.append(wass)
                            low_lip_test_list.append(low_lip)
                            up_lip_test_list.append(up_lip)
                        else :
                            wass_test_list.append(wass_test)
                            low_lip_test_list.append(low_lip_test)
                            up_lip_test_list.append(up_lip_test)
                        time_list.append(exec_time)
                        if args.transform == "DirectOptimization" and prev_K is not None and wass_list[-1] is not None and wass_list[-1] < 1000 and prev_K > 1/low_lip_list[-1] and K > up_lip_list[-1]:
                            replicate = True
                        prev_K = K
                        print("Exec time label", y_orig, "cluster", i, "K", K, "Method", args.transform,":", exec_time, "seconds")
                    # Create df and save to csv
                    df_results = pd.DataFrame({'K': K_list, 'Wasserstein': wass_list, 'Wasserstein test': wass_test_list,
                                               'Empirical lower bound': low_lip_list,
                                               'Empirical lower bound test': low_lip_test_list,
                                               'Empirical upper bound': up_lip_list,
                                               'Empirical upper bound test': up_lip_test_list,
                                               'Validity test': validity_list,
                                               'Exec time': time_list})
                    df_results.to_csv(os.path.join(transform_path, f'label_{y_orig}_cluster_{i}.csv'), index=False)
            # Non linear using Pymoo
            else :
                # Pick which solver to use:
                solver_name = "pymoo"  # or "sgd"

                if solver_name == "sgd":
                    raise NotImplementedError("SGD solver not implemented for this experiment")

                elif solver_name == "pymoo":
                    # Example with NSGA2
                    solver = PyMooSolver(
                        algorithm=NSGA2(pop_size=100, eliminate_duplicates=True, verbose =args.verbose,
                                        seed=args.random_seed),
                        termination=DefaultMultiObjectiveTermination(),
                        verbose=args.verbose,
                        min_acc=y_prime_conf
                    )
                else:
                    raise ValueError("Unknown solver")
                df_results, exec_time = cross_experiment_pymoo(transform, X_sub, f, y_prime, y_prime_conf, solver, random_seed= args.random_seed
                )
                print("Solved label", y_orig, "cluster", i, "in", exec_time, "seconds")
                # Save results to csv
                df_results.to_csv(os.path.join(transform_path, f'label_{y_orig}_cluster_{i}.csv'), index=False)
                # Store exec time
                index = (y_orig, i)
                df_time.loc[index] = exec_time



    # Save exec time dataframe (if math_opt is false)
    if not args.math_opt :
        df_time.to_csv(os.path.join(transform_path, "exec_times.csv"), index=True)
    print("Total exec time:", time.time() - t0)