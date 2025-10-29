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
    DiagonalAffine, SymmetricPSDAffine
from group_cfx.transforms.probabilistic_transforms import GMMForwardTransform, ProbabilisticTransform
from group_cfx.transforms.gaussian_transforms import GaussianTransform, GaussianCommutativeTransform, \
    GaussianScaleTransform
from utils import synthetic_2d, get_openml_dataset, train_lg, \
    cross_experiment, cross_experiment_pymoo


import joblib

import cvxpy as cv

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
                        choices=['FullAffine', 'FullAffine_proxy', 'SymmetricPSDAffine', 'SymmetricPSDAffine_proxy', 'DiagonalAffine',
                                 'DirectOptimization',
                                 'GaussianCommutativeTransform', 'GaussianTransform', 'GaussianScaleTransform',
                                 'GMMForwardTransform'])
    parser.add_argument('--math_opt', action='store_true', help='Use mathematical optimization')
    parser.add_argument('--only_train', action='store_true', help='Only train the classifier and density estimator')
    args = parser.parse_args()

    t0 = time.time()

    # Create output directory if it does not exist
    data_path = os.path.join(args.output_dir, f"data_{args.data_id}")
    models_path = os.path.join(data_path, "models")
    transform_path = os.path.join(data_path,str(args.n_clusters), "math_opt" if args.math_opt else "heuristic", args.transform)
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
    X_test, y_test = X[split_idx:], y[split_idx:]
    X, y = X[:split_idx], y[:split_idx]


    # ============================
    # Step 2: Torch classifier f and density estimator de
    # ============================

    device = "cpu"
    fet1 = 0
    fet2 = 1

    '''
    f = train_gbt(X, y, random_state = args.random_seed)
    # Select two most important features according to GBT feature importance
    important_features = np.argsort(f.feature_importances_)[-2:]
    fet1 = important_features[-1]
    fet2 = important_features[-2]
    '''

    # Load model if it exists
    if os.path.exists(os.path.join(models_path, "lg.pkl")) and not args.only_train:
        f = joblib.load(os.path.join(models_path, "lg.pkl"))
        print("Loaded model from", os.path.join(models_path, "lg.pkl"))
        print("Best validation accuracy:", f.score(X, y))

    else :
        # Train a logistic regression model
        scoring = "f1_macro"
        f, params, score = train_lg(X, y, scoring=scoring, random_state=args.random_seed, max_iter = 10000)

        # Pickle model to file
        joblib.dump(f, os.path.join(models_path, "lg.pkl"))

        # Save the training results (param and score) to a text file
        with open(os.path.join(models_path, "lg_params.txt"), "w") as file :
            file.write(f"Best validation {scoring}: {score}\n")
            file.write(f"Best parameters: {params}\n")

    if args.only_train :
        print("Only training the classifiers. Exiting.")
        exit(0)

    # Compute max and mins per feature
    xl = np.min(X)
    xu = np.max(X)


    # ============================
    # Step 3: Subgroup samples
    # ============================

    unique_labels = np.unique(y)

    n_cluster = args.n_clusters


    # Dataframe for the exec time (only if non linear, pyomo)
    if not args.math_opt :
        df_index = pd.MultiIndex.from_product([unique_labels, range(n_cluster)], names=['label', 'cluster'])
        df_time = pd.Series(index = df_index, name='exec_time')

    for label in unique_labels:
        y_orig = label
        y_prime = unique_labels[unique_labels != label][0]
        print("Y_orig =", y_orig, "y_prime =", y_prime)

        # Alternative: Find interesting groups to explain by applying clustering
        sub_data = X_test[y_test == label]

        '''
        # Remove also instances that are very close to the decision boundary
        if isinstance(f, torch.nn.Module):
            with torch.no_grad():
                logits = f(torch.tensor(sub_data, dtype=torch.float32))
                probs = torch.softmax(logits, dim=1)[:, 1].numpy()
        else:
            probs = f.predict_proba(sub_data)[:, 1]

        # Save the discarded instances for reference
        conf_selection = 0.5
        if y_prime == 0:
            probs = 1 - probs
        df_discarded = sub_data[probs >= 1 - conf_selection]
        sub_data = sub_data[probs < 1 - conf_selection]'''

        # Use kmeans clustering (sklearn)
        cluster_alg = KMeans(n_clusters=args.n_clusters, random_state=args.random_seed)
        cluster_labels = cluster_alg.fit_predict(sub_data)

        # Get "sub" datasets for each cluster
        X_sub_list = []
        for c in np.unique(cluster_labels):
            X_c = sub_data[cluster_labels == c]
            # If len X_c < 100, raise an Exception
            if X_c.shape[0] < 10:
                raise ValueError(f"Cluster {c} has less than 100 samples ({X_c.shape[0]} samples)")

            X_sub_list.append(torch.tensor(X_c, dtype=torch.float32))

        # Confidence for y_prime
        y_prime_conf = 0.9


        # ============================
        # Step 5: Solve and analyse
        # ============================
        for i, X_sub in enumerate(X_sub_list):
            transform = None
            if args.transform == 'FullAffine':
                transform = FullAffine(d, blp_proxy= False)
            elif args.transform == 'FullAffine_proxy':
                transform = FullAffine(d, blp_proxy= True)
            elif args.transform == 'SymmetricPSDAffine':
                transform = SymmetricPSDAffine(d, blp_proxy= False)
            elif args.transform == 'SymmetricPSDAffine_proxy':
                transform = SymmetricPSDAffine(d, blp_proxy= True)
            elif args.transform == 'DiagonalAffine':
                transform = DiagonalAffine(d)
            elif args.transform == 'DirectOptimization':
                transform = DirectOptimization(X_sub, xl, xu)
            elif args.transform == 'GaussianCommutativeTransform':
                transform = GaussianCommutativeTransform(d)
            elif args.transform == 'GaussianTransform':
                transform = GaussianTransform(d)
            elif args.transform == 'GaussianScaleTransform':
                transform = GaussianScaleTransform(d)
            elif args.transform == 'GMMForwardTransform':
                transform = GMMForwardTransform(d, n_components=3)
            else:
                raise ValueError("Unknown transform")

            if isinstance(transform, ProbabilisticTransform):
                transform.fit_prior(X_sub)
            transform.to(device)

            if args.math_opt and args.transform not in ['FullAffine', 'SymmetricPSDAffine', 'DiagonalAffine',
                                                      'GaussianCommutativeTransform', 'GaussianScaleTransform',
                                                      'DirectOptimization']:
                raise ValueError("Linear solver cannot be used with transform " + args.transform)

            if args.math_opt :
                solver = cv.MOSEK if not args.transform in ["DirectOptimization","FullAffine"] else "gurobi"
                K_list = [1.1, 1.5, 2.0, 2.5, 3.0]
                wass_list = []
                time_list = []
                for K in K_list :
                    wass, exec_time = cross_experiment(transform, X_sub, f, y_prime, y_prime_conf, K=K,
                                                 solver=solver)
                    wass_list.append(wass)
                    time_list.append(exec_time)
                # Create df and save to csv
                df_results = pd.DataFrame({'K': K_list, 'Wasserstein': wass_list, 'exec_time': time_list})
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
                        algorithm=NSGA2(pop_size=100, eliminate_duplicates=True),
                        termination=DefaultMultiObjectiveTermination(),
                        verbose=args.verbose,
                        min_acc=y_prime_conf
                    )
                else:
                    raise ValueError("Unknown solver")
                df_results, exec_time = cross_experiment_pymoo(transform, X_sub, f, y_prime, y_prime_conf, solver, random_seed= args.random_seed
                )
                # Store exec time
                index = (y_orig, i)
                df_time.loc[index] = exec_time



    # Save exec time dataframe (if math_opt is false)
    if not args.math_opt :
        df_time.to_csv(os.path.join(transform_path, "exec_times.csv"), index=True)
    print("Total exec time:", time.time() - t0)