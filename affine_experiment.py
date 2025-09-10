import argparse
import time

import numpy as np
import pandas as pd
import sklearn
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.termination import get_termination
from pymoo.operators.mutation.pm import PolynomialMutation
import torch
import matplotlib.pyplot as plt
import os

from pymoo.termination.default import DefaultMultiObjectiveTermination
from sklearn.cluster import KMeans


from group_cfx.solver.pymoo_solver import PyMooSolver
from group_cfx.solver.sgd_solver import SGDSolver
from group_cfx.transforms.functional_transforms import FullAffine, LowRankAffine, SmallMLP, DirectOptimization
from group_cfx.transforms.probabilistic_transforms import GaussianNoScaleTransform, GaussianScaleTransform, \
    GaussianTransform
from utils import synthetic_2d, train_classifier, print_plot_solutions, get_openml_dataset, train_gbt

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


if __name__ == "__main__":
    # Argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_id', type=int, default=-1, help='OpenML dataset ID')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    args = parser.parse_args()

    args.verbose = True

    # ============================
    # Step 1: Load dataset
    # ============================

    if args.data_id == -1:
        X,y = synthetic_2d(noise_scale=0.05)
        x_lims = 0, 1
        y_lims = 0, 1
    else :
        X,y,label_dict = get_openml_dataset(44091)
        x_lims = None, None
        y_lims = None, None
        # Standardize X
        X = (X - X.mean(axis=0)) / X.std(axis=0)
    d = X.shape[1]

    # Shuffle X and y
    X, y = sklearn.utils.shuffle(X, y, random_state=0)

    # ============================
    # Step 2: Torch classifier f and density estimator de
    # ============================

    device = "cpu"
    '''    #f = train_classifier(X, y, device = device)
    fet1 = 0
    fet2 = 1'''
    f = train_gbt(X, y)
    # Select two most important features according to GBT feature importance
    important_features = np.argsort(f.feature_importances_)[-2:]
    fet1 = important_features[-1]
    fet2 = important_features[-2]

    # Subsample 1000 instances for density estimator
    X_de = X[:500, :]

    de = sklearn.neighbors.KernelDensity(kernel='gaussian', bandwidth=2)
    de.fit(X_de)
    '''print(np.mean(de.score_samples(X[1000:])))
    print(np.std(de.score_samples(X[1000:])))
    print(np.quantile(de.score_samples(X[1000:]), 0.75))
    print(np.quantile(de.score_samples(X[1000:]), 0.95))
    print(np.median(de.score_samples(X[1000:])))'''

    # ============================
    # Step 3: Subgroup samples
    # ============================

    unique_labels = np.unique(y)

    for label in unique_labels:
        y_orig = label
        y_prime = unique_labels[unique_labels != label][0]

        # Alternative: Find interesting groups to explain by applying clustering
        sub_data = X[y == label]

        # Remove also instances that are very close to the decision boundary
        if isinstance(f, torch.nn.Module):
            with torch.no_grad():
                logits = f(torch.tensor(sub_data, dtype=torch.float32))
                probs = torch.softmax(logits, dim=1)[:, 1].numpy()
        else:
            probs = f.predict_proba(sub_data)[:, 1]

        # Save the discarded instances for reference
        conf = 0.9
        df_discarded = sub_data[(probs >= 1-conf) & (probs <= conf)]
        sub_data = sub_data[(probs < 1-conf) | (probs > conf)]

        # Use kmeans clustering (sklearn)
        cluster_alg = KMeans(n_clusters=5, random_state=0)
        cluster_labels = cluster_alg.fit_predict(sub_data)

        # Get "sub" datasets for each cluster
        X_sub_list = []
        for c in np.unique(cluster_labels):
            X_c = sub_data[cluster_labels == c]
            X_sub_list.append(torch.tensor(X_c, dtype=torch.float32))


        if args.verbose:
            # Plot original data, color by class (blue and red)
            fig = plt.figure()
            ax = fig.gca()
            sc = ax.scatter(X[:, fet1], X[:, fet2], c=['blue' if label == 0 else 'red' for label in y], alpha=0.5)
            ax.set_title("Original data by class")
            plt.show()


            # Plot the subgroups with different colors (do not use red)
            fig = plt.figure()
            ax = fig.gca()
            colors = ['blue', 'green', 'orange', 'purple', 'brown']
            for i, X_sub in enumerate(X_sub_list):
                ax.scatter(X_sub[:, fet1], X_sub[:, fet2], color=colors[i % len(colors)], label=f'Cluster {i}', alpha=0.5)
            # Plot the discarded instances in gray
            if len(df_discarded) > 0:
                ax.scatter(df_discarded[:, fet1], df_discarded[:, fet2], color='gray', label='Discarded', alpha=0.5)
            ax.set_title("Subgroups by clustering")
            ax.legend()
            plt.show()

        # ============================
        # Step 4: Choose  solver
        # ============================
        # Pick which solver to use:
        solver_name = "pymoo"   # or "sgd"

        if solver_name == "sgd":
            solver = SGDSolver(
                lr=1e-2,
                epochs=1000,
                alpha=1.0,
                beta=100.0,
                gamma=1e-3
            )

        elif solver_name == "pymoo":
            # Example with NSGA2
            mut = PolynomialMutation()
            solver = PyMooSolver(
                algorithm=NSGA2(pop_size=100, eliminate_duplicates=True, mutation=mut),
                termination= DefaultMultiObjectiveTermination(),
                verbose=args.verbose,
                min_acc=0.9
            )
        else :
            raise ValueError("Unknown solver")

        # ============================
        # Step 5: Solve and analyse
        # ============================
        for i, X_sub in enumerate(X_sub_list):
            transform = FullAffine(d)
            # transform = LowRankAffine(d, r=1)
            # transform = SmallMLP(d, hidden=16)
            # transform = DirectOptimization(X_sub, box_clip=True)
            transform = GaussianTransform(d)
            transform.fit_prior(X_sub)
            transform.to(device)

            t0 = time.time()
            X_prime, res_f, res_x = solver.solve(transform, f, de, X_sub, y_target=y_prime)
            tn = time.time()
            print("Solver time:", tn - t0)

            '''# Remove solutions with lower density
            # Sort by f3 (density)
            order = np.argsort([f3 for f1, f2, f3 in res_f])
            # Retain only half
            n_retain = max(1, len(res_f) // 2)
            res_f = [res_f[i] for i in order[:n_retain]]
            res_x = [res_x[i] for i in order[:n_retain]]'''

            # Order solutions by f2 values
            order = np.argsort([i[1] for i in res_f])
            res_f = [res_f[i] for i in order]
            res_x = [res_x[i] for i in order]

            if args.verbose:
                print_plot_solutions(res_f, res_x, transform, X_sub, n_pics=4, x_lims=x_lims, y_lims=y_lims, fets =(fet1, fet2))
