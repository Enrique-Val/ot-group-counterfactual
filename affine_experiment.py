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
from torch import nn

from group_cfx.solver.pymoo_solver import PyMooSolver
from group_cfx.solver.sgd_solver import SGDSolver
from group_cfx.transforms.functional_transforms import FullAffine, LowRankAffine, SmallMLP, DirectOptimization
from group_cfx.transforms.probabilistic_transforms import GaussianNoScaleTransform, GaussianPolynomialTransform, \
    GaussianTransform, GMMForwardTransform, ProbabilisticTransform
from group_cfx.transforms.utils import bi_lipschitz_metric
from utils import synthetic_2d, train_classifier, print_plot_solutions, get_openml_dataset, train_gbt

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


if __name__ == "__main__":
    # Argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_id', type=int, default=-1, help='OpenML dataset ID')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    parser.add_argument('--output_dir', type=str, default='./results', help='Output directory')
    parser.add_argument('--device', type=str, default='cpu', help='Device to use (cpu or cuda)')
    parser.add_argument('--random_seed', type=int, default=0, help='Random seed for reproducibility')
    parser.add_argument('--n_clusters', type=int, default=5, help='Number of clusters for subgrouping')
    parser.add_argument('--transform', type=str, default='FullAffine', help='Type of transform to use',
                        choices=['FullAffine', 'LowRankAffine', 'SmallMLP', 'DirectOptimization',
                                 'GaussianPolynomialTransform', 'GaussianTransform', 'GMMForwardTransform'])
    args = parser.parse_args()

    # Create output directory if it does not exist
    dir_path = os.path.join(args.output_dir, str(args.n_clusters), f"data_{args.data_id}")
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


    # ============================
    # Step 1: Load dataset
    # ============================
    if args.data_id == -1:
        X,y = synthetic_2d(noise_scale=0.05, random_state=args.random_seed, size=1000)
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
    X, y = sklearn.utils.shuffle(X, y, random_state=args.random_seed)

    # ============================
    # Step 2: Torch classifier f and density estimator de
    # ============================

    device = "cpu"
    '''    #f = train_classifier(X, y, device = device)
    fet1 = 0
    fet2 = 1'''
    f = train_gbt(X, y, random_state = args.random_seed)
    # Select two most important features according to GBT feature importance
    important_features = np.argsort(f.feature_importances_)[-2:]
    fet1 = important_features[-1]
    fet2 = important_features[-2]

    # Train a Logistic Regression instead
    f = sklearn.linear_model.LogisticRegression()
    f.fit(X, y)

    # Subsample 1000 instances for density estimator
    X_de = X[:500, :]

    de = sklearn.neighbors.KernelDensity(kernel='gaussian', bandwidth=2)
    de.fit(X_de)
    '''print(np.mean(de.score_samples(X[1000:])))
    print(np.std(de.score_samples(X[1000:])))
    print(np.quantile(de.score_samples(X[1000:]), 0.75))
    print(np.quantile(de.score_samples(X[1000:]), 0.95))
    print(np.median(de.score_samples(X[1000:])))'''

    # Compute max and mins per feature
    xl = np.min(X)
    xu = np.max(X)

    # ============================
    # Step 3: Subgroup samples
    # ============================

    unique_labels = np.unique(y)

    n_cluster = 5


    # Dataframe for the exec time
    df_index = pd.MultiIndex.from_product([unique_labels, range(n_cluster)], names=['label', 'cluster'])
    df_time = pd.DataFrame(columns=['label', 'cluster', 'exec_time'], index = df_index)

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
        conf = 0.8
        df_discarded = sub_data[(probs >= 1-conf) & (probs <= conf)]
        sub_data = sub_data[(probs < 1-conf) | (probs > conf)]

        # Use kmeans clustering (sklearn)
        cluster_alg = KMeans(n_clusters=args.n_clusters, random_state=args.random_seed)
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

            args.transform = 'GaussianTransform'
            transform = None
            if args.transform == 'FullAffine':
                transform = FullAffine(d)
            elif args.transform == 'LowRankAffine':
                transform = LowRankAffine(d, r=1)
            elif args.transform == 'SmallMLP':
                transform = SmallMLP(d, hidden=16)
            elif args.transform == 'DirectOptimization':
                transform = DirectOptimization(X_sub, xl, xu, box_clip=True)
            elif args.transform == 'GaussianPolynomialTransform':
                transform = GaussianPolynomialTransform(d, n_components=3)
            elif args.transform == 'GaussianTransform':
                transform = GaussianTransform(d)
            elif args.transform == 'GMMForwardTransform':
                transform = GMMForwardTransform(d, n_components=3)
            else:
                raise ValueError("Unknown transform")

            if isinstance(transform, ProbabilisticTransform):
                transform.fit_prior(X_sub)
            transform.to(device)

            t0 = time.time()
            pv = transform.cvxpy_solving(X_sub, f, y_prime=y_prime, y_prime_confidence=0.8, K = 100)
            #pv = transform.pyomo_solving(X_sub, f, y_prime=y_prime, y_prime_confidence=0.8)
            tn = time.time()
            print("QP solving time:", tn - t0)

            if isinstance(transform, FullAffine):
                print("QP solution A:", transform.A)
            # Plot the affine transformation
            fig = plt.figure()
            ax = fig.gca()
            # Plot original points
            ax.scatter(X_sub[:, fet1], X_sub[:, fet2], color='blue', label='Original', alpha=0.5)
            # Plot transformed points
            with torch.no_grad():
                X_transformed = transform(X_sub.to(device)).cpu().numpy()
            ax.scatter(X_transformed[:, fet1], X_transformed[:, fet2], color='orange', label='Transformed', alpha=0.5)
            # Plot arrows
            for j in range(X_sub.shape[0]):
                ax.arrow(X_sub[j, fet1], X_sub[j, fet2], X_transformed[j, fet1] - X_sub[j, fet1],
                         X_transformed[j, fet2] - X_sub[j, fet2],
                         head_width=0.01, head_length=0.01, fc='gray', ec='gray', alpha=0.3)
            ax.set_title(f"QP solution for cluster {i} (transform {args.transform})")
            ax.legend()
            plt.show()

            raise ValueError("Stop here")

            t0 = time.time()
            X_prime, res_f, res_x = solver.solve(transform, f, de, X_sub, y_target=y_prime, seed=args.random_seed)
            tn = time.time()
            print("Solver time:", tn - t0)

            # Compute also the Lipschitz values
            lip_real = np.zeros(len(res_x))
            for j in range(len(res_x)):
                transform.load_parameters(res_x[j])
                with torch.no_grad():
                    lip = bi_lipschitz_metric(X_sub, transform(X_sub))
                # Convert to 1-Lipschitz
                lip_real[j] = lip
            res_f = np.hstack([res_f, lip_real.reshape(-1,1)])

            # Store results in a dataframe
            df_results = pd.DataFrame(res_f, columns=['Wasserstein', 'Lipschitz proxy', 'Lipschitz'])

            # Save to csv
            file_name = os.path.join(dir_path, f'results_transform_{args.transform}_label_{y_orig}_cluster_{i}.csv')
            #df_results.to_csv(file_name, index=False)

            # Store exec time
            df_time.loc[(y_orig, i), 'exec_time'] = tn - t0

            # Order solutions by f2 values
            order = np.argsort([i[1] for i in res_f])
            res_f = [res_f[i] for i in order]
            res_x = [res_x[i] for i in order]

            if args.verbose:
                print_plot_solutions(res_f, res_x, transform, X_sub, n_pics=4, x_lims=x_lims, y_lims=y_lims,
                                     fets =(fet1, fet2), exec_time = tn - t0)
            raise ValueError("Stop here")
    # Save exec time dataframe
    df_time.to_csv(os.path.join(dir_path, "exec_times.csv"), index=False)