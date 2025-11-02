import os
import time
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import torch
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.termination.default import DefaultMultiObjectiveTermination
from sklearn.cluster import KMeans
import sklearn.utils
import joblib
import cvxpy as cv

from group_cfx.solver.pymoo_solver import PyMooSolver
from group_cfx.solver.sgd_solver import SGDSolver
from group_cfx.transforms.functional_transforms import FullAffine, DiagonalAffine, PSDAffine, DirectOptimization
from group_cfx.transforms.gaussian_transforms import GaussianCommutativeTransform, GaussianTransform, \
    GaussianScaleTransform
from group_cfx.transforms.probabilistic_transforms import GMMForwardTransform, ProbabilisticTransform
from group_cfx.transforms.utils import bi_lipschitz_metric
from utils import synthetic_2d, get_openml_dataset, train_lg, print_plot_solutions

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
                        choices=['FullAffine', 'PSDAffine' 'LowRankAffine', 'SmallMLP', 'DirectOptimization',
                                 'GaussianCommutativeTransform', 'GaussianTransform', 'GMMForwardTransform'])
    args = parser.parse_args()

    args.data_id = 44127
    args.transform = 'GaussianTransform'
    exact = True

    # Create output directory if it does not exist
    data_path = os.path.join(args.output_dir, f"data_{args.data_id}")
    models_path = os.path.join(data_path, "models")
    transform_path = os.path.join(data_path,str(args.n_clusters), "math_opt" if exact else "heuristic", args.transform)
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

    # Leave 20% of data for testing
    n_test = int(0.2 * X.shape[0])
    X_test, y_test = X[:n_test], y[:n_test]
    X, y = X[n_test:], y[n_test:]

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
    if os.path.exists(os.path.join(models_path, "lg.pkl")) :
        f = joblib.load(os.path.join(models_path, "lg.pkl"))
        print("Loaded model from", os.path.join(models_path, "lg.pkl"))
        print("Best validation accuracy:", f.score(X, y))

    else :
        # Train a logistic regression model
        scoring = "f1_macro"
        f, params, score = train_lg(X, y, scoring=scoring, random_state=args.random_seed, max_iter = 1000)




    # Subsample 1000 instances for density estimator
    '''
    X_de = X[:500, :]

    de = sklearn.neighbors.KernelDensity(kernel='gaussian', bandwidth=2)
    de.fit(X_de)
    print(np.mean(de.score_samples(X[1000:])))
    print(np.std(de.score_samples(X[1000:])))
    print(np.quantile(de.score_samples(X[1000:]), 0.75))
    print(np.quantile(de.score_samples(X[1000:]), 0.95))
    print(np.median(de.score_samples(X[1000:])))'''
    de = None

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
        print("Y_orig =", y_orig, "y_prime =", y_prime)

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
        conf = 0.7
        if y_prime == 0:
            probs = 1 - probs
        df_discarded = sub_data[probs >= 1 - conf]
        sub_data = sub_data[probs < 1 - conf]

        # Use kmeans clustering (sklearn)
        cluster_alg = KMeans(n_clusters=args.n_clusters, random_state=args.random_seed)
        cluster_labels = cluster_alg.fit_predict(sub_data)

        # Get "sub" datasets for each cluster
        X_sub_list = []
        for c in np.unique(cluster_labels):
            X_c = sub_data[cluster_labels == c]
            X_sub_list.append(torch.tensor(X_c, dtype=torch.float32))

        # Do the same for the test data
        # First, filter by label
        sub_data_test = X_test[y_test == label]
        # Remove also instances that are very close to the decision boundary
        if isinstance(f, torch.nn.Module):
            with torch.no_grad():
                logits = f(torch.tensor(sub_data_test, dtype=torch.float32))
                probs = torch.softmax(logits, dim=1)[:, 1].numpy()
        else:
            probs = f.predict_proba(sub_data_test)[:, 1]
        if y_prime == 0:
            probs = 1 - probs
        sub_data_test = sub_data_test[probs < 1 - conf]

        cluster_labels_test = cluster_alg.predict(sub_data_test)
        X_sub_list_test = []
        for c in np.unique(cluster_labels_test):
            X_c = sub_data_test[cluster_labels_test == c]
            X_sub_list_test.append(torch.tensor(X_c, dtype=torch.float32))



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


            # Visualize also the test clusters
            # Plot the subgroups with different colors (do not use red)
            fig = plt.figure()
            ax = fig.gca()
            colors = ['blue', 'green', 'orange', 'purple', 'brown']
            for i, X_sub in enumerate(X_sub_list_test):
                ax.scatter(X_sub[:, fet1], X_sub[:, fet2], color=colors[i % len(colors)], label=f'Cluster {i}',
                           alpha=0.5)
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
            solver = PyMooSolver(
                algorithm=NSGA2(pop_size=100, eliminate_duplicates=True),
                termination= DefaultMultiObjectiveTermination(n_max_gen=2000),
                verbose=args.verbose,
                min_acc=0.9
            )
        elif solver_name == "pymoolm":
            solver = PyMooLMSolver(
                algorithm=NSGA2(pop_size=100, eliminate_duplicates=True),
                termination= DefaultMultiObjectiveTermination(),
                verbose=args.verbose,
                min_acc=0.9,
                K = 1.1
            )
        else :
            raise ValueError("Unknown solver")

        # ============================
        # Step 5: Solve and analyse
        # ============================
        for i, X_sub in enumerate(X_sub_list):
            transform = None
            if args.transform == 'FullAffine':
                transform = FullAffine(d)
            elif args.transform == 'PSDAffine':
                transform = PSDAffine(d)
            elif args.transform == 'DiagonalAffine':
                transform = DiagonalAffine(d)
            elif args.transform == 'LowRankAffine':
                transform = LowRankAffine(d, r=max(int(d/4),1))
            elif args.transform == 'SmallMLP':
                transform = SmallMLP(d, hidden=16)
            elif args.transform == 'DirectOptimization':
                transform = DirectOptimization(X_sub, xl, xu, box_clip=True)
            elif args.transform == 'GaussianCommutativeTransform':
                transform = GaussianCommutativeTransform(d)
            elif args.transform == 'GaussianTransform':
                transform = GaussianTransform(d)
            elif args.transform == 'GaussianScaleTransform':
                transform = GaussianScaleTransform(d)
            elif args.transform == 'GMMForwardTransform':
                transform = GMMForwardTransform(d, n_components=3)
            elif args.transform == 'TStudentTransform':
                transform = TStudentTransform(d)
            else:
                raise ValueError("Unknown transform")

            if isinstance(transform, ProbabilisticTransform):
                transform.fit_prior(X_sub)
            transform.to(device)

            if exact :
                K = 50
                t0 = time.time()
                if args.transform == "DirectOptimization" or args.transform == "FullAffine" or args.transform == "LowRankAffine":
                    pv = transform.pyomo_solving(X_sub, f, y_prime=y_prime, y_prime_confidence=0.9, K = K, solver = 'ipopt')
                else :
                    pv = transform.cvxpy_solving(X_sub, f, y_prime=y_prime, y_prime_confidence=0.9, K = K, solver = cv.SCS)
                tn = time.time()
                print("QP solving time:", tn - t0)


                fig = plt.figure()
                ax = fig.gca()
                # Plot original points
                ax.scatter(X_sub[:, fet1], X_sub[:, fet2], color='blue', label='Original', alpha=0.5)
                # Plot transformed points
                with torch.no_grad():
                    X_transformed = transform(X_sub.to(device)).cpu().numpy()
                # Print Wasserstein distance of transformed points
                wass = np.mean(np.linalg.norm(X_transformed - X_sub.numpy(), axis=-1, ord=2))
                print("Wasserstein distance of QP solution:", wass)
                # print empirical Lipschitz constant
                lip = 1-bi_lipschitz_metric(X_sub, torch.tensor(X_transformed, dtype=torch.float32))
                print("Empirical Lipschitz constant of QP solution:", lip)
                # Real Bilipschitz (bigger and smaller singular value of A)
                if args.transform == "FullAffine":
                    U, S, Vt = np.linalg.svd(transform.A.detach().cpu().numpy())
                    print("Real Bilipschitz constant of QP solution:", S[0], S[-1])

                ax.scatter(X_transformed[:, fet1], X_transformed[:, fet2], color='orange', label='Transformed', alpha=0.5)
                # Set lims to 0,1
                if x_lims[0] is not None and x_lims[1] is not None:
                    ax.set_xlim(x_lims[0], x_lims[1])
                if y_lims[0] is not None and y_lims[1] is not None:
                    ax.set_ylim(y_lims[0], y_lims[1])

                # Plot arrows
                for j in range(X_sub.shape[0]):
                    ax.arrow(X_sub[j, fet1], X_sub[j, fet2], X_transformed[j, fet1] - X_sub[j, fet1],
                             X_transformed[j, fet2] - X_sub[j, fet2],
                             head_width=0.01, head_length=0.01, fc='gray', ec='gray', alpha=0.3)
                ax.set_title(f"QP solution for cluster {i} (transform {args.transform})")
                ax.legend()
                plt.show()

                raise Exception("Stop")


            t0 = time.time()
            X_prime, res_f, res_x = solver.solve(transform, f, X_sub, y_prime=y_prime, y_prime_confidence = 0.9,
                                                 seed=args.random_seed)
            tn = time.time()
            print("Solver time:", tn - t0)

            if res_f is not None:
                # Order solutions by f2 values
                order = np.argsort([i[1] for i in res_f])
                res_f = [res_f[i] for i in order]
                res_x = [res_x[i] for i in order]

                # Compute also the Lipschitz values
                lip_real = np.zeros(len(res_x))
                for j in range(len(res_x)):
                    transform.load_parameters(res_x[j])
                    with torch.no_grad():
                        lip = bi_lipschitz_metric(X_sub, transform(X_sub))
                    # Convert to 1-Lipschitz
                    lip_real[j] = lip
                res_f = np.hstack([res_f, lip_real.reshape(-1,1)])

                X_sub_test = X_sub_list_test[i]
                # To tensor
                X_sub_test = torch.tensor(X_sub_test, dtype=torch.float32)
                # If the transform is NOT DirectOptimization, forward the test sample and obtain Wasserstein and Lipschitz
                if not args.transform == 'DirectOptimization' and len(X_sub_list_test) > i:
                    res_f_test = np.zeros((len(res_x), 2))
                    for j in range(len(res_x)):
                        transform.load_parameters(res_x[j])
                        with torch.no_grad():
                            X_sub_test_prime = transform(X_sub_test.to(device))
                        # Wasserstein
                        diff = (X_sub_test_prime - X_sub_test).cpu().numpy()
                        wasserstein = np.mean(np.linalg.norm(diff, axis=-1, ord=2))
                        res_f_test[j, 0] = wasserstein
                        # Empirical test Lipschitz
                        lip = bi_lipschitz_metric(X_sub_test, X_sub_test_prime)
                        res_f_test[j, 1] = lip
                    print("Test set results (Wasserstein, Lipschitz proxy, Lipschitz):")
                    print(res_f_test)
                else:
                    # Fill with nans
                    res_f_test = np.full((len(res_x), 2), np.nan)


                # Store results in a dataframe
                df_results = pd.DataFrame(np.hstack([res_f, res_f_test]), columns=['Wasserstein', 'Lipschitz proxy', 'Lipschitz', 'Wasserstein test', 'Lipschitz test'])

                if args.verbose:
                    if args.transform == 'DirectOptimization':
                        X_sub_test = None

                    print_plot_solutions(res_f, res_x, transform, X_sub, X_sub_test, n_pics=4, x_lims=x_lims,
                                         y_lims=y_lims,
                                         fets=(fet1, fet2), exec_time=tn - t0)

            # No covergence, empty dataset
            else :
                df_results = pd.DataFrame(columns=['Wasserstein', 'Lipschitz proxy', 'Lipschitz', 'Wasserstein test', 'Lipschitz test'])
            # Save to csv
            file_name = os.path.join(transform_path, f'label_{y_orig}_cluster_{i}.csv')
            #df_results.to_csv(file_name, index=False)

            print("Time", tn-t0)

            # Store exec time
            df_time.loc[(y_orig, i), 'exec_time'] = tn - t0

            raise Exception("Stop after 1 cluster")


    # Save exec time dataframe
    df_time.to_csv(os.path.join(transform_path, "exec_times.csv"), index=False)