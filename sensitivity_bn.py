import argparse
import time

import networkx as nx
import numpy as np
import pandas as pd
import sklearn
from pymoo.algorithms.moo.nsga2 import NSGA2
import torch
import matplotlib.pyplot as plt
import os

from pymoo.termination.default import DefaultMultiObjectiveTermination

from group_cfx.solver.pymoo_solver import PyMooSolver
from group_cfx.transforms.probabilistic_transforms import ProbabilisticTransform
from utils import synthetic_2d, get_openml_dataset, get_transform

import psutil

import joblib

import pybnesian as pb

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

def train_bn(X, blacklist=[]):
    test = pb.MutualInformation(df=X)
    est = pb.PC()
    cpdag = est.estimate(hypot_test=test, arc_blacklist=blacklist)
    dag = cpdag.to_dag()
    bn = pb.BayesianNetwork(pb.GaussianNetworkType(), dag)
    bn.fit(X)
    return bn

def print_mem(prefix=""):
    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss / 1e6  # MB
    print(f"{prefix} memory: {mem:.2f} MB")

def get_structure(bn, nodes, arcs):
    # Extract graph and plot it
    nodes = bn.nodes()
    arcs = bn.arcs()
    G = nx.DiGraph()
    G.add_nodes_from(nodes)
    G.add_edges_from(arcs)
    pos = nx.spring_layout(G)
    # Dot layout
    pos = nx.nx_pydot.pydot_layout(G)
    return G, pos

def plot_bn_structure(bn, title="", filename=None):
    G, pos = get_structure(bn, bn.nodes(), bn.arcs())
    fig = plt.figure(figsize=(6, 4))
    ax = fig.gca()
    # Same color (skyblue), except for K, which is lightcoral
    colors = []
    for node in bn.nodes():
        if node == "K":
            colors.append("lightcoral")
        elif node.endswith("\'"):
            colors.append("lightgreen")
        else:
            colors.append("skyblue")
    nx.draw(G, pos, with_labels=True, node_size=2000, node_color=colors, font_size=16, font_weight='bold',
            arrowsize=20, ax=ax)
    ax.margins(0.1)
    # ax.set_title(f"Learned Bayesian Network Structure (only primes and bilip) for label {y_orig}, cluster {cluster_id}")
    fig.tight_layout()
    if filename is not None:
        fig.savefig(filename, bbox_inches='tight')
    fig.show()

if __name__ == "__main__":
    # Argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_id', type=int, default=-1, help='OpenML dataset ID')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    parser.add_argument('--output_dir', type=str, default='./results', help='Output directory')
    parser.add_argument('--device', type=str, default='cpu', help='Device to use (cpu or cuda)')
    parser.add_argument('--random_seed', type=int, default=0, help='Random seed for reproducibility')
    parser.add_argument('--n_clusters', type=int, default=10, help='Number of clusters for subgrouping')
    parser.add_argument('--math_opt', action='store_true', help='Use mathematical optimization')
    parser.add_argument('--transform', type=str, default='FullAffine', help='Type of transform to use',
                        choices=['FullAffine', 'FullAffine_proxy', 'PSDAffine', 'PSDAffine_proxy', 'DiagonalAffine',
                                 'DirectOptimization',
                                 'GaussianCommutativeTransform', 'GaussianTransform', 'GaussianTransform_proxy', 'GaussianScaleTransform',
                                 'GMMForwardTransform'])
    args = parser.parse_args()

    np.random.seed(args.random_seed)

    args.data_id = -1
    args.transform = 'PSDAffine'
    args.verbose = True
    args.math_opt = True

    t0 = time.time()

    # Create output directory if it does not exist
    data_path = os.path.join(args.output_dir, f"data_{args.data_id}")
    models_path = os.path.join(data_path, "models")
    cluster_path = os.path.join(data_path, str(args.n_clusters))


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
    cluster_id = 0
    cluster_alg_dir = os.path.join(cluster_path, "label_"+str(y_orig)+".pkl")



    if not os.path.exists(cluster_alg_dir):
        raise ValueError("No clustering algorithm found")
    cluster_alg = joblib.load(cluster_alg_dir)

    y_prime = unique_labels[unique_labels != y_orig][0]
    print("Y_orig =", y_orig, "y_prime =", y_prime)

    sub_data = X_test[y_test == y_orig]
    cluster_labels = cluster_alg.predict(sub_data)

    X_sub = sub_data[cluster_labels == cluster_id]
    # Limit to 200 instances for testing
    X_sub = X_sub[:200]
    min_samples = 20
    if X_sub.shape[0] < min_samples:
        # Warn the user and introduce synthetic samples by jittering up to min_samples
        print(f"Warning: Cluster {cluster_id} has less than 20 samples ({X_sub.shape[0]} samples). Augmenting data by jittering.")
        n_needed = min_samples - X_sub.shape[0]
        noise = np.random.normal(0, 0.01, size=(n_needed, X_sub.shape[1]))
        jittered_samples = X_sub[np.random.choice(X_sub.shape[0], n_needed, replace=True)] + noise
        X_sub = np.vstack([X_sub, jittered_samples])

    # Confidence for y_prime
    y_prime_conf = 0.8

    # ============================
    # Step 5: Solve and analyse
    # ============================
    transform = get_transform(args.transform, X_sub, device="cpu")

    if isinstance(transform, ProbabilisticTransform):
        transform.fit_prior(X_sub)

    with torch.no_grad():
        X_sub_tensor = torch.tensor(X_sub, dtype=torch.float32)

    if args.math_opt:
        X_global = []
        for K in [1.01,1.5,2.0, 3.5,5.0]:
            transform.cvxpy_solving(X_sub, f, y_prime, y_prime_confidence=y_prime_conf, K=K)
            X_prime_t = transform(X_sub_tensor)
            # Stach horizontall X_sub and X_prime_t
            stacked = np.hstack([X_sub, X_prime_t.detach().numpy()])
            # Add an additional column with the value of bilipschitz proxy
            bilip_column = np.full((stacked.shape[0], 1), K)
            stacked = np.hstack([stacked, bilip_column])
            X_global.append(stacked)

            # Also, plot the transformed samples
            plt.figure(figsize=(8, 6))
            plt.scatter(X_sub[:, 0], X_sub[:, 1], color = 'blue', label='Original Samples', alpha=0.5)
            plt.scatter(X_prime_t[:, 0].detach().numpy(), X_prime_t[:, 1].detach().numpy(), color='red',
                        label='Transformed Samples', alpha=0.5)
            plt.title(f"Transformed Samples for label {y_orig}, cluster {cluster_id}, K={K}")
            plt.xlabel('Feature 1')
            plt.ylabel('Feature 2')
            plt.legend()
            plt.show()
        X_global = np.vstack(X_global)

    else :
        solver = PyMooSolver(
            algorithm=NSGA2(pop_size=200, eliminate_duplicates=True, verbose =args.verbose, seed=args.random_seed),
            termination=DefaultMultiObjectiveTermination(n_max_gen=1000),
            verbose=args.verbose,
            min_acc=y_prime_conf
        )

        X_prime, res_f, res_x = solver.solve(transform, f, X_sub_tensor, y_prime=y_prime,
                                             y_prime_confidence=y_prime_conf, seed=args.random_seed)

        # First, reconstruct each transform from res_x. We want to create a dataset of transformed samples
        X_global = []
        for i in range(res_x.shape[0]):
            transform.load_parameters(res_x[i])
            X_prime_t = transform(X_sub_tensor)
            # Stach horizontall X_sub and X_prime_t
            stacked = np.hstack([X_prime, X_prime_t.detach().numpy()])
            # Add an additional column with the value of bilipschitz proxy
            bilip_proxy = res_f[i,1]
            bilip_column = np.full((stacked.shape[0], 1), 1/(1-bilip_proxy))
            stacked = np.hstack([stacked, bilip_column])
            X_global.append(stacked)

            '''# Also, plot the transformed samples
            plt.figure(figsize=(8, 6))
            plt.scatter(X_sub[:, 0], X_sub[:, 1], color='blue', label='Original Samples', alpha=0.5)
            plt.scatter(X_prime_t[:, 0].detach().numpy(), X_prime_t[:, 1].detach().numpy(), color='red',
                        label='Transformed Samples', alpha=0.5)
            plt.title(f"Transformed Samples for label {y_orig}, cluster {cluster_id}, solution {i}")
            plt.xlabel('Feature 1')
            plt.ylabel('Feature 2')
            plt.legend()
            plt.show()'''
        X_global = np.vstack(X_global)

    # Create a Dataframe with columns named x1, x2, ..., xn, x1_prime, x2_prime, ..., xn_prime, bilip_proxy
    columns = []
    for i in range(d):
        columns.append(f"x{i+1}")
    for i in range(d):
        columns.append(f"x{i+1}\'")
    columns.append("K")

    X_global = pd.DataFrame(X_global, columns=columns)
    # Log transform K to log scale
    #X_global["K"] = np.log(X_global["K"]-1)
    # Min max scale each column to [0,1]
    for col in X_global.columns:
        continue
        X_global[col] = (X_global[col] - X_global[col].min()) / (X_global[col].max() - X_global[col].min())

    # Make it differential. The values of X_prime should be X_prime - X
    for i in range(d):
        X_global[f"x{i+1}\'"] = X_global[f"x{i+1}\'"]# - X_global[f"x{i+1}"]

    # First, define blacklist of edges to avoid
    #Bilipschitz cannot go into original features
    blacklist = []
    for i in range(d):
        blacklist.append(( "K", f"x{i+1}"))

    # Transformed features cannot go into original features
    for i in range(d):
        for j in range(d):
            blacklist.append((f"x{i+1}\'", f"x{j+1}"))

    # No nodes can go into bilipschitz
    for i in range(d):
        blacklist.append((f"x{i+1}", "K"))
        blacklist.append((f"x{i+1}\'", "K"))

    bn = train_bn(X_global, blacklist=blacklist)

    plot_bn_structure(bn)

    # Print cpts of prime nodes:
    print("--------------------------------")
    print("FOR BN WITH PRIME-PRIME EDGES:")
    for node in bn.nodes():
        if node.endswith("\'"):
            cpt = bn.cpd(node)
            print(f"CPT for node {node}:")
            print(cpt)
    print("--------------------------------")


    # Repeat procedure, but add to blacklist connection between primes
    for i in range(d):
        for j in range(d):
            if i != j:
                blacklist.append((f"x{i+1}\'", f"x{j+1}\'"))

    bn2 = train_bn(X_global, blacklist=blacklist)

    # Extract graph and plot it
    plot_bn_structure(bn2, filename=os.path.join(data_path, f"bn_structure_label_{y_orig}_cluster_{cluster_id}.pdf"))

    # Print cpts of prime nodes:
    print("--------------------------------")
    print("FOR BN WITH NO PRIME-PRIME EDGES:")
    for node in bn2.nodes():
        if node.endswith("\'"):
            cpt = bn2.cpd(node)
            print(f"CPT for node {node}:")
            print(cpt)
    print("--------------------------------")

    print("Total exec time:", time.time() - t0)