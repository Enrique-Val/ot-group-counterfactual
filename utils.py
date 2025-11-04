import time

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from scipy.stats import uniform, norm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
import openml as oml
from sklearn.ensemble import GradientBoostingClassifier

from group_cfx.transforms.functional_transforms import DirectOptimization
from group_cfx.transforms.utils import bi_lipschitz_metric


def synthetic_2d(noise_scale=0, size = 1000, random_state=0) -> tuple[np.ndarray, np.ndarray]:
    x1 = uniform.rvs(loc=0, scale=1, size=size, random_state=random_state)
    x2 = uniform.rvs(loc=0, scale=1, size=size, random_state=random_state+1)
    noise = norm.rvs(loc=0, scale=noise_scale, size=size, random_state=random_state+2)
    y = np.zeros_like(x1)
    y[x1 + x2 + noise > 1] = 1

    X = np.column_stack((x1, x2))
    y = y.astype(np.int64)
    return X, y

def get_openml_dataset(data_id: int) -> tuple[np.ndarray, np.ndarray, dict]:
    dataset_oml = oml.datasets.get_dataset(data_id, download_data=True, download_qualities=False,
                                        download_features_meta_data=False)
    X_df, y_ser, _, _ = dataset_oml.get_data(target = dataset_oml.default_target_attribute, dataset_format="numpy")
    X = X_df.to_numpy()
    # Encode y with ordinals
    y = y_ser.to_numpy()
    labels = np.unique(y)
    label_dict = {label: i for i, label in enumerate(labels)}
    y = np.array([label_dict[label] for label in y], dtype=np.int64)
    return X, y, label_dict

class Classifier(nn.Module):
    def __init__(self, d, hidden=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 2)
        )
    def forward(self, x):
        return self.net(x)

def train_classifier(X, y, batch_size = 64, device="cpu"):
    # Torch tensors
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)
    dataset = TensorDataset(X_tensor, y_tensor)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    d = X.shape[1]
    f = Classifier(d).to(device)

    ce = nn.CrossEntropyLoss()
    # Define scheduler and optimizer
    opt_f = optim.Adam(f.parameters(), lr=1e-2)
    scheduler_f = optim.lr_scheduler.StepLR(opt_f, step_size=20, gamma=0.5)


    # Train classifier
    epochs = 100
    for epoch in range(epochs):
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = f(xb)
            loss = ce(logits, yb)
            opt_f.zero_grad()
            loss.backward()
            opt_f.step()
            scheduler_f.step()
            print(f"Epoch {epoch}, loss: {loss.item():.4f}", end='\r')
        print(f"Epoch {epoch}")
        # Compute loss
        with torch.no_grad():
            logits = f(X_tensor.to(device))
            loss = ce(logits, y_tensor.to(device))
            preds = logits.argmax(1).cpu().numpy()
            acc = (preds == y).mean()
            print(f"Epoch {epoch}, loss: {loss.item():.4f}, acc: {acc:.4f}")
    with torch.no_grad():
        preds = f(X_tensor.to(device)).argmax(1).cpu().numpy()
        acc = (preds == y).mean()
        print("Torch classifier accuracy:", acc)

    return f

def train_gbt(X, y, random_state):
    # Train using CV for validation
    from sklearn.model_selection import GridSearchCV
    param_grid = {
        #'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.2
        ],
        'max_depth': [3, 5, 7]
    }
    gbt_cv = GridSearchCV(GradientBoostingClassifier(random_state=random_state), param_grid, cv=10, n_jobs=-1)
    gbt_cv.fit(X, y)



    print("GBT classifier accuracy:", gbt_cv.best_score_)
    return gbt_cv.best_estimator_

def train_lg(X, y, scoring = "accuracy", random_state = 0, max_iter = 100):
    param_grid = {
        'C': np.logspace(-3, 3, 20),
        'penalty': ['l1', 'l2'],
        'solver': ['saga', 'liblinear'],
    }

    grid = GridSearchCV(LogisticRegression(random_state = random_state, max_iter = max_iter), param_grid, cv=10,
                        scoring= scoring, n_jobs=10)
    grid.fit(X, y)

    # Train another grid with elasticnet penalty
    param_grid = {
        'C': np.logspace(-3, 3, 20),
        'penalty': ['elasticnet'],
        'solver': ['saga'],
        'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9]
    }

    grid_en = GridSearchCV(LogisticRegression(random_state = random_state, max_iter = max_iter), param_grid, cv=10,
                           scoring= scoring, n_jobs=10)
    grid_en.fit(X, y)

    if grid_en.best_score_ > grid.best_score_:
        grid = grid_en

    print("LG classifier " + scoring+":" , grid.best_score_)

    return grid.best_estimator_, grid.best_params_, grid.best_score_

def print_plot_solutions(res_f, res_x, transform, X_sub, X_sub_test = None, n_pics = 4, x_lims = (None,None), y_lims = (None,None),
                         fets=(0,1), exec_time=None) :
    # Infer some params
    n = X_sub.shape[0]
    d = X_sub.shape[1]
    fet1 = fets[0]
    fet2 = fets[1]
    mod_number = max(1, (len(res_f)) // (n_pics-1) +1 )
    device = transform.parameters().__next__().device

    # Print each solution with its value
    # Optimal subplot grid based on n_pics
    n_rows = int(np.ceil(np.sqrt(n_pics)))
    n_cols = int(np.ceil(n_pics / n_rows))
    fig,axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 4*n_rows))
    for i in range(1,len(res_f)):
        # Plot it. Load params into transform and predict
        if i % mod_number > 0 and not i == len(res_f) - 1 and not i == 1: continue  # plot only some solutions
        #res_x[i][:d*d] = 0.0  # zero out first d^2 params
        transform.load_parameters(res_x[i])
        for p in transform.parameters():
            print(p)
        with torch.no_grad():
            X_sub_prime = transform(X_sub.to(device)).cpu().numpy()
        ax_i = n_pics-1 if i == len(res_f) - 1 else i // mod_number
        ax = axes.flatten()[ax_i]
        sc1 = ax.scatter(X_sub[:, fet1], X_sub[:, fet2], c='blue', label='Original', alpha=0.5)
        sc2 = ax.scatter(X_sub_prime[:, fet1], X_sub_prime[:, fet2], c='red', label='Transformed', alpha=0.5)
        ax.legend()
        # Arrows
        for j in range(X_sub.shape[0]):
            ax.arrow(X_sub[j, fet1], X_sub[j, fet2], X_sub_prime[j, fet1] - X_sub[j, fet1], X_sub_prime[j, fet2] - X_sub[j, fet2],
                     head_width=0.01, head_length=0.01, fc='gray', ec='gray', alpha=0.3)
        # Set lims to 0,1
        ax.set_xlim(x_lims[0], x_lims[1])
        ax.set_ylim(y_lims[0], y_lims[1])
        ax.set_title(f"Wass.={np.round(res_f[i][0],2)}, LP.={np.round(1-res_f[i][1],2)}, LR {np.round(1-res_f[i][2],2)} (Sol. {i})")
    #Title the fig
    transform_type = type(transform).__name__
    if exec_time is not None:
        fig.suptitle(f"Transformations found ({transform_type}, time: {exec_time:.1f}s)", fontsize=16)
    else:
        fig.suptitle(f"Transformations found", fontsize=16)
    fig.show()

    if X_sub_test is not None:
        # Print also test solutions
        # Optimal subplot grid based on n_pics
        n_rows = int(np.ceil(np.sqrt(n_pics)))
        n_cols = int(np.ceil(n_pics / n_rows))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
        for i in range(1, len(res_f)):
            # Plot it. Load params into transform and predict
            if i % mod_number > 0 and not i == len(res_f) - 1 and not i == 1: continue  # plot only some solutions
            # res_x[i][:d*d] = 0.0  # zero out first d^2 params
            transform.load_parameters(res_x[i])
            for p in transform.parameters():
                print(p)
            with torch.no_grad():
                X_sub_prime = transform(X_sub_test.to(device)).cpu().numpy()
            ax_i = n_pics - 1 if i == len(res_f) - 1 else i // mod_number
            ax = axes.flatten()[ax_i]
            sc1 = ax.scatter(X_sub_test[:, fet1], X_sub_test[:, fet2], c='blue', label='Original', alpha=0.5)
            sc2 = ax.scatter(X_sub_prime[:, fet1], X_sub_prime[:, fet2], c='red', label='Transformed', alpha=0.5)
            ax.legend()
            # Arrows
            for j in range(X_sub_test.shape[0]):
                ax.arrow(X_sub_test[j, fet1], X_sub_test[j, fet2], X_sub_prime[j, fet1] - X_sub_test[j, fet1],
                         X_sub_prime[j, fet2] - X_sub_test[j, fet2],
                         head_width=0.01, head_length=0.01, fc='gray', ec='gray', alpha=0.3)
            # Set lims to 0,1
            ax.set_xlim(x_lims[0], x_lims[1])
            ax.set_ylim(y_lims[0], y_lims[1])
            ax.set_title(f"Wass.={np.round(res_f[i][0], 2)} Lipsc.={np.round(1 - res_f[i][1], 2)} (Sol. {i})")
        # Title the fig
        transform_type = type(transform).__name__
        if exec_time is not None:
            fig.suptitle(f"Transformations found ({transform_type}, time: {exec_time:.1f}s)", fontsize=16)
        else:
            fig.suptitle(f"Transformations found", fontsize=16)
        fig.show()

    # Print also the Pareto front
    fig2 = plt.figure()
    ax2 = fig2.gca()
    ax2.scatter([f[0] for f in res_f],[1 - f[2] for f in res_f], c='blue')
    ax2.set_xlabel("Wasserstein distance")
    ax2.set_ylabel("1 - Bi-Lipschitz metric")
    ax2.set_title("Pareto front")
    fig2.show()

def direct_experiment(transform, X_sub_train : torch.Tensor, X_sub_test : torch.Tensor, f, y_prime, y_prime_confidence, K, solver, device = "cpu"):
    if transform.is_cvx() :
        t0 = time.time()
        pv = transform.cvxpy_solving(X_sub_train, f, y_prime=y_prime, y_prime_confidence=y_prime_confidence, K=K, solver=solver)
        tn = time.time()
    else :
        t0 = time.time()
        pv = transform.pyomo_solving(X_sub_train, f , y_prime=y_prime, y_prime_confidence=y_prime_confidence, K=K, solver=solver)
        tn = time.time()
        time.sleep(10)

    wass_test = None
    if X_sub_test is not None:
        with torch.no_grad():
            X_transformed = transform(X_sub_test.to(device)).cpu().numpy()
        wass_test = np.mean(np.linalg.norm(X_transformed - X_sub_test.numpy(), axis=-1, ord=2))
    with torch.no_grad():
        X_transformed = transform(X_sub_train.to(device)).cpu().numpy()
    wass = np.mean(np.linalg.norm(X_transformed - X_sub_train.numpy(), axis=-1, ord=2))
    return wass, wass_test, tn - t0

def cross_experiment(transform, X_sub : torch.Tensor, f, y_prime, y_prime_confidence, K, solver, device = "cpu"):
    if isinstance(transform, DirectOptimization):
        return direct_experiment(transform, X_sub, None, f, y_prime, y_prime_confidence, K, solver, device)
    # else :
    # Divide X_sub in 10, use 9 for fitting the transform, 1 for testing iteratively
    n = X_sub.shape[0]
    fold_size = n // 10
    wass_list = []
    wass_test_list = []
    time_list = []
    for i in range(10):
        start = i * fold_size
        end = (i + 1) * fold_size if i < 9 else n
        X_sub_train = torch.cat([X_sub[:start], X_sub[end:]], dim=0)
        X_sub_test = X_sub[start:end]

        wass, wass_test, time = direct_experiment(transform, X_sub_train, X_sub_test, f, y_prime, y_prime_confidence, K, solver,
                                       device)

        wass_list.append(wass)
        wass_test_list.append(wass_test)
        time_list.append(time)
    return np.mean(wass_list), np.mean(wass_test_list), np.mean(time_list)

def direct_experiment_pymoo(transform, X_sub_train : torch.Tensor, X_sub_test : torch.Tensor, f, y_prime,
                            y_prime_confidence, solver, device = "cpu", random_seed = 0):
    t0 = time.time()
    X_prime, res_f, res_x = solver.solve(transform, f, X_sub_train, y_prime=y_prime,
                                         y_prime_confidence = y_prime_confidence, seed=random_seed)
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
                lip = bi_lipschitz_metric(X_sub_train, transform(X_sub_train))
            # Convert to 1-Lipschitz
            lip_real[j] = lip
        res_f = np.hstack([res_f, lip_real.reshape(-1, 1)])

        # If the transform is NOT DirectOptimization, forward the test sample and obtain Wasserstein and Lipschitz
        if isinstance(transform, DirectOptimization):
            # Then just append NAs for test set results
            res_f_test = np.full((len(res_x), 2), np.nan)
            df_results = pd.DataFrame(np.hstack([res_f, res_f_test]),
                                      columns=['Wasserstein', 'Lipschitz proxy', 'Lipschitz', 'Wasserstein test',
                                               'Lipschitz test'])
        else:
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

            # Store results in a dataframe
            df_results = pd.DataFrame(np.hstack([res_f, res_f_test]),
                                      columns=['Wasserstein', 'Lipschitz proxy', 'Lipschitz', 'Wasserstein test',
                                               'Lipschitz test'])

    # No covergence, empty dataset
    else:
        df_results = pd.DataFrame(
            columns=['Wasserstein', 'Lipschitz proxy', 'Lipschitz', 'Wasserstein test', 'Lipschitz test'])
    # Save to csv
    return df_results, tn - t0

def cross_experiment_pymoo(transform, X_sub : torch.Tensor, f, y_prime, y_prime_confidence, solver, device = "cpu",
                           random_seed = 0, k_folds = 10):
    if isinstance(transform, DirectOptimization):
        return direct_experiment_pymoo(transform, X_sub, None, f, y_prime, y_prime_confidence, solver, device, random_seed)
    # Divide X_sub in 10, use 9 for fitting the transform, 1 for testing iteratively
    n = X_sub.shape[0]
    fold_size = n // k_folds
    df_list = []
    time_list = []
    for i in range(k_folds):
        start = i * fold_size
        end = (i + 1) * fold_size if i < (k_folds-1) else n
        X_sub_train = torch.cat([X_sub[:start], X_sub[end:]], dim=0)
        X_sub_test = X_sub[start:end]

        df, time = direct_experiment_pymoo(transform, X_sub_train, X_sub_test, f, y_prime, y_prime_confidence, solver,
                                           device, random_seed)

        # Add to df a first column with the fold number
        df.insert(0, 'CVF', i+1)

        df_list.append(df)
        time_list.append(time)
    return pd.concat(df_list), np.mean(time_list)

