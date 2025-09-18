import time

import numpy as np
import torch
from matplotlib import pyplot as plt
from scipy.stats import uniform, norm
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
import openml as oml
from sklearn.ensemble import GradientBoostingClassifier


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
    gbt = GradientBoostingClassifier(random_state=random_state)
    gbt.fit(X, y)
    print("GBT classifier accuracy:", gbt.score(X, y))
    return gbt

def print_plot_solutions(res_f, res_x, transform, X_sub, n_pics = 4, x_lims = (None,None), y_lims = (None,None),
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
        ax.set_title(f"Wass.={np.round(res_f[i][0],2)} Lipsc.={np.round(1-res_f[i][1],2)} (Sol. {i})")
    #Title the fig
    transform_type = type(transform).__name__
    if exec_time is not None:
        fig.suptitle(f"Transformations found ({transform_type}, time: {exec_time:.1f}s)", fontsize=16)
    else:
        fig.suptitle(f"Transformations found", fontsize=16)
    fig.show()

    # Print also the Pareto front
    fig2 = plt.figure()
    ax2 = fig2.gca()
    ax2.scatter([f[0] for f in res_f],[1 - f[1] for f in res_f], c='blue')
    ax2.set_xlabel("Wasserstein distance")
    ax2.set_ylabel("1 - Bi-Lipschitz metric")
    ax2.set_title("Pareto front")
    fig2.show()



