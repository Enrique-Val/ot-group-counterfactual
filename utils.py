import time

import numpy as np
import torch
from matplotlib import pyplot as plt
from scipy.stats import uniform, norm
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
import openml as oml
from sklearn.ensemble import GradientBoostingClassifier


def synthetic_2d(noise_scale=0, size = 1000) :
    x1 = uniform.rvs(loc=0, scale=1, size=size)
    x2 = uniform.rvs(loc=0, scale=1, size=size)
    noise = norm.rvs(loc=0, scale=noise_scale, size=size)
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


def bi_lipschitz_metric(X, Y, eps=1e-12):
    """
    Empirical bi-Lipschitz constant between X and Y.

    X, Y: tensors of shape (n, d) or (d,)  (before/after transform)
    eps: numerical stability

    Returns:
        M: estimated bi-Lipschitz constant
    """

    # Compute distance between every pair of X
    n = X.shape[0]
    sq_norms = (X ** 2).sum(dim=1, keepdim=True)
    dists_sq = sq_norms + sq_norms.T - 2 * X @ X.T
    dists_sq = torch.clamp(dists_sq, min=0.0)
    dists_X = torch.sqrt(dists_sq)

    sq_norms = (Y ** 2).sum(dim=1, keepdim=True)
    dists_sq = sq_norms + sq_norms.T - 2 * Y @ Y.T
    dists_sq = torch.clamp(dists_sq, min=0.0)
    dists_Y = torch.sqrt(dists_sq)

    # Avoid division by zero by adding eps to distances
    ratio1 = dists_Y / (dists_X + eps)
    ratio2 = dists_X / (dists_Y + eps)

    # Elementwise min of the two ratios
    ratio = torch.min(ratio1, ratio2)
    M = ratio.mean().item()
    return 1-M

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

def train_gbt(X, y):
    gbt = GradientBoostingClassifier(random_state=0)
    gbt.fit(X, y)
    print("GBT classifier accuracy:", gbt.score(X, y))
    return gbt

def print_plot_solutions(res_f, res_x, transform, X_sub, n_pics = 4, x_lims = (None,None), y_lims = (None,None),
                         fets=(0,1)):
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
    for i in range(len(res_f)):
        # Plot it. Load params into transform and predict
        if i % mod_number > 0 and not i == len(res_f) - 1: continue  # plot only some solutions
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
        ax.set_title(f"Solution {i}: L. proxy={np.round(res_f[i][1],2)}")
    fig.show()


def build_covariance_matrix(marginal_stds, correlation_triangle) :
    """
    Build the covariance matrix from diagonal and lower-triangular correlation parameters.

    marginal_variances: tensor of shape (d,)  (stddev)
    correlation_triangle: tensor of shape (d*(d-1)/2,)  (lower-triangular correlation entries)

    Returns:
        Sigma: tensor of shape (d, d)  (covariance matrix)
    """
    d = marginal_stds.shape[0]
    C = np.eye(d)
    tril_idx = np.tril_indices(d, -1)
    C[tril_idx] = correlation_triangle
    C[(tril_idx[1], tril_idx[0])] = correlation_triangle  # symmetry

    # scale by std
    D = np.diag(marginal_stds)
    Sigma = D @ C @ D

    # For stability, symmetrize
    Sigma = (Sigma + Sigma.T) / 2
    return Sigma

def compute_A(sigma1: np.ndarray, sigma2: np.ndarray, tol: float = 1e-12) -> np.ndarray:
    """
    Compute A = Sigma1^{-1/2} ( Sigma1^{1/2} Sigma2 Sigma1^{1/2} )^{1/2} Sigma1^{-1/2}.
    Works for PSD (possibly singular) Sigma1 and PSD Sigma2.
    Returns a symmetric matrix (numerical symmetrization applied).
    """
    # symmetrize inputs
    sigma1 = (sigma1 + sigma1.T) / 2
    sigma2 = (sigma2 + sigma2.T) / 2

    # eigendecomposition of sigma1
    w1, V1 = np.linalg.eigh(sigma1)
    # threshold small/negative eigenvalues
    w1_clipped = np.clip(w1, a_min=0.0, a_max=None)
    # sqrt and inv-sqrt (use 0 where eigenvalue ~ 0)
    sqrt_w1 = np.sqrt(w1_clipped)
    inv_sqrt_w1 = np.zeros_like(sqrt_w1)
    mask = sqrt_w1 > tol
    inv_sqrt_w1[mask] = 1.0 / sqrt_w1[mask]

    Sigma1_sqrt = (V1 * sqrt_w1) @ V1.T
    Sigma1_inv_sqrt = (V1 * inv_sqrt_w1) @ V1.T

    # middle = Sigma1^{1/2} Sigma2 Sigma1^{1/2}
    middle = Sigma1_sqrt @ sigma2 @ Sigma1_sqrt
    middle = (middle + middle.T) / 2

    # eigen-decomposition of middle and its sqrt (clip small negative evs due to num error)
    w2, V2 = np.linalg.eigh(middle)
    w2_clipped = np.clip(w2, a_min=0.0, a_max=None)
    sqrt_w2 = np.sqrt(w2_clipped)
    middle_sqrt = (V2 * sqrt_w2) @ V2.T

    # assemble A
    A = Sigma1_inv_sqrt @ middle_sqrt @ Sigma1_inv_sqrt
    A = (A + A.T) / 2  # ensure symmetry
    # Convert it to np.float32
    A = A.astype(np.float32)
    return A

def compute_A_commuting(sigma1: np.ndarray, sigma2: np.ndarray, tol: float = 1e-12) -> np.ndarray:
    """
    Fast computation of A when sigma1 and sigma2 commute.
    sigma1, sigma2 assumed symmetric PSD.
    """
    # symmetrize
    sigma1 = (sigma1 + sigma1.T) / 2
    sigma2 = (sigma2 + sigma2.T) / 2

    # diagonalize sigma1
    w1, Q = np.linalg.eigh(sigma1)
    print("Eigenvalues of sigma1:", w1)
    # sigma2 in same basis
    sigma2_diag = np.diag(Q.T @ sigma2 @ Q)
    w2 = sigma2_diag
    print("Eigenvalues of sigma2:", w2)
    print("Alternative computation of sigma2 diag:", np.linalg.eigh(sigma2))
    raise NotImplementedError("This function is not yet implemented correctly.")

    # mask nonzero directions
    nonzero = w1 > tol
    lamA = np.zeros_like(w1)

    # check consistency: if w1≈0 then w2 must also≈0
    if np.any((~nonzero) & (np.abs(w2) > tol)):
        raise ValueError("Incompatible: sigma1 has zero eigenvalue where sigma2 is nonzero.")

    #lamA[nonzero] = np.sqrt(np.clip(w2[nonzero], 0, None) / w1[nonzero])

    # reconstruct A
    A = (Q * lamA) @ Q.T
    A = A.astype(np.float32)
    return A  # symmetrize for safety

# Example
if __name__ == "__main__":
    #np.random.seed(0)
    n = 500
    G = np.random.randn(n, n)
    Sigma1 = G @ G.T + 1e-6 * np.eye(n)  # PD
    H = np.random.randn(n, n)
    Sigma2 = Sigma1*2#H @ H.T                     # PSD
    t0 = time.time()
    A = compute_A(Sigma1, Sigma2)
    tn = time.time()
    # verify A Sigma1 A ≈ Sigma2
    print("norm(A Sigma1 A - Sigma2) =", np.linalg.norm(A @ Sigma1 @ A - Sigma2))
    print("A symmetric residual =", np.linalg.norm(A - A.T))
    print("A =", A)
    print("Sigma1 =", Sigma1)
    print("Sigma2 =", Sigma2)

    # Sample from sigma1 (assume zero mean)
    m = 10000
    X1 = np.random.multivariate_normal(mean=np.zeros(n), cov=Sigma1, size=m)
    # Transform samples
    X2_trans = A @ X1.T
    # Sample from sigma2
    X2 = np.random.multivariate_normal(mean=np.zeros(n), cov=Sigma2, size=m).T

    # Compare empirical covariances
    emp_cov_2_trans = np.cov(X2_trans)
    emp_cov_2 = np.cov(X2)

    print()
    print(emp_cov_2)
    print(emp_cov_2_trans)
    print("Diff =", emp_cov_2 - emp_cov_2_trans)

    '''#Plot both of them
    import matplotlib.pyplot as plt
    plt.scatter(X2[0, :], X2[1, :], alpha=0.5, label='Sigma2 samples')
    plt.scatter(X2_trans[0, :], X2_trans[1, :], alpha=0.5, label='Sigma2 samples trans', color="red")
    plt.axis('equal')
    plt.legend()
    plt.show()'''

    print()
    print("Commuting case:")
    t0_com = time.time()
    A = compute_A_commuting(Sigma1, Sigma2)
    tn_com = time.time()
    # verify A Sigma1 A ≈ Sigma2
    print("norm(A Sigma1 A - Sigma2) =", np.linalg.norm(A @ Sigma1 @ A - Sigma2))
    print("A symmetric residual =", np.linalg.norm(A - A.T))
    print("A =", A)
    print("Sigma1 =", Sigma1)
    print("Sigma2 =", Sigma2)

    # Sample from sigma1 (assume zero mean)
    m = 10000
    X1 = np.random.multivariate_normal(mean=np.zeros(n), cov=Sigma1, size=m)
    # Transform samples
    X2_trans = A @ X1.T
    # Sample from sigma2
    X2 = np.random.multivariate_normal(mean=np.zeros(n), cov=Sigma2, size=m).T

    # Compare empirical covariances
    emp_cov_2_trans = np.cov(X2_trans)
    emp_cov_2 = np.cov(X2)

    print()
    print(emp_cov_2)
    print(emp_cov_2_trans)
    print("Diff =", emp_cov_2 - emp_cov_2_trans)


    '''plt.scatter(X2[0, :], X2[1, :], alpha=0.5, label='Sigma2 samples')
    plt.scatter(X2_trans[0, :], X2_trans[1, :], alpha=0.5, label='Sigma2 samples trans', color="red")
    plt.axis('equal')
    plt.legend()
    plt.show()'''

    print("Time:", tn - t0)
    print("Time commuting:", tn_com - t0_com)






