import numpy as np
import sklearn
import torch
from scipy.linalg import sqrtm, inv


def wasserstein_distance_normals(m0, C0, m1, C1, sqrt_C0=None) :
    # Mean term
    mean_diff = np.linalg.norm(m0 - m1) ** 2

    # Covariance term
    if sqrt_C0 is None:
        sqrt_C0 = sqrtm(C0)
    middle = sqrtm(sqrt_C0 @ C1 @ sqrt_C0)
    cov_term = np.trace(C0 + C1 - 2 * middle)

    return mean_diff + cov_term


def get_lipschitz_bounds(X, Y, eps=1e-8):
    """
    Verifies if the Lipschitz ratio ||Ay - Ay'|| / ||x - x'||
    stays within expected bounds.
    """
    # Previous step, convert tensor to float64 for better numerical stability
    X = X.double()
    Y = Y.double()

    # 1. Use torch.cdist for numerically stable Euclidean distance
    # p=2 computes the standard L2 norm
    dist_X = torch.cdist(X, X, p=2)
    dist_Y = torch.cdist(Y, Y, p=2)

    # 2. Handle the "Diagonal Problem"
    # We create a mask for non-diagonal elements
    n = X.shape[0]
    mask = ~torch.eye(n, dtype=torch.bool, device=X.device)

    # Filter out the diagonals (where distance is 0)
    valid_dist_X = dist_X[mask]
    valid_dist_Y = dist_Y[mask]

    # 3. Calculate the actual expansion ratios
    # We only care about valid distances > eps to avoid instability
    valid_indices = valid_dist_X > eps
    final_ratios = valid_dist_Y[valid_indices] / valid_dist_X[valid_indices]

    return {
        "min_expansion": final_ratios.min().item(),
        "max_expansion": final_ratios.max().item(),
    }

def distortion_metric(X, Y, eps=1e-8):
    """
    Empirical distortion between X and Y.

    X, Y: tensors of shape (n, d) or (d,)  (before/after transform)
    eps: numerical stability

    Returns:
        M: estimated distortion, where M = 0 means isometric, M > 0 means distorted.
    """

    # 1. Get Lipschitz bounds
    lipschitz_bounds = get_lipschitz_bounds(X, Y, eps=eps)

    # Get the one that represents the bigger distortion
    distortion = min(lipschitz_bounds["min_expansion"], 1.0 / lipschitz_bounds["max_expansion"])

    # Invert it to get a metric where 1 means isometric, 0 means distorted
    return 1-distortion

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
    C = C @C .T

    # scale by std
    D = np.diag(marginal_stds)
    Sigma = D @ C @ D + 1e-6 * np.eye(d)  # add small value for numerical stability

    # For stability, symmetrize
    Sigma = (Sigma + Sigma.T) / 2
    return Sigma

def compute_A(sigma1: np.ndarray, sigma2: np.ndarray, Sigma1_sqrt = None, Sigma1_inv_sqrt= None, tol: float = 1e-12) -> np.ndarray:
    """
    Compute A = Sigma1^{-1/2} ( Sigma1^{1/2} Sigma2 Sigma1^{1/2} )^{1/2} Sigma1^{-1/2}.
    Works for PSD (possibly singular) Sigma1 and PSD Sigma2.
    Returns a symmetric matrix (numerical symmetrization applied).
    """
    '''# symmetrize inputs
    sigma1 = (sigma1 + sigma1.T) / 2
    sigma2 = (sigma2 + sigma2.T) / 2'''

    if Sigma1_sqrt is None or Sigma1_inv_sqrt is None:
        Sigma1_sqrt = sqrtm(sigma1)
        Sigma1_inv_sqrt = inv(sqrtm(sigma1))

    # middle = Sigma1^{1/2} Sigma2 Sigma1^{1/2}
    middle = Sigma1_sqrt @ sigma2 @ Sigma1_sqrt
    #middle = (middle + middle.T) / 2

    # eigen-decomposition of middle and its sqrt (clip small negative evs due to num error)
    '''w2, V2 = np.linalg.eigh(middle)
    w2_clipped = np.clip(w2, a_min=0.0, a_max=None)
    sqrt_w2 = np.sqrt(w2_clipped)
    middle_sqrt = (V2 * sqrt_w2) @ V2.T'''
    middle_sqrt = sqrtm(middle)


    # assemble A
    A = Sigma1_inv_sqrt @ middle_sqrt @ Sigma1_inv_sqrt
    A = A.real
    #A = (A + A.T) / 2  # ensure symmetry
    # Convert it to np.float32
    A = A.astype(np.float32)
    return A.real

def compute_A_commuting(sigma1: np.ndarray, sigma2: np.ndarray, w1=None, Q=None, w2=None, tol: float = 1e-12) -> np.ndarray:
    """
    Fast computation of A when sigma1 and sigma2 commute.
    sigma1, sigma2 assumed symmetric PSD.
    """
    '''# symmetrize
    sigma1 = (sigma1 + sigma1.T) / 2
    sigma2 = (sigma2 + sigma2.T) / 2'''

    # diagonalize sigma1 if not given
    if w1 is None or Q is None:
        w1, Q = np.linalg.eigh(sigma1)

    # sigma2 in same basis
    if w2 is None:
        w2 = np.diag(Q.T @ sigma2 @ Q)

    # mask nonzero directions
    nonzero = w1 > tol
    lamA = np.zeros_like(w1)

    # check consistency: if w1≈0 then w2 must also≈0
    if np.any((~nonzero) & (np.abs(w2) > tol)):
        raise ValueError("Incompatible: sigma1 has zero eigenvalue where sigma2 is nonzero.")

    lamA[nonzero] = np.sqrt(np.clip(w2[nonzero], tol, None) / w1[nonzero])

    # reconstruct A
    A = (Q * lamA) @ Q.T
    A = A.astype(np.float32)
    return A

def init_solving(x: np.ndarray, model: sklearn.linear_model.LogisticRegression, y_prime,
                 y_prime_confidence):
    # Solve the QP to find the worst-case Lipschitz constant
    n = x.shape[0]
    d = x.shape[1]
    assert model.coef_.shape[1] == d, "Dimension mismatch"
    assert model.coef_.shape[0] == 1, "Only binary classification supported, for now"
    assert y_prime in [0, 1], "Only binary classification supported"

    # Get the linear decision boundary parameters
    w = model.coef_[0]
    b_model = model.intercept_[0]

    # Derive the margin logit from the confidence
    # First, adapt margin concerning if y_prime is 0 or 1
    if y_prime == 1:
        margin = y_prime_confidence
    else:
        margin = 1 - y_prime_confidence
    margin_logit = np.log(margin / (1 - margin))

    return n, d, w, b_model, margin_logit

# Example
if __name__ == "__main__":
    #np.random.seed(0)
    n = 2
    G = np.random.randn(n, n)
    Sigma1 = G @ G.T + 1e-6 * np.eye(n)

    A = np.random.randn(n, n)
    A = A @ A.T + 1e-6 * np.eye(n)                    # PSD

    Sigma2 = A @ Sigma1 @ A.T

    # Retrieve A from Sigma1, Sigma2
    A_retrieved = compute_A(Sigma1, Sigma2)
    print("Original A:")
    print(A)

    print("Retrieved A:")
    print(A_retrieved)

    raise ValueError("Stop here")

    # PD
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




