import numpy as np
import sklearn
import torch
from torch import nn
from scipy.stats import multivariate_normal
import cvxpy as cp

from group_cfx.transforms.probabilistic_transforms import ProbabilisticTransform
from group_cfx.transforms.utils import wasserstein_distance_normals, compute_A, build_covariance_matrix, \
    compute_A_commuting, init_solving


class BaseGaussianTransform(ProbabilisticTransform):
    """
    A base class for Gaussian transforms.
    """
    def __init__(self):
        super().__init__()
        self.prior_mvn : multivariate_normal = None  # Placeholder for prior distribution
        self.posterior_mvn : multivariate_normal = None
        self.A = None
        self.B = None

    def fit_prior(self, X_orig):
        cov = np.cov(X_orig.detach().cpu().numpy(), rowvar=False) + 1e-6 * np.eye(X_orig.shape[1])
        print(np.linalg.eigvalsh(cov))
        self.prior_mvn = multivariate_normal(mean=X_orig.mean(axis=0).detach().cpu().numpy(), cov=np.cov(X_orig.detach().cpu().numpy(), rowvar=False))
        self.build_mvn()
        self.derive_affine_transform()

    def derive_affine_transform(self):
        self.A = torch.tensor(compute_A(self.prior_mvn.cov, self.posterior_mvn.cov))
        self.B = torch.tensor((self.posterior_mvn.mean - self.prior_mvn.mean @ self.A.detach().cpu().numpy().T))

    def forward(self, x):
        #Print type of x matrix
        return self.clip_to_box(x @ self.A.T + self.B)

    def wasserstein_projection_distance(self, X_orig = None):
        m0 = self.prior_mvn.mean
        m1 = self.posterior_mvn.mean

        # Covariances
        C0 = self.prior_mvn.cov
        C1 = self.posterior_mvn.cov

        return wasserstein_distance_normals(m0, C0, m1, C1)

    def load_parameters(self, flatten_new_params):
        # Call super method
        super().load_parameters(flatten_new_params)
        self.build_mvn()
        self.derive_affine_transform()

    def build_mvn(self):
        raise NotImplementedError("This is a Base class that serves as a template. Subclasses should implement build_mvn method.")

    def lipschitz_proxy(self, X_orig = None):
        # By default, use the difference of self.A from identity
        I = torch.eye(self.A.shape[0])
        A_non_param = self.A.detach()
        #print(f"Lipschitz proxy: ||A - I||_F^2 = {((A_non_param - I) ** 2).mean().item()}")
        return ((A_non_param - I) ** 2).mean()

class GaussianTransform(BaseGaussianTransform):
    """
    A simple Gaussian transform that adds Gaussian noise to the input.
    The mean and stddev of the noise are learnable parameters.
    """
    def __init__(self, d):
        super().__init__()
        self.posterior_mu_offset = nn.Parameter(torch.zeros(d))
        self.posterior_marginal_stds = nn.Parameter(torch.zeros(d)+0.00001)
        self.posterior_corr_triang = nn.Parameter(torch.zeros(d * (d - 1) // 2)+0.00001)
        self.xl = [-5.0]*d + [0.001]*d + [-0.999]*(d*(d-1)//2)
        self.xu = [5.0]*d + [1.5]*d + [0.999]*(d*(d-1)//2)

    def build_mvn(self):
        covariance_matrix_posterior = build_covariance_matrix(self.posterior_marginal_stds.detach().cpu().numpy(), self.posterior_corr_triang.detach().cpu().numpy())
        # Check if covariance_matrix_posterior is positive definite
        eigvals = np.linalg.eigvalsh(covariance_matrix_posterior)
        if np.any(eigvals <= 0):
            #print("Warning: Posterior covariance matrix is not positive definite. Adjusting to nearest PD matrix.")
            # Adjust to nearest positive definite matrix (simple fix)
            min_eig = np.min(eigvals)
            covariance_matrix_posterior += (-min_eig + 1e-6) * np.eye(covariance_matrix_posterior.shape[0])
        self.posterior_mvn = multivariate_normal(mean=self.prior_mvn.mean + self.posterior_mu_offset.detach().cpu().numpy(), cov=covariance_matrix_posterior)


class GaussianCommutativeTransform(BaseGaussianTransform) :
    def __init__(self, d):
        super().__init__()
        self.prior_eigenvalues = None
        self.prior_eigenvectors = None
        self.posterior_mu_offset = nn.Parameter(torch.zeros(d))
        # Scaling factor. A single parameter
        self.posterior_eigenvalues  = nn.Parameter(torch.ones(d)-0.5)
        self.xl = [-5.0] * d + [0.00001]*d
        self.xu = [5.0] * d + [1]*d

    def fit_prior(self, X_orig):
        self.prior_mvn = multivariate_normal(mean=X_orig.mean(axis=0).detach().cpu().numpy(),
                                             cov=np.cov(X_orig.detach().cpu().numpy(), rowvar=False))
        self.prior_eigenvalues, self.prior_eigenvectors = np.linalg.eigh(self.prior_mvn.cov)
        self.build_mvn()
        self.derive_affine_transform()
        # Compute eigen decomposition of prior covariance

    def derive_affine_transform(self):
        self.A = torch.tensor(compute_A_commuting(self.prior_mvn.cov, self.posterior_mvn.cov))
        self.B = torch.tensor((self.posterior_mvn.mean - self.prior_mvn.mean @ self.A.detach().cpu().numpy().T))

    def build_mvn(self):
        covariance_matrix_posterior = (self.prior_eigenvectors @
                                      np.diag(self.posterior_eigenvalues.detach().cpu().numpy()) @
                                      self.prior_eigenvectors.T)
        self.posterior_mvn = multivariate_normal(mean= self.prior_mvn.mean + self.posterior_mu_offset.detach().cpu().numpy(), cov=covariance_matrix_posterior)

    def is_cvx(self):
        return True

    def cvxpy_solving(self, x : np.ndarray, model : sklearn.linear_model.LogisticRegression, y_prime, y_prime_confidence,
                      K =1.1, solver = cp.MOSEK) -> float:
        # Convert x to numpy if it's a torch tensor
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()

        eps_reg = 1e-9
        n, d, w_model, b_model, margin_logit = init_solving(x, model, y_prime, y_prime_confidence)

        mu0 = self.prior_mvn.mean.reshape(d, )
        Sigma0 = 0.5 * (self.prior_mvn.cov + self.prior_mvn.cov.T) + eps_reg * np.eye(d)

        # spectral decompose Sigma0
        d0, U = np.linalg.eigh(Sigma0)  # d0 sorted ascending
        # avoid numerical issues
        d0 = np.clip(d0, eps_reg, None)
        sqrt_d0 = np.sqrt(d0)
        inv_sqrt_d0 = 1.0 / sqrt_d0

        # precompute outer products u_i u_i^T as dense constant matrices
        U_cols = [U[:, i:i + 1] for i in range(d)]
        UiUiT = [(U_cols[i] @ U_cols[i].T) for i in range(d)]  # each is (d,d) numpy array

        # CVXPY decision variables
        mu1_offset = cp.Variable(d)
        mu1 = mu0 + mu1_offset  # posterior mean
        s = cp.Variable(d, nonneg=True)  # s_i = sqrt(d1_i) (the sqrt of Sigma1 eigenvalues in basis U)

        # Build Bures term in this parametrization:
        # sum_i (sqrt(d0_i) - s_i)^2  = sum d0 + sum s^2 - 2 sum sqrt(d0) * s
        bures_const = float(np.sum(d0))
        bures = bures_const + cp.sum(cp.square(s)) - 2 * (sqrt_d0 @ s)

        # Objective: ||mu0 - mu1||^2 + bures
        objective = cp.Minimize(cp.sum_squares(mu1_offset) + bures)

        constraints = []

        # Lipschitz upper bound: lambda_max(Sigma0^{-1} Sigma1) <= K^2
        # with our parametrization this is: (s_i^2 / d0_i) <= K^2 -> s_i <= K * sqrt(d0_i)
        constraints.append(s <= (K * sqrt_d0))

        # optional lower bound for bi-Lipschitz (uncomment if needed)
        constraints.append(s >= (1.0 / K) * sqrt_d0)

        # per-sample classification constraints.
        # Express z_i = A (x_i - mu0) + mu1, with A = sum_j (s_j / sqrt_d0_j) * (u_j u_j^T).
        X_centered = (x - mu0.reshape(1, -1))  # (n,d) numpy
        for i in range(n):
            v = X_centered[i]  # numpy vector
            # compute A v as a linear combination of the fixed matrices UiUiT times coefficients c_j = (s_j / sqrt_d0_j)
            # A v = sum_j (s_j / sqrt_d0_j) * (UiUiT[j] @ v)
            # build affine expression for A v
            Av_expr = sum(
                (s[j] * inv_sqrt_d0[j]) * (UiUiT[j] @ v) for j in range(d))  # each term is cp expression (d,)
            zi = Av_expr + mu1  # cp expression (d,)
            logit_i = w_model @ zi + b_model  # scalar cp affine expression
            if int(y_prime) == 1:
                constraints.append(logit_i >= margin_logit)
            else:
                constraints.append(logit_i <= margin_logit)

        prob = cp.Problem(objective, constraints)
        prob.solve(solver=solver)

        # extract results
        mu1_val = mu1.value
        mu1_offset_val = mu1_offset.value
        s_val = np.clip(s.value, 0.0, None)  # numerical safety
        d1_val = s_val ** 2
        Sigma1_val = (U @ np.diag(d1_val) @ U.T)
        # construct A and b
        A = U @ np.diag(s_val * inv_sqrt_d0) @ U.T
        b = mu1_val - A @ mu0

        # update your internal pytorch variables
        with torch.no_grad():
            self.posterior_mu_offset.copy_(
                torch.tensor(mu1_offset_val, dtype=self.posterior_mu.dtype, device=self.posterior_mu.device))
            self.posterior_eigenvalues.copy_(
                torch.tensor(d1_val, dtype=self.posterior_eigenvalues.dtype, device=self.posterior_eigenvalues.device))
            self.build_mvn()
            self.derive_affine_transform()

        # return same structure as before
        return prob


class GaussianScaleTransform(BaseGaussianTransform) :
    def __init__(self, d):
        super().__init__()
        self.posterior_mu_offset = nn.Parameter(torch.zeros(d))
        self.scaling = nn.Parameter(torch.ones(1))
        self.xl = [-3.0] * d + [0.0001]
        self.xu = [3.0] * d + [2.0]

    def build_mvn(self):
        covariance_matrix_posterior = self.prior_mvn.cov
        self.posterior_mvn = multivariate_normal(mean=self.prior_mvn.mean + self.posterior_mu_offset.detach().cpu().numpy(),
                                                 cov=self.scaling.item() * covariance_matrix_posterior)

    def derive_affine_transform(self):
        self.B = torch.tensor((self.posterior_mvn.mean - self.prior_mvn.mean))

    def wasserstein_projection_distance(self, X_orig = None):
        m0 = self.prior_mvn.mean
        m1 = self.posterior_mvn.mean

        # Mean term
        mean_diff = np.linalg.norm(m0 - m1) ** 2 + (self.scaling.item() - 1) ** 2 * np.trace(self.prior_mvn.cov)
        return mean_diff

    def forward(self, x):
        return self.clip_to_box(x + self.B)

    def lipschitz_proxy(self, X_orig):
        return self.scaling.item() - 1.0

    def is_cvx(self):
        return True

    def cvxpy_solving(self, x: np.ndarray, model: sklearn.linear_model.LogisticRegression, y_prime, y_prime_confidence,
                      K=1.1, solver = cp.MOSEK) -> float:
        """
        Optimize a scalar scaling factor s > 0 and a mean offset b (vector) to minimize
        squared W2 distance between x and s*x + b, subject to:
          - bi-Lipschitz: 1/K <= s <= K
          - logistic classification constraint for projected points
        """
        n, d, w_model, b_model, margin_logit = init_solving(x, model, y_prime, y_prime_confidence)


        # Compute empirical mean and covariance
        mu_0 = self.prior_mvn.mean
        Sigma_0 = self.prior_mvn.cov
        tr_Sigma_0 = np.trace(Sigma_0)

        # --- Variables ------------------------------------------------------------
        b = cp.Variable(d)  # mean offset
        # s = cp.Variable()  # scalar scaling
        t = cp.Variable()  # t = sqrt(s)

        # --- W2 objective (closed form) ------------------------------------------
        # W2^2 = ||mu_1 - m_x||^2 + (sqrt(s) - 1)^2 * Tr(Sigma_x)
        mu_1 = mu_0 + b
        w2_cov_term = cp.square(t - 1) * tr_Sigma_0
        objective = cp.Minimize(cp.sum_squares(b) + w2_cov_term)

        # --- Constraints ----------------------------------------------------------
        constraints = [t >= 1 / np.sqrt(K), t <= np.sqrt(K)]

        # --- OT affine map applied to all points ---------------------------------
        # x -> t*x + (b + (1-t)*m_x)
        b_ot = b + (1 - t) * mu_0
        fX = t * x + b_ot  # shape (n,d)

        # Logistic classification constraints (linearized)
        margin_logit = np.log(y_prime_confidence / (1 - y_prime_confidence))
        for i in range(n) :
            logits = w_model @ fX[i, :] + b_model
            if y_prime == 1:
                constraints.append(logits >= margin_logit)
            else:
                constraints.append(logits <= margin_logit)

        # --- Solve ----------------------------------------------------------------
        prob = cp.Problem(objective, constraints)
        prob.solve(solver=solver)
        if prob.status != cp.OPTIMAL:
            print("Warning: Problem did not converge to optimal solution")

        # Update parameters in torch
        with torch.no_grad():
            self.scaling.copy_(torch.tensor(t.value**2, dtype=torch.float32))
            self.posterior_mu_offset.copy_(torch.tensor(b.value, dtype=torch.float32))

        self.build_mvn()
        self.derive_affine_transform()

        return prob