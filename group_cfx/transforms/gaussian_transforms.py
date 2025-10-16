import numpy as np
import sklearn
import torch
from pyomo.core.expr.ndarray import NumericNDArray
from scipy.linalg import sqrtm
from torch import nn
from scipy.stats import multivariate_normal
import cvxpy as cp
import pyomo.environ as pyo

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
        self.prior_mvn = multivariate_normal(mean=X_orig.mean(axis=0).detach().cpu().numpy(), cov=np.cov(X_orig.detach().cpu().numpy(), rowvar=False))
        #self.prior_mvn = multivariate_normal(mean=[0, 0], cov=np.eye(2))
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
        self.posterior_mu = nn.Parameter(torch.zeros(d))
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
        self.posterior_mvn = multivariate_normal(mean=self.posterior_mu.detach().cpu().numpy(), cov=covariance_matrix_posterior)


    def cvxpy_solving(self, x, model,
                          y_prime, y_prime_confidence, K=1.1,
                      eps_reg=1e-9):
        """
        Convex optimization of target Gaussian (mu1, Sigma1) using Bures SDP.
        Then reconstruct OT map (A,b) and evaluate transformed samples.
        """

        n, d, w_model, b_model, margin_logit = init_solving(x, model, y_prime, y_prime_confidence)

        mu0 = self.prior_mvn.mean
        Sigma0 = self.prior_mvn.cov + eps_reg * np.eye(d)

        # Decision variables
        mu1 = cp.Variable(d)
        Sigma1 = cp.Variable((d, d), symmetric=True)
        X = cp.Variable((d, d), symmetric=True)

        # Constraints
        constraints = [Sigma1 >> 0, X >> 0,
                       cp.bmat([[Sigma0, X],
                                [X, Sigma1]]) >> 0]
        constraints += [Sigma1 << (K ** 2) * Sigma0,
                        Sigma1 >> (1.0 / K ** 2) * Sigma0]

        # Classification at mean
        if int(y_prime) == 1:
            constraints.append(w_model @ mu1 + b_model >= margin_logit)
        else:
            constraints.append(w_model @ mu1 + b_model <= margin_logit)

        # Objective
        objective = (cp.sum_squares(mu0 - mu1)
                     + np.trace(Sigma0)
                     + cp.trace(Sigma1)
                     - 2 * cp.trace(X))

        prob = cp.Problem(cp.Minimize(objective), constraints)
        prob.solve(solver=cp.CVXOPT)

        Sigma1_val = Sigma1.value + eps_reg * np.eye(d)  # numerical safety

        print(Sigma0)
        print(Sigma1_val)
        print(X.value)

        with torch.no_grad():
            self.posterior_mu.copy_(torch.tensor(mu1.value, dtype=self.posterior_mu.dtype, device=self.posterior_mu.device))
            # Decompose Sigma1_val into marginal stds and correlations
            marginal_stds = np.sqrt(np.diag(Sigma1_val))
            corr_matrix = Sigma1_val / np.outer(marginal_stds, marginal_stds)
            # Extract the lower triangular part (excluding diagonal) as a flat array
            tril_indices = np.tril_indices(d, k=-1)
            corr_triang = corr_matrix[tril_indices]
            self.posterior_marginal_stds.copy_(torch.tensor(marginal_stds, dtype=self.posterior_marginal_stds.dtype, device=self.posterior_marginal_stds.device))
            self.posterior_corr_triang.copy_(torch.tensor(corr_triang, dtype=self.posterior_corr_triang.dtype, device=self.posterior_corr_triang.device))
            self.build_mvn()
            self.derive_affine_transform()

        return prob

    def pyomo_solving(self, x, model,
                              y_prime=1, y_prime_confidence=0.9, K=1.1,
                              solver_name='ipopt', eps_reg=1e-9):
        """
        Optimize A and b with exact W2 objective (nonconvex) and per-sample classification.
        """
        n,d, w_model, b_model, margin_logit = init_solving(x, model, y_prime, y_prime_confidence)

        x = np.asarray(x)
        mu0 = np.asarray(self.prior_mvn.mean).reshape(d, )
        Sigma0 = 0.5 * (np.asarray(self.prior_mvn.cov) + np.asarray(self.prior_mvn.cov).T) + eps_reg * np.eye(d)

        model = pyo.ConcreteModel()
        model.D = pyo.RangeSet(0, d - 1)

        # Decision variables
        model.A = pyo.Var(model.D, model.D, domain=pyo.Reals)
        model.b = pyo.Var(model.D, domain=pyo.Reals)

        # Derived: transformed points (per dimension)
        # i: Instance
        # j: Dimension
        def Z(i, j):
            return sum(model.A[j,k] * float(x[i, k]) for k in range(d)) + model.b[j]

        # Classification constraints on transformed points
        model.N = pyo.RangeSet(0, n - 1)

        def cls_rule(m, i):
            logit = sum(w_model[j] * Z(i, j) for j in range(d)) + b_model

            if int(y_prime) == 1:
                return logit >= margin_logit
            else:
                return logit <= margin_logit

        model.cls = pyo.Constraint(model.N, rule=cls_rule)

        # Optional: bilipschitz constraints on transformed points
        def lip_upper_rule(m, i, j):
            if i >= j:
                return pyo.Constraint.Skip
            lhs = sum((Z(i, k) - Z(j, k)) ** 2 for k in range(d))
            rhs = (K ** 2) * np.linalg.norm(x[i] - x[j]) ** 2
            return lhs <= rhs

        model.lip_up = pyo.Constraint(model.N, model.N, rule=lip_upper_rule)

        def lip_lower_rule(m, i, j):
            if i >= j:
                return pyo.Constraint.Skip
            lhs = sum((Z(i, k) - Z(j, k)) ** 2 for k in range(d))
            rhs = (1.0 / K ** 2) * np.linalg.norm(x[i] - x[j]) ** 2
            return lhs >= rhs

        model.lip_low = pyo.Constraint(model.N, model.N, rule=lip_lower_rule)

        # Objective: exact W2^2 between N(mu0,Sigma0) and N(mu1,Sigma1=A Sigma0 A^T)

        # Add constraints to the model. Sigma1 shouldn't have negative values
        def sigma1_expr(m, i, j):
            s = 0
            for k in range(d):
                for l in range(d):
                    s += m.A[i, k] * Sigma0[k, l] * m.A[j, l]
            return s
            #return sum(m.A[i, k] * Sigma0[k, l] * m.A[j, l] for k in range(d) for l in range(d))

        def sigma1_pos_rule(m, i, j):
            if i != j:
                return pyo.Constraint.Skip
            return sigma1_expr(m, i, j) >= 0
        model.sigma1_pos = pyo.Constraint(model.D, model.D, rule=sigma1_pos_rule)

        # Check that correlation is between -1 and 1 (strictly)
        def corr_rule(m, i, j):
            if i >= j:
                return pyo.Constraint.Skip
            return pyo.inequality(-0.99, sigma1_expr(m, i, j) / (pyo.sqrt(sigma1_expr(m, i, i) * sigma1_expr(m, j, j) + eps_reg)), 0.99)

        model.corr_constr = pyo.Constraint(model.D, model.D, rule=corr_rule)

        # Pyomo objective function using Python function
        def obj_rule(m):
            # First, obtain mu1 and Sigma1
            mu1 = m.A @ mu0 + m.b
            # mu1 without numpy operations:
            Sigma1 = m.A @ Sigma0 @ np.transpose(m.A)
            #Sigma1 = [[sigma1_expr(m, i, j) for j in range(d)] for i in range(d)]
            return sum((mu0 - mu1) ** 2) #+ Sigma1[1][1]



        model.obj = pyo.Objective(rule=obj_rule, sense=pyo.minimize )


        # Sigma 1 should be PSD
        def sigma1_psd_rule(m, v):
            Sigma1 = m.A @ Sigma0 @ np.transpose(m.A)
            vec = np.array([v[i] for i in range(d)])
            prod = vec @ Sigma1 @ vec
            return prod >= 0.0
        #model.sigma1_psd = pyo.Constraint(model.D, rule=sigma1_psd_rule)

        # Solve
        solver = pyo.SolverFactory("ipopt")
        solver.solve(model, tee=False)

        # Extract solution
        A_val = np.array([[pyo.value(model.A[i, j]) for j in range(d)] for i in range(d)])
        b_val = np.array([pyo.value(model.b[j]) for j in range(d)])



        # Fill the parameters back into the model
        with torch.no_grad():
            self.posterior_mu.copy_(torch.tensor(A_val @ mu0 + b_val, dtype=self.posterior_mu.dtype, device=self.posterior_mu.device))

            print("Learned A", A_val)
            print("Learned b", b_val)
            Sigma1_val = A_val @ Sigma0 @ A_val.T
            print("Learned Sigma1", Sigma1_val)
            marginal_stds = np.sqrt(np.diag(Sigma1_val))
            corr_matrix = Sigma1_val / np.outer(marginal_stds, marginal_stds)
            print("corr_matrix", corr_matrix)
            tril_indices = np.tril_indices(d, k=-1)
            corr_triang = corr_matrix[tril_indices]
            self.posterior_marginal_stds.copy_(torch.tensor(marginal_stds, dtype=self.posterior_marginal_stds.dtype, device=self.posterior_marginal_stds.device))
            self.posterior_corr_triang.copy_(torch.tensor(corr_triang, dtype=self.posterior_corr_triang.dtype, device=self.posterior_corr_triang.device))

            self.build_mvn()
            self.derive_affine_transform()

        return model


class GaussianPolynomialTransform(BaseGaussianTransform) :
    def __init__(self, d):
        super().__init__()
        self.prior_eigenvalues = None
        self.prior_eigenvectors = None
        self.posterior_mu = nn.Parameter(torch.zeros(d))
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
        self.posterior_mvn = multivariate_normal(mean=self.posterior_mu.detach().cpu().numpy(), cov=covariance_matrix_posterior)

    def cvxpy_solving(self, x : np.ndarray, model : sklearn.linear_model.LogisticRegression, y_prime, y_prime_confidence,
                      K =1.1) -> float:
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
        mu1 = cp.Variable(d)
        s = cp.Variable(d, nonneg=True)  # s_i = sqrt(d1_i) (the sqrt of Sigma1 eigenvalues in basis U)

        # Build Bures term in this parametrization:
        # sum_i (sqrt(d0_i) - s_i)^2  = sum d0 + sum s^2 - 2 sum sqrt(d0) * s
        bures_const = float(np.sum(d0))
        bures = bures_const + cp.sum(cp.square(s)) - 2 * (sqrt_d0 @ s)

        # Objective: ||mu0 - mu1||^2 + bures
        objective = cp.Minimize(cp.sum_squares(mu0 - mu1) + bures)

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
        prob.solve(solver=cp.SCS)

        # extract results
        mu1_val = mu1.value
        s_val = np.clip(s.value, 0.0, None)  # numerical safety
        d1_val = s_val ** 2
        Sigma1_val = (U @ np.diag(d1_val) @ U.T)
        # construct A and b
        A = U @ np.diag(s_val * inv_sqrt_d0) @ U.T
        b = mu1_val - A @ mu0

        # transformed samples and diagnostics
        z = x @ A.T + b
        logits = z @ w_model + b_model
        if int(y_prime) == 1:
            correct_mask = logits >= margin_logit
        else:
            correct_mask = logits <= margin_logit

        # update your internal pytorch variables
        with torch.no_grad():
            self.posterior_mu.copy_(
                torch.tensor(mu1_val, dtype=self.posterior_mu.dtype, device=self.posterior_mu.device))
            self.posterior_eigenvalues.copy_(
                torch.tensor(d1_val, dtype=self.posterior_eigenvalues.dtype, device=self.posterior_eigenvalues.device))
            self.build_mvn()
            self.derive_affine_transform()

        # return same structure as before
        return prob


class GaussianNoScaleTransform(BaseGaussianTransform) :
    def __init__(self, d):
        super().__init__()
        self.posterior_mu = nn.Parameter(torch.zeros(d))
        self.xl = [-5.0] * d
        self.xu = [5.0] * d

    def build_mvn(self):
        covariance_matrix_posterior = self.prior_mvn.cov
        self.posterior_mvn = multivariate_normal(mean=self.posterior_mu.detach().cpu().numpy(), cov=covariance_matrix_posterior)

    def derive_affine_transform(self):
        self.B = torch.tensor((self.posterior_mvn.mean - self.prior_mvn.mean))

    def wasserstein_projection_distance(self, X_orig):
        m0 = self.prior_mvn.mean
        m1 = self.posterior_mvn.mean

        # Mean term
        mean_diff = np.linalg.norm(m0 - m1) ** 2

        return mean_diff

    def forward(self, x):
        return self.clip_to_box(x + self.B)

    def lipschitz_proxy(self, X_orig):
        return 0.0
