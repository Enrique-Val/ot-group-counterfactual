import cvxpy as cp
import matplotlib.pyplot as plt
import numpy as np
import sklearn
import torch
from scipy.linalg import sqrtm
from scipy.special import digamma
from scipy.stats._multivariate import multivariate_normal_gen
from torch import nn
from scipy.stats import multivariate_normal, multivariate_t
from sklearn.mixture import GaussianMixture

from group_cfx.transforms.functional_transforms import BaseTransform
from group_cfx.transforms.utils import wasserstein_distance_normals, compute_A, build_covariance_matrix, \
    bi_lipschitz_metric, init_solving


class ProbabilisticTransform(BaseTransform):
    """
    A base class for probabilistic transforms.
    """
    def __init__(self):
        super().__init__()
        self.xl = None
        self.xu = None

    def wasserstein_projection_distance(self, X_orig = None) -> float:
        raise NotImplementedError()

    def lipschitz_proxy(self, X_orig = None) -> float:
        if X_orig is None:
            raise NotImplementedError()
        else:
            # Called method of parent class
            return super().lipschitz_proxy(X_orig)


class GMMForwardTransform(ProbabilisticTransform) :
    """
    A Gaussian Mixture Model (GMM) based transform.
    The GMM parameters (means, covariances, and weights) are learnable.
    """
    def __init__(self, d, n_components=3):
        super().__init__()
        self.n_components = n_components
        self.prior_gmm : list[multivariate_normal_gen] = []  # Placeholder for prior distribution
        self.prior_gmm_skl : GaussianMixture = None   # Placeholder for sklearn GMM
        self.posterior_gmm : list[multivariate_normal_gen] = []
        self.log_weights = []
        self.A = []
        self.B = []
        self.means_offset = nn.Parameter(torch.zeros(n_components, d))
        self.marginal_stds = nn.Parameter(torch.ones(n_components, d)*0.1)
        self.corr_triang = nn.Parameter(torch.zeros(n_components, d * (d - 1) // 2)+0.1)
        self.xl = [-8.0]* (n_components * d) + [0.00001]*(n_components * d) + [-0.99999]*(n_components * (d*(d-1)//2))
        self.xu = [8.0]* (n_components * d) + [1]*(n_components * d) + [0.99999]*(n_components * (d*(d-1)//2))

        # Accelerate computing
        self.prior_sqrt_cov = None
        self.prior_inv_sqrt_cov = None

    def fit_prior(self, X_orig):
        # Fit a GMM to the original data using sklearn
        X_np = X_orig.detach().cpu().numpy()
        gmm = None
        for eps in [1e-6, 1e-5, 1e-4, 1e-3]:
            try:
                gmm = GaussianMixture(n_components=self.n_components, covariance_type='full', random_state=0, reg_covar=eps)
                gmm.fit(X_np)
                break
            except ValueError as e:
                print(f"LinAlgError with reg_covar={eps}, trying larger value.")
                if eps == 1e-3:
                    raise e
        self.prior_gmm_skl = gmm
        self.prior_gmm = []
        self.prior_sqrt_cov = []
        self.prior_inv_sqrt_cov = []
        for i in range(self.n_components):
            mvn = multivariate_normal(mean=gmm.means_[i], cov=gmm.covariances_[i])
            self.prior_gmm.append(mvn)
            self.log_weights.append(np.log(gmm.weights_[i] + 1e-10))  # Avoid log(0)
            # Precompute sqrt and inv sqrt of covariance
            sqrt_cov = sqrtm(gmm.covariances_[i])
            inv_sqrt_cov = np.linalg.inv(sqrt_cov)
            self.prior_sqrt_cov.append(sqrt_cov)
            self.prior_inv_sqrt_cov.append(inv_sqrt_cov)
        self.build_gmm()
        self.derive_affine_transform()

    def load_parameters(self, flatten_new_params):
        # Call super method
        super().load_parameters(flatten_new_params)
        self.build_gmm()
        self.derive_affine_transform()

    def build_gmm(self):
        self.posterior_gmm = []
        for i in range(self.n_components):
            cov_matrix = build_covariance_matrix(self.marginal_stds[i].detach().cpu().numpy(), self.corr_triang[i].detach().cpu().numpy())
            mvn = multivariate_normal(mean=self.means_offset[i].detach().cpu().numpy() + self.prior_gmm[i].mean , cov=cov_matrix)
            self.posterior_gmm.append(mvn)
        # Print means and covariances
        '''for i in range(self.n_components):
            print("Component", i)
            print("Posterior mvn mean:", self.posterior_gmm[i].mean)
            print("Posterior mvn covariance matrix:", self.posterior_gmm[i].cov)

        print()'''

    def derive_affine_transform(self):
        self.A = []
        self.B = []
        for i in range(self.n_components):
            A_i_array = compute_A(self.prior_gmm[i].cov, self.posterior_gmm[i].cov, self.prior_sqrt_cov[i], self.prior_inv_sqrt_cov[i])
            A_i = torch.tensor(A_i_array)
            B_i = torch.tensor((self.posterior_gmm[i].mean - self.prior_gmm[i].mean @ A_i_array.T))
            self.A.append(A_i)
            self.B.append(B_i)

    def forward(self, x):
        # Compute responsibilities
        x_np = x.detach().cpu().numpy()
        log_probs = np.array([mvn.logpdf(x_np) for mvn in self.prior_gmm]).T
        log_weights = np.array(self.log_weights)
        log_responsibilities = log_probs + log_weights
        # Random approach. Sample one component according to responsibilities
        max_log_responsibilities = log_responsibilities.max(axis=1, keepdims=True)
        responsibilities = np.exp(log_responsibilities - max_log_responsibilities)
        responsibilities /= responsibilities.sum(axis=1, keepdims=True)
        component_choices = [np.random.choice(self.n_components, p=responsibilities[i]) for i in range(x_np.shape[0])]
        # Transform each point according to all components. Then select the transformed point according to the sampled component
        x_transformed_all = np.ndarray(shape=(x_np.shape[0], x_np.shape[1], self.n_components))
        for i in range(self.n_components):
            x_transformed_all[:, :, i] = (x_np @ self.A[i].T.numpy()) + self.B[i].numpy()
        # Select the transformed points
        x_transformed_selected = np.array([x_transformed_all[j, :, component_choices[j]] for j in range(x_np.shape[0])])
        return self.clip_to_box(torch.tensor(x_transformed_selected, dtype=x.dtype, device=x.device))

    def forward_probabilistic(self ,x) -> list[multivariate_normal_gen]:
        # Compute responsibilities
        x_np = x.detach().cpu().numpy()
        log_probs = np.array([mvn.logpdf(x_np) for mvn in self.prior_gmm]).T
        log_weights = np.array(self.log_weights)
        log_responsibilities = log_probs + log_weights
        # Normalize to get responsibilities
        max_log_responsibilities = log_responsibilities.max(axis=1, keepdims=True)
        responsibilities = np.exp(log_responsibilities - max_log_responsibilities)
        responsibilities /= responsibilities.sum(axis=1, keepdims=True)
        # Build list of posterior distributions for each input point
        posterior_list = []
        for i in range(x_np.shape[0]):
            # For each point, build a mixture of Gaussians with the posterior components weighted by responsibilities
            means = []
            covariances = []
            weights = []
            for j in range(self.n_components):
                means.append(self.posterior_gmm[j].mean)
                covariances.append(self.posterior_gmm[j].cov)
                weights.append(responsibilities[i, j])
            # Create a GaussianMixture object to represent the mixture
            gmm_post = GaussianMixture(n_components=self.n_components, covariance_type='full')
            gmm_post.means_ = np.array(means)
            gmm_post.covariances_ = np.array(covariances)
            gmm_post.weights_ = np.array(weights)
            gmm_post.precisions_cholesky_ = np.array([np.linalg.cholesky(np.linalg.inv(cov)) for cov in covariances])
            posterior_list.append(gmm_post)
        return posterior_list

    def wasserstein_projection_distance(self, X_orig = None):
        # Compute pairwise distances between all components
        W = np.zeros((self.n_components, ))
        for i in range(self.n_components):
            m0 = self.prior_gmm[i].mean
            m1 = self.posterior_gmm[i].mean

            # Covariances
            C0 = self.prior_gmm[i].cov
            C1 = self.posterior_gmm[i].cov

            # Sqrt of C0
            sqrt_C0 = self.prior_sqrt_cov[i]

            W[i] = wasserstein_distance_normals(m0, C0, m1, C1, sqrt_C0=sqrt_C0)

        # Weight by mixture weights
        weights_prior = np.exp(np.array(self.log_weights))
        weights_prior /= weights_prior.sum()
        weighted_wasserstein = np.sum(W * weights_prior)
        return weighted_wasserstein

    def lipschitz_proxy(self, X_orig = None) -> float:
        # Compute mean of A matrices weighted by mixture weights
        A_mean = torch.zeros_like(self.A[0])
        weights_prior = np.exp(torch.tensor(self.log_weights))
        weights_prior /= weights_prior.sum()
        for i in range(self.n_components):
            A_mean += weights_prior[i] * self.A[i]
        # Compute spectral norm of A_mean (A is symmetric PSD)
        sym_A = 0.5 * (A_mean + A_mean.T)
        with torch.no_grad():
            s, _ = torch.linalg.eigh(sym_A)
            s = torch.sqrt(torch.clamp(s, 0.0))
            lipschitz_upper = s.max().item()
            lipschitz_lower = s.min().item()
            lipschitz = np.min([1 / lipschitz_upper, lipschitz_lower])
        return 1 - lipschitz

        '''if X_orig is None:
            # Compute the bi-Lipschitz metric by sampling from the joint distribution
            X_orig = self.prior_gmm_skl.sample(1000)[0]
        return super().lipschitz_proxy(torch.Tensor(X_orig))'''


    def is_cvx(self):
        return True

    def cvxpy_solving(self, x : np.ndarray, model : sklearn.linear_model.LogisticRegression, y_prime, y_prime_confidence,
                      K =1.1, solver = cp.MOSEK) -> float:
        # Convert x to numpy if it's a torch tensor
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()

        eps_reg = 1e-9
        n, d, w_model, b_model, margin_logit = init_solving(x, model, y_prime, y_prime_confidence)

        constraints = []

        mu1_list = []
        Sigma1_list = []
        A_list = []
        objective_list = []
        #A_mu = cp.Variable((d, d), PSD=True)
        #b_mu = cp.Variable(d)
        for j,component in enumerate(self.prior_gmm):
            mu0 = component.mean.reshape(d, )
            Sigma0 = component.cov
            weight = np.exp(self.log_weights[j])

            # decision variables for Q = (mu1, Sigma1) and coupling R
            mu1 = cp.Variable(d)
            #mu1 = b_mu + A_mu @ mu0
            mu1_list.append(mu1)
            # Sigma1 = cp.Variable((d, d), PSD=True)
            A = cp.Variable((d, d), PSD=True)
            A_list.append(A)
            R = Sigma0 @ A
            Sigma1 = cp.Variable((d, d), PSD=True)
            Sigma1_list.append(Sigma1)

            # convex PSD constraint for Gaussian coupling
            constraints.append(cp.bmat([[Sigma0, R],
                                    [R.T, Sigma1]]) >> 0)

            # Wasserstein-2 objective
            bures_const = np.trace(Sigma0)
            objective_i = weight*(cp.sum_squares(mu1 - mu0)
                                    + cp.trace(Sigma1)
                                    + bures_const
                                    - 2 * cp.trace(R))
            objective_list.append(objective_i)

            # affine transport map from P→Q: T(x)=mu1 + R.T @ Sigma0_inv @ (x - mu0)
            X_centered = x - mu0
            Z_expr = X_centered @ A + mu1

            # classifier constraints
            logits_expr = Z_expr @ w_model + b_model
            if int(y_prime) == 1:
                constraints.append(logits_expr >= margin_logit)
            else:
                constraints.append(logits_expr <= margin_logit)

            # Bilipschitz constraints on covariance ratio
            #constraints.append(A >> (1 / K) * np.eye(d))
            #constraints.append(A << K * np.eye(d))
        objective = cp.Minimize(cp.sum(objective_list))

        A_mean = sum([np.exp(self.log_weights[j]) * A_list[j] for j in range(self.n_components)])

        # Bilipschitz constraints on covariance ratio
        constraints.append(A_mean >> (1 / K) * np.eye(d))
        constraints.append(A_mean << K * np.eye(d))

        '''# Bilipschitz constraints for offset
        constraints.append(A_mu >> (1 / K) * np.eye(d))
        constraints.append(A_mu << K * np.eye(d))'''

        prob = cp.Problem(objective, constraints)
        prob.solve(solver=solver)
        if prob.status != cp.OPTIMAL:
            print("Warning: Problem did not converge")

        with torch.no_grad():
            # Update model parameters with the solution
            marginal_stds = []
            corr_triang = []
            means_offset = []
            self.A = []
            self.B = []
            for j in range(self.n_components):
                mu1_sol = mu1_list[j].value
                A_sol = A_list[j].value
                # Compute B matrix
                B_sol = mu1_sol - self.prior_gmm[j].mean @ A_sol.T

                # Manually set affine transforms
                self.A.append(torch.tensor(A_sol))
                self.B.append(torch.tensor(B_sol))

                # Update mean offset
                new_mean_offset = mu1_sol - self.prior_gmm[j].mean
                means_offset.append(new_mean_offset)

                # Update covariance parameters from A_sol
                # Compute posterior covariance
                Sigma1_sol = Sigma1_list[j].value
                # Decompose Sigma1_sol to stds and correlations
                stds = np.sqrt(np.diag(Sigma1_sol))
                corrs = Sigma1_sol / (stds[:, None] @ stds[None, :])
                # Update parameters
                marginal_stds.append(stds)
                # Extract lower-triangular correlations
                tril_indices = np.tril_indices(d, k=-1)
                corr_tril = corrs[tril_indices]
                corr_triang.append(corr_tril)
            # Rebuild GMM and affine transforms
            self.means_offset.copy_(torch.tensor(np.array(means_offset), dtype=self.means_offset.dtype, device=self.means_offset.device))
            self.marginal_stds.copy_(torch.tensor(np.array(marginal_stds), dtype=self.marginal_stds.dtype, device=self.marginal_stds.device))
            self.corr_triang.copy_(torch.tensor(np.array(corr_triang), dtype=self.corr_triang.dtype, device=self.corr_triang.device))
            self.build_gmm()

'''
# Solve the optimal transport problem using the Hungarian algorithm
# Basically, solve the assignment problem for the mixtures
# This is a potential alternative to having 1-to-1 correspondence between components
from scipy.optimize import linear_sum_assignment
row_ind, col_ind = linear_sum_assignment(W)
optimal_cost = W[row_ind, col_ind].sum()
return optimal_cost
'''

def ensure_positive_definite(matrix, epsilon=1e-6, method = 'diagonal_shift'):
    """Ensure the matrix is positive definite by adding a small value to its diagonal."""
    if method == 'diagonal_shift':
        # Add epsilon to the diagonal elements
        matrix = (matrix + matrix.T) / 2  # Ensure symmetry
        adjusted_matrix = matrix + epsilon * np.eye(matrix.shape[0])
        return adjusted_matrix
    elif method == 'eigenvalue_clipping':
        # Eigenvalue decomposition
        eigvals, eigvecs = np.linalg.eigh(matrix)
        # Clip negative eigenvalues to a small positive value
        eigvals_clipped = np.clip(eigvals, epsilon, None)
        # Reconstruct the matrix
        adjusted_matrix = eigvecs @ np.diag(eigvals_clipped) @ eigvecs.T
        return adjusted_matrix


def fit_multivariate_t(X, nu_init=10.0, tol=1e-6, max_iter=100):
    """
    Fit multivariate Student-t distribution to data X (n x d).
    Returns mean, scale matrix, and degrees of freedom.
    """

    X = np.asarray(X)
    n, d = X.shape
    mu = np.mean(X, axis=0)
    Sigma = np.cov(X, rowvar=False)
    nu = nu_init

    for _ in range(max_iter):
        diff = X - mu
        inv_Sigma = np.linalg.inv(Sigma)
        Q = np.sum(diff @ inv_Sigma * diff, axis=1)
        w = (nu + d) / (nu + Q)

        # Update parameters
        mu_new = np.sum(w[:, None] * X, axis=0) / np.sum(w)
        diff = X - mu_new
        Sigma_new = (diff.T @ (w[:, None] * diff)) / n

        # Update nu via Newton-Raphson
        def f(nu_val):
            return -digamma(nu_val / 2) + np.log(nu_val / 2) + 1 + np.mean(
                np.log(w) - w
            )

        def fprime(nu_val):
            from scipy.special import polygamma
            return -0.5 * polygamma(1, nu_val / 2) + 1 / nu_val

        for _ in range(10):
            nu_new = nu - f(nu) / fprime(nu)
            if abs(nu_new - nu) < 1e-6:
                break
            nu = min(max(nu_new, 1e-3), 50)  # Keep nu in a reasonable range

        # Check convergence
        if (
            np.linalg.norm(mu_new - mu) < tol
            and np.linalg.norm(Sigma_new - Sigma) < tol
        ):
            mu, Sigma = mu_new, Sigma_new
            break

        mu, Sigma = mu_new, Sigma_new

    Sigma = ensure_positive_definite(Sigma, method = 'eigenvalue_clipping')
    mvt_scipy = multivariate_t(loc=mu, shape=Sigma, df=nu)

    return mvt_scipy


# Example:
# X = np.random.randn(500, 3)
# mu, Sigma, nu = fit_multivariate_t(X)
# print(mu, Sigma, nu)



class TStudentTransform(ProbabilisticTransform) :
    """
    A multivariate t-Student distribution based transform.
    The t-Student parameters (mean, covariance, degrees of freedom) are learnable.
    """
    def __init__(self, d):
        super().__init__()
        self.prior_t : multivariate_t = None  # Placeholder for prior distribution
        self.posterior_t : multivariate_t = None
        self.cross_covariance_matrix = None
        self.mean_offset = nn.Parameter(torch.zeros(d))
        self.marginal_stds = nn.Parameter(torch.ones(d)*0.1)
        self.corr_triang = nn.Parameter(torch.zeros(d * (d - 1) // 2)+0.01)
        self.cross_corr = nn.Parameter(torch.zeros(d,d)+0.1)
        self.xl = [-5.0]* d + [0.00001]*d + [-0.99999]*(d*(d-1)//2) + [-0.99999]*d**2
        self.xu = [5.0]* d + [1]*d + [0.99999]*(d*(d-1)//2) + [0.99999]*d**2


    def build_joint(self):
        # Build the posterior multivariate t distribution and the cross-covariance matrix
        # We assume that prior_t is already set
        cov_matrix = build_covariance_matrix(self.marginal_stds.detach().cpu().numpy(), self.corr_triang.detach().cpu().numpy())
        self.posterior_t = multivariate_t(loc=self.mean_offset.detach().cpu().numpy() + self.prior_t.loc, shape=cov_matrix, df=self.prior_t.df)
        d = self.mean_offset.shape[0]
        # Cross-correlation matrix from lower-triangular parameter
        cross_corr = np.tanh(self.cross_corr.detach().cpu().numpy())

        # Cholesky factors of marginal covariances
        Lx = np.linalg.cholesky(self.prior_t.shape)
        Ly = np.linalg.cholesky(cov_matrix)

        print(cross_corr)

        # construct cross-covariance ensuring global PD
        self.cross_covariance_matrix = Ly @ cross_corr @ Lx.T + 1e-6 * np.eye(d)

        # Enforce spectral norm < 1
        u, s, vh = np.linalg.svd(self.cross_covariance_matrix)
        s_clipped = np.clip(s, 1e-6, 0.99)
        self.cross_covariance_matrix = u @ np.diag(s_clipped) @ vh


        # Check if cross-covariance is valid
        eigvals = np.linalg.eigvalsh(self.cross_covariance_matrix)
        if np.any(eigvals <= 0):
            print(eigvals)
            raise ValueError("Cross-covariance matrix is not positive definite.")

        # Check that the joint covariance is positive definite
        joint_cov = self.get_joint().shape
        print("Joint covariance matrix:\n", joint_cov)
        # If there is a negative eigenvalue, raise Error
        eigvals = np.linalg.eigvalsh(joint_cov)
        if np.any(eigvals <= 0):
            raise ValueError("Joint covariance matrix is not positive definite.")


    def fit_prior(self, X_orig):
        # Fit a Gaussian to the original data using sklearn
        self.prior_t = fit_multivariate_t(X_orig)
        self.build_joint()

    def load_parameters(self, flatten_new_params):
        # Call super method
        super().load_parameters(flatten_new_params)
        self.build_joint()

    def forward_probabilistic(self, x):
        # Probabilistic forward pass: sample from the conditional distribution
        x_np = x.detach().cpu().numpy()
        d = self.mean_offset.shape[0]
        n = x_np.shape[0]
        mean_cond = np.zeros((n, d))
        cov_cond = np.zeros((n, d, d))
        dof_cond = np.zeros(n)
        inv_prior_shape = np.linalg.inv(self.prior_t.shape)
        for i in range(n):
            diff = (x_np[i] - self.prior_t.loc).reshape(-1, 1)
            mean_cond[i] = self.posterior_t.loc + (self.cross_covariance_matrix @ inv_prior_shape @ diff).reshape((1, -1))
            cov_cond_ns = self.posterior_t.shape - self.cross_covariance_matrix @ inv_prior_shape @ self.cross_covariance_matrix.T
            dof_incr = diff.T @ inv_prior_shape @ diff
            scaling_factor = (self.prior_t.df + dof_incr) / (self.prior_t.df + d)
            assert scaling_factor > 0, "Scaling factor must be positive"
            cov_cond[i] = scaling_factor * cov_cond_ns
            cov_cond[i] = 0.5 * (cov_cond[i] + cov_cond[i].T)
            # Enforce positive definiteness
            cov_cond[i] = ensure_positive_definite(cov_cond[i], method = 'diagonal_shift', epsilon=1e-2)
            dof_cond[i] = self.prior_t.df + d
        # Build n multivariate t distributions
        mv_t_list = []
        for i in range(n):
            #print(cov_cond[i])
            mv_t_list.append(multivariate_t(loc=mean_cond[i], shape=cov_cond[i], df=dof_cond[i]))
        return mv_t_list

    def get_joint(self) :
        # Build the joint distribution of (X, X')
        d = self.mean_offset.shape[0]
        joint_mean = np.concatenate([self.prior_t.loc, self.posterior_t.loc])
        joint_cov = np.zeros((2*d, 2*d))
        joint_cov[:d, :d] = self.prior_t.shape
        joint_cov[d:, d:] = self.posterior_t.shape
        joint_cov[:d, d:] = self.cross_covariance_matrix
        joint_cov[d:, :d] = self.cross_covariance_matrix.T
        # Print eigenvalues
        eigvals = np.linalg.eigvalsh(joint_cov)
        print(eigvals)
        #joint_cov = ensure_positive_definite(joint_cov, method = 'eigenvalue_clipping')
        joint_t = multivariate_t(loc=joint_mean, shape=joint_cov, df=self.prior_t.df)
        return joint_t

    def forward(self, x):
        mv_t_list = self.forward_probabilistic(x)
        # Sample one point from each distribution
        samples = np.array([mv_t_list[i].rvs() for i in range(len(mv_t_list))])
        return self.clip_to_box(torch.tensor(samples, dtype=x.dtype, device=x.device))

    def sample(self, n):
        joint_t = self.get_joint()
        samples = joint_t.rvs(size=n)
        return torch.tensor(samples, dtype=torch.float32)

    def wasserstein_projection_distance(self, X_orig = None):
        if X_orig is None :
            # Sample a large number of points from the joint distribution
            n_samples = 1000
            samples = self.sample(n_samples).numpy()
            X = samples[:, :self.mean_offset.shape[0]]
            Y = samples[:, self.mean_offset.shape[0]:]
            diff = X - Y
            wasserstein = np.mean(np.linalg.norm(diff, axis=-1, ord=2))
            return wasserstein
        else :
            X_prime = self.forward(X_orig).detach().cpu().numpy()
            diff = X_orig.detach().cpu().numpy() - X_prime
            wasserstein = np.mean(np.linalg.norm(diff, axis=-1, ord=2))
            return wasserstein

    def lipschitz_proxy(self, X_orig = None):
        if X_orig is None:
            # Compute the bi-Lipschitz metric by sampling from the conditional distributions
            samples = self.sample()
            X = samples[:, :self.mean_offset.shape[0]]
            Y = samples[:, self.mean_offset.shape[0]:]
            return bi_lipschitz_metric(X,Y)
        else :
            X_prime = self.forward(X_orig).detach().cpu().numpy()
            return bi_lipschitz_metric(X_orig.detach().cpu(), torch.tensor(X_prime, dtype=X_orig.dtype, device=X_orig.device))




