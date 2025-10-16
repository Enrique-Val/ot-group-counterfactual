import numpy as np
import torch
from scipy.special import digamma
from torch import nn
from scipy.stats import multivariate_normal, multivariate_t

from group_cfx.transforms.functional_transforms import BaseTransform
from group_cfx.transforms.utils import wasserstein_distance_normals, compute_A, build_covariance_matrix, \
    bi_lipschitz_metric


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
        raise NotImplementedError()


class GMMForwardTransform(ProbabilisticTransform) :
    """
    A Gaussian Mixture Model (GMM) based transform.
    The GMM parameters (means, covariances, and weights) are learnable.
    """
    def __init__(self, d, n_components=3):
        super().__init__()
        self.n_components = n_components
        self.prior_gmm : list[multivariate_normal] = []  # Placeholder for prior distribution
        self.posterior_gmm : list[multivariate_normal] = []
        self.log_weights = []
        self.A = []
        self.B = []
        self.means = nn.Parameter(torch.zeros(n_components, d))
        self.marginal_stds = nn.Parameter(torch.ones(n_components, d)*0.1)
        self.corr_triang = nn.Parameter(torch.zeros(n_components, d * (d - 1) // 2)+0.1)
        self.xl = [0.2]* (n_components * d) + [0.00001]*(n_components * d) + [-0.99999]*(n_components * (d*(d-1)//2))
        self.xu = [1.0]* (n_components * d) + [1]*(n_components * d) + [0.99999]*(n_components * (d*(d-1)//2))

    def fit_prior(self, X_orig):
        # Fit a GMM to the original data using sklearn
        from sklearn.mixture import GaussianMixture
        X_np = X_orig.detach().cpu().numpy()
        gmm = GaussianMixture(n_components=self.n_components, covariance_type='full', random_state=0)
        gmm.fit(X_np)
        self.prior_gmm = []
        for i in range(self.n_components):
            mvn = multivariate_normal(mean=gmm.means_[i], cov=gmm.covariances_[i])
            self.prior_gmm.append(mvn)
            self.log_weights.append(np.log(gmm.weights_[i] + 1e-10))  # Avoid log(0)
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
            mvn = multivariate_normal(mean=self.means[i].detach().cpu().numpy(), cov=cov_matrix)
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
            A_i = torch.tensor(compute_A(self.prior_gmm[i].cov, self.posterior_gmm[i].cov))
            B_i = torch.tensor((self.posterior_gmm[i].mean - self.prior_gmm[i].mean))
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

    def wasserstein_projection_distance(self, X_orig):
        # Compute pairwise distances between all components
        W = np.zeros((self.n_components, ))
        for i in range(self.n_components):
            m0 = self.prior_gmm[i].mean
            m1 = self.posterior_gmm[i].mean

            # Covariances
            C0 = self.prior_gmm[i].cov
            C1 = self.posterior_gmm[i].cov

            W[i] = wasserstein_distance_normals(m0, C0, m1, C1)

        # Weight by mixture weights
        weights_prior = np.exp(np.array(self.log_weights))
        weights_prior /= weights_prior.sum()
        weighted_wasserstein = np.sum(W * weights_prior)
        return weighted_wasserstein



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
            nu = max(nu_new, 1e-3)

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
        self.mean = nn.Parameter(torch.zeros(d))
        self.marginal_stds = nn.Parameter(torch.ones(d)*0.1)
        self.corr_triang = nn.Parameter(torch.zeros(d * (d - 1) // 2)+0.1)
        self.cross_corr_triang = nn.Parameter(torch.zeros(d * (d + 1) // 2)+0.1)
        self.xl = [0.2]* d + [0.00001]*d + [-0.99999]*(d*(d-1)//2) + [-0.99999]*(d*(d+1)//2)
        self.xu = [1.0]* d + [1]*d + [0.99999]*(d*(d-1)//2) + [0.99999]*(d*(d+1)//2)


    def build_joint(self):
        # Build the posterior multivariate t distribution and the cross-covariance matrix
        # We assume that prior_t is already set
        cov_matrix = build_covariance_matrix(self.marginal_stds.detach().cpu().numpy(), self.corr_triang.detach().cpu().numpy())
        self.posterior_t = multivariate_t(loc=self.mean.detach().cpu().numpy(), shape=cov_matrix, df=self.prior_t.df)
        d = self.mean.shape[0]
        cross_cov_matrix = np.zeros((d, d))
        tril_idx = np.tril_indices(d)
        cross_cov_matrix[tril_idx] = self.cross_corr_triang.detach().cpu().numpy()
        cross_cov_matrix[(tril_idx[1], tril_idx[0])] = self.cross_corr_triang.detach().cpu().numpy()  # symmetry
        # scale by std
        D_prior = np.diag(np.sqrt(np.diag(self.prior_t.cov)))
        D_post = np.diag(self.marginal_stds.detach().cpu().numpy())
        self.cross_covariance_matrix = D_post @ cross_cov_matrix @ D_prior
        # For stability, symmetrize
        self.cross_covariance_matrix = (self.cross_covariance_matrix + self.cross_covariance_matrix.T) / 2


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
        d = self.mean.shape[0]
        n = x_np.shape[0]
        mean_cond = np.zeros((n, d))
        cov_cond = np.zeros((n, d, d))
        dof_cond = np.zeros((n, d))
        for i in range(n):
            diff = (x_np[i] - self.prior_t.mean).reshape(-1, 1)
            mean_cond[i] = self.mean.detach().cpu().numpy() + self.cross_covariance_matrix @ np.linalg.inv(self.prior_t.cov) @ diff
            cov_cond_ns = self.posterior_t.cov - self.cross_covariance_matrix @ np.linalg.inv(self.prior_t.cov) @ self.cross_covariance_matrix.T
            dof_incr = diff.T @ np.linalg.inv(self.prior_t.cov) @ diff
            cov_cond[i] = (self.prior_t.df + dof_incr) / (self.prior_t.df + d) * cov_cond_ns
            dof_cond[i] = self.prior_t.df + d
        # Build n multivariate t distributions
        mv_t_list = []
        for i in range(n):
            mv_t_list[i] = multivariate_t(loc=mean_cond[i], shape=cov_cond[i], df=dof_cond[i])
        return mv_t_list

    def get_joint(self) :
        # Build the joint distribution of (X, X')
        d = self.mean.shape[0]
        joint_mean = np.concatenate([self.prior_t.mean, self.mean.detach().cpu().numpy()])
        joint_cov = np.zeros((2*d, 2*d))
        joint_cov[:d, :d] = self.prior_t.cov
        joint_cov[d:, d:] = self.posterior_t.cov
        joint_cov[:d, d:] = self.cross_covariance_matrix
        joint_cov[d:, :d] = self.cross_covariance_matrix.T
        joint_cov = ensure_positive_definite(joint_cov, method = 'eigenvalue_clipping')
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
            X = samples[:, :self.mean.shape[0]]
            Y = samples[:, self.mean.shape[0]:]
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
            X = samples[:, :self.mean.shape[0]]
            Y = samples[:, self.mean.shape[0]:]
            return bi_lipschitz_metric(X,Y)
        else :
            X_prime = self.forward(X_orig).detach().cpu().numpy()
            return bi_lipschitz_metric(X_orig.detach().cpu(), torch.tensor(X_prime, dtype=X_orig.dtype, device=X_orig.device))




