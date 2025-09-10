import numpy as np
import torch
from scipy.linalg import sqrtm
from torch import nn
from scipy.stats import multivariate_normal

from group_cfx.transforms.functional_transforms import BaseTransform
from utils import bi_lipschitz_metric, build_covariance_matrix, compute_A, compute_A_commuting


class ProbabilisticTransform(BaseTransform):
    """
    A base class for probabilistic transforms.
    """
    def __init__(self):
        super().__init__()
        self.xl = None
        self.xu = None


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
        self.build_mvn()
        self.derive_affine_transform()

    def derive_affine_transform(self):
        self.A = torch.tensor(compute_A(self.prior_mvn.cov, self.posterior_mvn.cov))
        self.B = torch.tensor((self.posterior_mvn.mean - self.prior_mvn.mean))

    def forward(self, x):
        #Print type of x matrix
        return self.clip_to_box(x @ self.A.T + self.B)

    def wasserstein_projection_distance(self, X_orig):
        m0 = self.prior_mvn.mean
        m1 = self.posterior_mvn.mean

        # Covariances
        C0 = self.prior_mvn.cov
        C1 = self.posterior_mvn.cov

        # Mean term
        mean_diff = np.linalg.norm(m0 - m1) ** 2

        # Covariance term
        sqrt_C0 = sqrtm(C0)
        middle = sqrtm(sqrt_C0 @ C1 @ sqrt_C0)
        cov_term = np.trace(C0 + C1 - 2 * middle)

        return mean_diff + cov_term

    def load_parameters(self, flatten_new_params):
        # Call super method
        super().load_parameters(flatten_new_params)
        self.build_mvn()
        self.derive_affine_transform()

    def build_mvn(self):
        raise NotImplementedError("This is a Base class that serves as a template. Subclasses should implement build_mvn method.")

    def lipschitz_proxy(self):
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
        self.xl = [-5.0]*d + [0.00001]*d + [-1.0]*(d*(d-1)//2)
        self.xu = [5.0]*d + [1.5]*d + [1.0]*(d*(d-1)//2)

    def build_mvn(self):
        covariance_matrix_posterior = build_covariance_matrix(self.posterior_marginal_stds.detach().cpu().numpy(), self.posterior_corr_triang.detach().cpu().numpy())
        self.posterior_mvn = multivariate_normal(mean=self.posterior_mu.detach().cpu().numpy(), cov=covariance_matrix_posterior)


class GaussianScaleTransform(BaseGaussianTransform) :
    def __init__(self, d):
        super().__init__()
        self.posterior_mu = nn.Parameter(torch.zeros(d))
        # Scaling factor. A single parameter
        self.scaling  = nn.Parameter(torch.ones(1))
        self.xl = [-5.0] * d + [0.00001]
        self.xu = [5.0] * d + [1]

    def derive_affine_transform(self):
        self.A = torch.tensor(compute_A(self.prior_mvn.cov, self.posterior_mvn.cov))
        '''print("Posterior mvn covariance matrix:", self.posterior_mvn.cov)
        print("Prior mvn covariance matrix:", self.prior_mvn.cov)
        print("Scaling factor:", self.scaling.item())
        print("Computed A matrix:", self.A)
        print()'''
        self.B = torch.tensor((self.posterior_mvn.mean - self.prior_mvn.mean))

    def build_mvn(self):
        covariance_matrix_posterior = self.prior_mvn.cov * (self.scaling.item() ** 2)
        self.posterior_mvn = multivariate_normal(mean=self.posterior_mu.detach().cpu().numpy(), cov=covariance_matrix_posterior)


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

    def lipschitz_proxy(self):
        return 0.0


