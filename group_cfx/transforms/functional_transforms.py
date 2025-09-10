# ============================
# Step 4: Transformation Models
# ============================
import numpy as np
import torch
from torch import nn

from utils import bi_lipschitz_metric


class BaseTransform(nn.Module):
    def clip_to_box(self, x):
        return x
        return x.clamp(0.0, 1.0)

    def load_parameters(transform, flatten_new_params):
        # Load parameters into transform
        with torch.no_grad():
            idx = 0
            for p in transform.parameters():
                numel = p.numel()
                new_p = torch.tensor(flatten_new_params[idx:idx + numel].reshape(p.shape))
                p.copy_(new_p)
                idx += numel
        assert idx == len(flatten_new_params), "Parameter size mismatch"

    def lipschitz_proxy(self):
        # By default, use the empirical Lipschitz computation, which can be slow
        return bi_lipschitz_metric(self.X_orig, self.forward(self.X_orig))

    def wasserstein_projection_distance(self, X_orig) -> float :
        return torch.mean(torch.norm(self.forward(X_orig) - X_orig, dim=1)).item()

class FullAffine(BaseTransform):
    def __init__(self, d):
        super().__init__()
        self.A = nn.Parameter(torch.eye(d))
        self.B = nn.Parameter(torch.zeros(d))
        self.xl = [-1.0]*(d*d) + [-5.0]*d
        self.xu = [1.0]*(d*d) + [5.0]*d
    def forward(self, x):
        return self.clip_to_box(x @ self.A.T + self.B)

    def lipschitz_proxy(self):
        # penalize deviation of A from identity
        I = torch.eye(self.A.shape[0])
        A_non_param = self.A.detach()
        return ((A_non_param - I) ** 2).mean()

class LowRankAffine(BaseTransform):
    def __init__(self, d, r=2):
        super().__init__()
        self.U = nn.Parameter(torch.zeros(d, r))
        self.V = nn.Parameter(torch.zeros(d, r))
        self.B = nn.Parameter(torch.zeros(d))
        self.xl = -5.0
        self.xu = 5.0
    def forward(self, x):
        A = torch.eye(x.shape[1]) + self.U @ self.V.T
        return self.clip_to_box(x @ A.T + self.B)

    def lipschitz_proxy(self):
        # penalize deviation of A from identity
        I = torch.eye(self.U.shape[0])
        A_non_param = (self.U @ self.V.T).detach() + I
        return ((A_non_param - I) ** 2).mean()

class SmallMLP(BaseTransform):
    def __init__(self, d, hidden=16):
        super().__init__()
        self.net = nn.Sequential(
            #utils.spectral_norm(nn.Linear(d, hidden)),
            nn.Linear(d, hidden),
            nn.ReLU(),
            #utils.spectral_norm(nn.Linear(hidden, hidden)),
            nn.Linear(hidden, hidden),

            nn.ReLU(),
            nn.Linear(hidden, d)
        )
    def forward(self, x):
        return self.clip_to_box(self.net(x))

class DirectOptimization(BaseTransform):
    """
    A 'model' that directly optimizes X' (counterfactuals for each individual).
    Compatible with your other transforms.
    """
    def __init__(self, X_init, box_clip=True):
        super().__init__()
        # store original subgroup (not a parameter)
        self.register_buffer("X_orig", X_init.clone())
        # optimize these directly
        self.X_prime = nn.Parameter(X_init.clone())
        self.box_clip = box_clip
        self.xl = np.min(X_init.detach().numpy())
        self.xu = np.max(X_init.detach().numpy())

    def forward(self, x=None):
        """
        x is ignored (we optimize stored X' directly).
        Returns the optimized subgroup points.
        """
        if self.box_clip:
            return self.X_prime.clamp(0.0, 1.0)
        else:
            return self.X_prime