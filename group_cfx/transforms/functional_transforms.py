# ============================
# Step 4: Transformation Models
# ============================
import numpy as np
import sklearn
import torch
from torch import nn
import cvxpy as cp
import pyomo.environ as pyo

from group_cfx.transforms.utils import bi_lipschitz_metric, init_solving


class BaseTransform(nn.Module):
    def clip_to_box(self, x):
        return x
        return x.clamp(0.0, 1.0)

    def load_parameters(self, flatten_new_params):
        # Load parameters into transform
        with torch.no_grad():
            idx = 0
            for p in self.parameters():
                numel = p.numel()
                new_p = torch.tensor(flatten_new_params[idx:idx + numel].reshape(p.shape))
                p.copy_(new_p)
                idx += numel
        assert idx == len(flatten_new_params), "Parameter size mismatch"

    def lipschitz_proxy(self,X_orig):
        # By default, use the empirical Lipschitz computation, which can be slow
        return bi_lipschitz_metric(X_orig, self.forward(X_orig))

    def wasserstein_projection_distance(self, X_orig) -> float :
        return torch.mean(torch.norm(self.forward(X_orig) - X_orig, dim=1)).item()

    def cvxpy_solving(self, x : np.ndarray, model : sklearn.linear_model.LogisticRegression, y_prime, y_prime_confidence,
                      K =1.1) -> float:
        raise NotImplementedError("CVXPY solving not implemented for this transform.")

    def pyomo_solving(self, x, model : sklearn.linear_model.LogisticRegression, y_prime, y_prime_confidence, K =1.1,
                        bilipschitz = False) -> float:
            raise NotImplementedError("Pyomo solving not implemented for this transform.")

class FullAffine(BaseTransform):
    def __init__(self, d):
        super().__init__()
        self.A_lower_triang = nn.Parameter(torch.zeros(d*(d+1)//2))
        self.A = None
        self.B = nn.Parameter(torch.zeros(d))
        self.xl = [-1.0]*(d*(d+1)//2) + [-5.0]*d
        self.xu = [1.0]*(d*(d+1)//2) + [5.0]*d
    def forward(self, x):
        return self.clip_to_box(x @ (self.A.T) + self.B)

    def load_parameters(self, flatten_new_params):
        super().load_parameters(flatten_new_params)
        # Reconstruct A from lower-triangular params
        d = self.B.shape[0]
        self.A = torch.zeros(d, d, device=self.B.device)
        # Fill lower triangle directly
        idx = torch.tril_indices(d, d)
        self.A[idx[0], idx[1]] = self.A_lower_triang
        # Make symmetric
        self.A = self.A + self.A.T - torch.diag(torch.diag(self.A))


    def lipschitz_proxy(self, X_orig):
        # penalize deviation of A from identity
        I = torch.eye(self.A.shape[0])
        A_non_param = self.A.detach()
        return ((A_non_param - I) ** 2).mean()
        # Min singular value (and 1/singular value) of A
        sym_A = (self.A.T @ self.A)
        with torch.no_grad():
            s, _ = torch.linalg.eigh(sym_A)
            s = torch.sqrt(torch.clamp(s,0.0))
            lipschitz_upper = s.max().item()
            lipschitz_lower = s.min().item()
            lipschitz = np.min([1/lipschitz_upper, lipschitz_lower])
        return 1-lipschitz

    def cvxpy_solving(self, x : np.ndarray, model : sklearn.linear_model.LogisticRegression, y_prime, y_prime_confidence,
                      K =1.1, solver = cp.MOSEK) -> cp.Problem:
        n, d, w_model, b_model, margin_logit = init_solving(x, model, y_prime, y_prime_confidence)

        # Decision variables
        A = cp.Variable((d, d), symmetric=True)
        b = cp.Variable(d)


        # Objective: minimize the average distance between transformed points
        # objective = cp.Minimize(cp.sum_squares(x @ A + b -x))
        objective = cp.Minimize(cp.sum([cp.sum_squares(A @ x[i] + b - x[i]) for i in range(n)]))

        # Constraints: transformed points must be classified as y_prime with confidence
        constraints = []
        # logistic linear constraints (elementwise)
        logits = cp.hstack([w_model @ (A @ x[i] + b) + b_model for i in range(n)])
        if y_prime == 1:
            constraints.append(logits >= margin_logit)
        else:
            constraints.append(logits <= margin_logit)

        # Real Lipschitz constraint: Check Eigenvalues of A
        constraints.append(A >> (1 / K) * np.eye(d))
        constraints.append(A << K * np.eye(d))

        # Solve the QP
        prob = cp.Problem(objective, constraints)
        prob.solve(solver=solver)
        if prob.status != cp.OPTIMAL:
            print("Warning: QP did not converge to optimal solution")
        # Update parameters
        with torch.no_grad():
            self.A = torch.tensor(A.value, dtype=torch.float32)
            self.A_lower_triang.copy_(self.A[torch.tril_indices(d, d)[0], torch.tril_indices(d, d)[1]])
            self.B.copy_(torch.tensor(b.value, dtype=torch.float32))
        return prob


class DiagonalAffine(BaseTransform):
    def __init__(self, d):
        super().__init__()
        self.A_diag = nn.Parameter(torch.zeros(d))
        self.B = nn.Parameter(torch.zeros(d))
        self.xl = [0.01]*d + [-5.0]*d
        self.xu = [1.5]*d + [5.0]*d

    def forward(self, x):
        return self.clip_to_box(x @ torch.diag(self.A_diag) + self.B)

    def lipschitz_proxy(self, X_orig):
        lipschitz_upper = self.A_diag.max().item()
        lipschitz_lower = self.A_diag.min().item()
        lipschitz = np.min([1/lipschitz_upper, lipschitz_lower])
        return 1-lipschitz

    def cvxpy_solving(self, x : np.ndarray, model : sklearn.linear_model.LogisticRegression, y_prime, y_prime_confidence,
                      K =1.1, solver = cp.MOSEK) -> cp.Problem:
        n, d, w_model, b_model, margin_logit = init_solving(x, model, y_prime, y_prime_confidence)

        # --- Variables ------------------------------------------------------------
        a = cp.Variable(d)  # diagonal entries of A
        b = cp.Variable(d)  # bias

        # Apply affine transform: f(x) = diag(a) * x + b
        fX = cp.multiply(x, a) + b  # shape (n,d)

        # --- Empirical squared W2 objective --------------------------------------
        # Here we pair points in order (x_i -> y_i). For real W2, you'd solve an assignment problem,
        # but this keeps the formulation comparable to typical linear W2-approximation setups.
        objective = cp.Minimize(cp.sum_squares(fX - x) / n)

        # --- Lipschitz constraint -------------------------------------------------
        # For diagonal A, ||A||_2 = max |a_i|
        constraints = [a <= K, a >= 1 / K]

        # Constraints: transformed points must be classified as y_prime with confidence
        # logistic linear constraints (elementwise)
        logits = cp.hstack([w_model @ (cp.multiply(x[i], a) + b) + b_model for i in range(n)])
        if y_prime == 1:
            constraints.append(logits >= margin_logit)
        else:
            constraints.append(logits <= margin_logit)

        # Solve the QP
        prob = cp.Problem(objective, constraints)
        prob.solve(solver=solver)
        if prob.status != cp.OPTIMAL:
            print("Warning: QP did not converge to optimal solution")
        # Update parameters
        with torch.no_grad():
            self.A_diag.copy_(torch.tensor(a.value, dtype=torch.float32))
            self.B.copy_(torch.tensor(b.value, dtype=torch.float32))
        return prob


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

    def lipschitz_proxy(self, X_orig):
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
    def __init__(self, X_init, xl, xu, box_clip=True):
        super().__init__()
        # store original subgroup (not a parameter)
        self.register_buffer("X_orig", X_init.clone())
        # optimize these directly
        self.X_prime = nn.Parameter(X_init.clone())
        self.box_clip = box_clip
        self.xl = xl
        self.xu = xu

    def forward(self, x=None):
        """
        x is ignored (we optimize stored X' directly).
        Returns the optimized subgroup points.
        """
        if self.box_clip:
            return self.clip_to_box(self.X_prime)
        else:
            return self.X_prime

    def cvxpy_solving(self, x: np.ndarray, model: sklearn.linear_model.LogisticRegression,
                      y_prime, y_prime_confidence, K = 1.1, bilipschitz = False) -> cp.Problem:
        n, d, w_model, b_model, margin_logit = init_solving(x, model, y_prime, y_prime_confidence)

        if bilipschitz :
            raise ValueError("CVXPY bilipschitz cannot implemented (non-convex). Solve using metaheuristics"
                             "or use Pyomo with non-convex solver (e.g. Ipopt).")

        # Decision variables: transformed points directly
        X_prime = cp.Variable((n, d))

        # Objective: keep transformed points close to originals
        objective = cp.Minimize(cp.sum([cp.sum_squares(X_prime[i, :] - x[i, :]) for i in range(n)]))

        # Constraints: classification as y_prime with confidence
        logits = cp.hstack([w_model @ X_prime[i, :] + b_model for i in range(n)])
        constraints = []
        if y_prime == 1:
            constraints.append(logits >= margin_logit)
        else:
            constraints.append(logits <= margin_logit)

        # Lipschitz-like constraints
        for i in range(n):
            for j in range(i + 1, n):
                v = x[i] - x[j]
                # enforce pairwise distance contraction/expansion bound
                constraints.append(cp.norm(X_prime[i, :] - X_prime[j, :], 2) <= K**2 * np.linalg.norm(v, 2))

        # Solve
        prob = cp.Problem(objective, constraints)
        prob.solve(solver=cp.SCS)
        if prob.status != cp.OPTIMAL:
            print("Warning: QP did not converge")

        # Update stored transformed points
        with torch.no_grad():
            self.X_prime.copy_(torch.tensor(X_prime.value, dtype=torch.float32))
        return prob

    def pyomo_solving(self, x, model : sklearn.linear_model.LogisticRegression, y_prime, y_prime_confidence, K =1.1,
                      bilipschitz = False, solver = 'mosek') -> float:
        n,d, w_model, b_model, margin_logit = init_solving(x, model, y_prime, y_prime_confidence)

        model = pyo.ConcreteModel()

        # Index sets
        model.N = pyo.RangeSet(0, n - 1)
        model.D = pyo.RangeSet(0, d - 1)

        # Decision variables
        model.Z = pyo.Var(model.N, model.D, domain=pyo.Reals)

        # Objective: sum of squared deviations
        def obj_rule(m):
            return np.sum((m.Z[i, j] - x[i, j].item()) ** 2 for i in m.N for j in m.D)

        model.obj = pyo.Objective(rule=obj_rule, sense=pyo.minimize)

        # Classification constraints
        def cls_rule(m, i):
            logit = np.sum(w_model[j] * m.Z[i, j] for j in range(d)) + b_model
            if y_prime == 1:
                return logit >= margin_logit
            else:
                return logit <= margin_logit

        model.cls_con = pyo.Constraint(model.N, rule=cls_rule)

        if bilipschitz :
            print("K", K**2)
            # Pairwise bi-Lipschitz (non-convex)
            def pairwise_rule_lower(m, i, j):
                if i >= j:
                    return pyo.Constraint.Skip
                # Euclidean distance squared
                dist_Z = sum((m.Z[i, k] - m.Z[j, k]) ** 2 for k in range(d))
                dist_X = np.linalg.norm(x[i] - x[j]) ** 2
                # enforce lower bound: dist_Z >= (1/K^2) * dist_X
                # Since the distance is squared, we use K^2 here
                return dist_Z >= (1.0 / K ** 2) * dist_X # and dist_Z <= (K ** 2) * dist_X

            model.lip_con_low = pyo.Constraint(model.N, model.N, rule=pairwise_rule_lower)

        def pairwise_rule_upper(m, i, j):
            if i >= j:
                return pyo.Constraint.Skip
            # Euclidean distance squared
            dist_Z = sum((m.Z[i, k] - m.Z[j, k]) ** 2 for k in range(d))
            dist_X = np.linalg.norm(x[i] - x[j]) ** 2
            # enforce lower bound: dist_Z >= (1/K^2) * dist_X
            # Since the distance is squared, we use K^2 here
            return dist_Z <= (K ** 2) * dist_X # and dist_Z <= (K ** 2) * dist_X

        model.lip_con_up = pyo.Constraint(model.N, model.N, rule=pairwise_rule_upper)


        # Solver (Ipopt)
        solver = pyo.SolverFactory(solver)
        result = solver.solve(model, tee=True)

        # Extract solution
        Z_opt = np.array([[pyo.value(model.Z[i, j]) for j in range(d)] for i in range(n)])
        with torch.no_grad():
            self.X_prime.copy_(torch.tensor(Z_opt, dtype=torch.float32))
        return result