# ============================
# Step 4: Transformation Models
# ============================
import numpy as np
import sklearn
import torch
from torch import nn
import cvxpy as cp
import pyomo.environ as pyo

from group_cfx.transforms.utils import distortion_metric, init_solving


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
        return distortion_metric(X_orig, self.forward(X_orig))

    def wasserstein_projection_distance(self, X_orig) -> float :
        return torch.mean(torch.norm(self.forward(X_orig) - X_orig, dim=1, p=2)).item()

    def cvxpy_solving(self, x : np.ndarray, model : sklearn.linear_model.LogisticRegression, y_prime, y_prime_confidence,
                      K =1.1, solver = cp.MOSEK) -> float:
        raise NotImplementedError("CVXPY solving not implemented for this transform.")

    def pyomo_solving(self, x, model : sklearn.linear_model.LogisticRegression, y_prime, y_prime_confidence, K =1.1,
                        bilipschitz = False) -> float:
            raise NotImplementedError("Pyomo solving not implemented for this transform.")

    def is_cvx(self):
        return False


class FullAffine(BaseTransform):
    def __init__(self, d, blp_proxy = True):
        super().__init__()
        self.A = nn.Parameter(torch.Tensor(d, d))
        self.B = nn.Parameter(torch.zeros(d))
        self.xl = [-1.5]*(d*d) + [-8.0]*d
        self.xu = [1.5]*(d*d) + [8.0]*d
        self.blp_proxy = blp_proxy

    def forward(self, x):
        return self.clip_to_box(x @ (self.A.T) + self.B)


    def lipschitz_proxy(self, X_orig):
        if self.blp_proxy :
            # penalize deviation of A from identity
            I = torch.eye(self.A.shape[0])
            A_non_param = self.A.detach()
            return ((A_non_param - I) ** 2).mean()
        else :
            # Min singular value (and 1/singular value) of A
            sym_A = (self.A.T @ self.A)
            with torch.no_grad():
                s, _ = torch.linalg.eigh(sym_A)
                s = torch.sqrt(torch.clamp(s,0.0))
                lipschitz_upper = s.max().item()
                lipschitz_lower = s.min().item()
                lipschitz = np.min([1/lipschitz_upper, lipschitz_lower])
            return 1-lipschitz

    def pyomo_solving(self, x: np.ndarray, model, y_prime, y_prime_confidence, K=1.1, solver='ipopt'):
        n, d, w_model, b_model, margin_logit = init_solving(x, model, y_prime, y_prime_confidence)
        # Pyomo model
        m = pyo.ConcreteModel()

        # Convert x to array if not already
        x = np.array(x)

        # Index sets
        m.I = pyo.RangeSet(0, n - 1)
        m.J = pyo.RangeSet(0, d - 1)
        m.K = pyo.RangeSet(0, d - 1)

        # Variables
        m.A = pyo.Var(m.J, m.K, domain=pyo.Reals)
        m.b = pyo.Var(m.J, domain=pyo.Reals)

        # Objective: sum of squared distances
        def obj_rule(m):
            return sum(
                sum((sum(m.A[j, k] * x[i, k] for k in range(d)) + m.b[j] - x[i, j]) ** 2 for j in range(d))
                for i in range(n)
            )

        m.obj = pyo.Objective(rule=obj_rule, sense=pyo.minimize)

        # Classification constraints
        def class_rule(m, i):
            logit = sum(w_model[j] * (sum(m.A[j, k] * x[i, k] for k in range(d)) + m.b[j]) for j in range(d)) + b_model
            if y_prime == 1:
                return logit >= margin_logit
            else:
                return logit <= margin_logit

        m.class_constr = pyo.Constraint(m.I, rule=class_rule)

        # Bi-Lipschitz pairwise constraints (nonconvex)
        def lip_lower(m, i, j):
            if i >= j:
                return pyo.Constraint.Skip
            diffX = x[i] - x[j]
            distX_sq = float(np.dot(diffX, diffX))
            distZ_sq = sum(
                (sum(m.A[r, k] * diffX[k] for k in range(d))) ** 2
                for r in range(d)
            )
            return distZ_sq >= (1 / (K ** 2)) * distX_sq

        def lip_upper(m, i, j):
            if i >= j:
                return pyo.Constraint.Skip
            diffX = x[i] - x[j]
            distX_sq = float(np.dot(diffX, diffX))
            distZ_sq = sum(
                (sum(m.A[r, k] * diffX[k] for k in range(d))) ** 2
                for r in range(d)
            )
            return distZ_sq <= (K ** 2) * distX_sq

        m.lip_low = pyo.Constraint(m.I, m.I, rule=lip_lower)
        m.lip_up = pyo.Constraint(m.I, m.I, rule=lip_upper)

        '''# Constraint the variable range as well
        def var_bounds_A_lower(m, j, k):
            return m.A[j,k] >= self.xl[0]
        def var_bounds_A_upper(m, j, k):
            return m.A[j,k] <= self.xu[0]
        def var_bounds_b_lower(m, j):
            return m.b[j] >= self.xl[-1]
        def var_bounds_b_upper(m, j):
            return m.b[j] <= self.xu[-1]

        m.var_bound_A_low = pyo.Constraint(m.J, m.K, rule=var_bounds_A_lower)
        m.var_bound_A_up = pyo.Constraint(m.J, m.K, rule=var_bounds_A_upper)
        m.var_bound_b_low = pyo.Constraint(m.J, rule=var_bounds_b_lower)
        m.var_bound_b_up = pyo.Constraint(m.J, rule=var_bounds_b_upper)'''

        # Solve
        solver_instance = pyo.SolverFactory(solver)
        if solver == "gurobi":
            solver_instance.options['NonConvex'] = 2
            solver_instance.options['TimeLimit'] = 4500
            solver_instance.options['NoRelHeurTime'] = 900
        try:
            res = solver_instance.solve(m)
        except ValueError as e:
            return None

        if res.solver.termination_condition != pyo.TerminationCondition.optimal:
            print("Warning: Pyomo did not converge to optimal solution")

        # Extract values
        A_val = np.array([[pyo.value(m.A[j, k]) for k in range(d)] for j in range(d)])
        b_val = np.array([pyo.value(m.b[j]) for j in range(d)])

        # Update parameters
        with torch.no_grad():
            self.A.copy_(torch.tensor(A_val, dtype=torch.float32))
            self.B.copy_(torch.tensor(b_val, dtype=torch.float32))

        return m


class PSDAffine(BaseTransform):
    def __init__(self, d, blp_proxy = True):
        super().__init__()
        self.A_cholesky_flatten = nn.Parameter(torch.zeros(d*(d+1)//2))
        self.A = None
        self.B = nn.Parameter(torch.zeros(d))
        self.xl = [-1.5]*(d*(d+1)//2) + [-8.0]*d
        self.xu = [1.5]*(d*(d+1)//2) + [8.0]*d
        self.blp_proxy = blp_proxy

    def forward(self, x):
        return self.clip_to_box(x @ (self.A.T) + self.B)

    def load_parameters(self, flatten_new_params):
        super().load_parameters(flatten_new_params)
        # Reconstruct A from lower-triangular params
        d = self.B.shape[0]
        self.A = torch.zeros(d, d, device=self.B.device)
        # Fill lower triangle directly
        idx = torch.tril_indices(d, d)
        self.A[idx[0], idx[1]] = self.A_cholesky_flatten
        # Make symmetric
        self.A = self.A + self.A.T - torch.diag(torch.diag(self.A))


    def lipschitz_proxy(self, X_orig):
        if self.blp_proxy :
            # penalize deviation of A from identity
            I = torch.eye(self.A.shape[0])
            A_non_param = self.A.detach()
            return ((A_non_param - I) ** 2).mean()
        else :
            # Min singular value (and 1/singular value) of A
            sym_A = (self.A.T @ self.A)
            with torch.no_grad():
                s, _ = torch.linalg.eigh(sym_A)
                s = torch.sqrt(torch.clamp(s,0.0))
                lipschitz_upper = s.max().item()
                lipschitz_lower = s.min().item()
                lipschitz = np.min([1/lipschitz_upper, lipschitz_lower])
            return 1-lipschitz

    def is_cvx(self):
        return True

    def cvxpy_solving(self, x : np.ndarray, model : sklearn.linear_model.LogisticRegression, y_prime, y_prime_confidence,
                      K =1.1, solver = cp.MOSEK) -> cp.Problem:
        n, d, w_model, b_model, margin_logit = init_solving(x, model, y_prime, y_prime_confidence)

        # Decision variables
        A = cp.Variable((d, d), PSD=True)
        b = cp.Variable(d)

        # Compute fX
        fX = x @ A + b  # shape (n,d)

        # Objective: minimize the average distance between transformed points
        objective = cp.Minimize(cp.sum_squares(fX -x))
        #objective = cp.Minimize(cp.sum_squares(cp.hstack([A @ x[i] + b - x[i] for i in range(n)])) / n)

        # Constraints: transformed points must be classified as y_prime with confidence
        constraints = []
        # logistic linear constraints (elementwise)
        logits = fX @ w_model.T + b_model
        if y_prime == 1:
            constraints.append(logits >= margin_logit)
        else:
            constraints.append(logits <= margin_logit)

        # Real Lipschitz constraint: Check Eigenvalues of A
        constraints.append(A >> (1 / K) * np.eye(d))
        constraints.append(A << K * np.eye(d))

        '''# Constraint the variable range as well
        constraints.append(A >= self.xl[0])
        constraints.append(A <= self.xu[0])
        constraints.append(b >= self.xl[-1])
        constraints.append(b <= self.xu[-1])'''

        # Solve the QP
        prob = cp.Problem(objective, constraints)
        prob.solve(solver=solver)
        if prob.status != cp.OPTIMAL:
            print("Warning: QP did not converge to optimal solution")
        # Update parameters
        with torch.no_grad():
            self.A = torch.tensor(A.value, dtype=torch.float32)
            self.A_cholesky_flatten.copy_(self.A[torch.tril_indices(d, d)[0], torch.tril_indices(d, d)[1]])
            self.B.copy_(torch.tensor(b.value, dtype=torch.float32))
        return prob


class DiagonalAffine(BaseTransform):
    def __init__(self, d):
        super().__init__()
        self.A_diag = nn.Parameter(torch.zeros(d))
        self.B = nn.Parameter(torch.zeros(d))
        self.xl = [0.01]*d + [-8.0]*d
        self.xu = [1.5]*d + [8.0]*d

    def forward(self, x):
        return self.clip_to_box(x @ torch.diag(self.A_diag) + self.B)

    def lipschitz_proxy(self, X_orig):
        lipschitz_upper = self.A_diag.max().item()
        lipschitz_lower = self.A_diag.min().item()
        lipschitz = np.min([1/lipschitz_upper, lipschitz_lower])
        return 1-lipschitz

    def is_cvx(self):
        return True

    def cvxpy_solving(self, x : np.ndarray, model : sklearn.linear_model.LogisticRegression, y_prime, y_prime_confidence,
                      K =1.1, solver = cp.MOSEK) -> cp.Problem:
        n, d, w_model, b_model, margin_logit = init_solving(x, model, y_prime, y_prime_confidence)

        # --- Variables ------------------------------------------------------------
        a = cp.Variable(d)  # diagonal entries of A
        b = cp.Variable(d)  # bias

        # Apply affine transform: f(x) = diag(a) * x + b
        fX = cp.multiply(x, a) + b  # shape (n,d)
        #diffs = [cp.sum_squares(cp.multiply(x[i], a) + b - x[i]) for i in range(n)]  # list of length n, each element shape (d,)

        # --- Empirical squared W2 objective --------------------------------------
        # Here we pair points in order (x_i -> y_i). For real W2, you'd solve an assignment problem,
        # but this keeps the formulation comparable to typical linear W2-approximation setups.
        #objective = cp.Minimize(sum(diffs) / n)
        objective = cp.Minimize(cp.sum_squares(fX - x) / n)

        # --- Lipschitz constraint -------------------------------------------------
        # For diagonal A, ||A||_2 = max |a_i|
        constraints = [a <= K, a >= 1 / K]

        # Constraints: transformed points must be classified as y_prime with confidence
        # logistic linear constraints (elementwise)
        #logits = cp.hstack([w_model @ (cp.multiply(x[i], a) + b) + b_model for i in range(n)])
        #logits = cp.hstack([w_model @ (np.A @ x[i] + b) + b_model for i in range(n)])
        #logits = (x @ cp.diag(a) + b) @ w_model + b_model
        logits = fX @ w_model.T + b_model
        if y_prime == 1:
            constraints.append(logits >= margin_logit)
        else:
            constraints.append(logits <= margin_logit)

        '''# Constraint the variable range as well
        constraints.append(a >= self.xl[0])
        constraints.append(a <= self.xu[0])
        constraints.append(b >= self.xl[-1])
        constraints.append(b <= self.xu[-1])'''

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
        self.r = r
        self.U = nn.Parameter(torch.zeros(d, r))
        self.V = nn.Parameter(torch.zeros(d, r))
        self.B = nn.Parameter(torch.zeros(d))
        self.xl = [-1.5]*(d*r) + [-1.5]*(d*r) + [-5.0]*d
        self.xu = [1.5]*(d*r) + [1.5]*(d*r) + [5.0]*d
    def forward(self, x):
        A = torch.eye(x.shape[1]) + self.U @ self.V.T
        return self.clip_to_box(x @ A.T + self.B)

    def lipschitz_proxy(self, X_orig):
        # penalize deviation of A from identity
        I = torch.eye(self.U.shape[0])
        A_non_param = (self.U @ self.V.T).detach() + I
        return ((A_non_param - I) ** 2).mean()


    def pyomo_solving(self, x: np.ndarray, model, y_prime, y_prime_confidence, K=1.1, solver='ipopt'):
        n, d, w_model, b_model, margin_logit = init_solving(x, model, y_prime, y_prime_confidence)
        # Pyomo model
        m = pyo.ConcreteModel()

        # Convert x to array if not already
        x = np.array(x)

        # Index sets
        m.I = pyo.RangeSet(0, n - 1)
        m.J = pyo.RangeSet(0, d - 1)
        m.R = pyo.RangeSet(0, self.r - 1)
        m.K = pyo.RangeSet(0, d - 1)

        # Variables
        m.U = pyo.Var(m.J, m.R, domain=pyo.Reals)
        m.V = pyo.Var(m.R, m.K, domain=pyo.Reals)
        m.b = pyo.Var(m.J, domain=pyo.Reals)

        # Helper function: A(x_vec) = U @ V @ x_vec + b
        def A(m, x_vec, j):
            """Returns j-th component of A(x_vec) = U V x_vec + b"""
            return sum(m.U[j, s] * sum(m.V[s, k] * x_vec[k] for k in range(d)) for s in range(self.r)) + m.b[j]

        # Objective: sum of squared distances
        def obj_rule(m):
            return sum(
                sum((A(m, x[i], j) - x[i, j]) ** 2 for j in range(d))
                for i in range(n)
            )

        m.obj = pyo.Objective(rule=obj_rule, sense=pyo.minimize)

        # Classification constraints
        def class_rule(m, i):
            logit = sum(w_model[j] * A(m, x[i], j) for j in range(d)) + b_model
            if y_prime == 1:
                return logit >= margin_logit
            else:
                return logit <= margin_logit

        m.class_constr = pyo.Constraint(m.I, rule=class_rule)

        # Bi-Lipschitz pairwise constraints (nonconvex)
        def lip_lower(m, i, j):
            if i >= j:
                return pyo.Constraint.Skip
            diffX = x[i] - x[j]
            distX_sq = float(np.dot(diffX, diffX))
            distZ_sq = sum(A(m, diffX, r_idx) ** 2 for r_idx in range(d))
            return distZ_sq >= (1 / (K ** 2)) * distX_sq

        def lip_upper(m, i, j):
            if i >= j:
                return pyo.Constraint.Skip
            diffX = x[i] - x[j]
            distX_sq = float(np.dot(diffX, diffX))
            distZ_sq = sum(A(m, diffX, r_idx) ** 2 for r_idx in range(d))
            return distZ_sq <= (K ** 2) * distX_sq

        m.lip_low = pyo.Constraint(m.I, m.I, rule=lip_lower)
        m.lip_up = pyo.Constraint(m.I, m.I, rule=lip_upper)

        # Constraint the variable range as well

        # Solve
        solver_instance = pyo.SolverFactory(solver)
        solver_instance.set_instance(m)
        res = solver_instance.solve(m, tee=True, options = {'NonConvex': 2})

        if res.solver.termination_condition != pyo.TerminationCondition.optimal:
            print("Warning: Pyomo did not converge to optimal solution")

        # Extract values
        U_val = np.array([[pyo.value(m.U[j, s]) for s in range(self.r)] for j in range(d)])
        V_val = np.array([[pyo.value(m.V[s, k]) for k in range(d)] for s in range(self.r)])
        b_val = np.array([pyo.value(m.b[j]) for j in range(d)])

        # Update parameters
        with torch.no_grad():
            self.U.copy_(torch.tensor(U_val, dtype=torch.float32))
            self.V.copy_(torch.tensor(V_val, dtype=torch.float32))
            self.B.copy_(torch.tensor(b_val, dtype=torch.float32))
        return m


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
    def __init__(self, X_init, xl, xu, box_clip=True, bilipschitz = True):
        super().__init__()
        # store original subgroup (not a parameter)
        self.register_buffer("X_orig", X_init.clone())
        # optimize these directly
        self.X_prime = nn.Parameter(X_init.clone())
        self.box_clip = box_clip
        self.xl = xl
        self.xu = xu
        self.bilipschitz = bilipschitz

    def forward(self, x=None):
        """
        x is ignored (we optimize stored X' directly).
        Returns the optimized subgroup points.
        """
        if self.box_clip:
            return self.clip_to_box(self.X_prime)
        else:
            return self.X_prime

    def is_cvx(self):
        # Only convex if no bilipschitz constraints
        return not self.bilipschitz

    def pyomo_solving(self, x, model : sklearn.linear_model.LogisticRegression, y_prime, y_prime_confidence, K =1.1,
                      solver = 'mosek') -> float:
        n,d, w_model, b_model, margin_logit = init_solving(x, model, y_prime, y_prime_confidence)

        #assert np.all(np.isfinite(x))
        assert np.isfinite(margin_logit)

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

        '''# Constraint the variable range as well
        def var_bounds_lower(m, i, j):
            return m.Z[i,j] >= self.xl
        def var_bounds_upper(m, i, j):
            return m.Z[i,j] <= self.xu
        model.var_bound_low = pyo.Constraint(model.N, model.D, rule=var_bounds_lower)
        model.var_bound_up = pyo.Constraint(model.N, model.D, rule=var_bounds_upper)'''

        # Solver
        solver_instance = pyo.SolverFactory(solver)
        if solver == "gurobi":
            solver_instance.options['NonConvex'] = 2
            solver_instance.options['TimeLimit'] = 1500
            solver_instance.options['NoRelHeurTime'] = 300
            solver_instance.options['MIPFocus'] = 1
        try :
            result = solver_instance.solve(model)
        except ValueError as e:
            return None

        # Extract solution
        Z_opt = np.array([[pyo.value(model.Z[i, j]) for j in range(d)] for i in range(n)])
        with torch.no_grad():
            self.X_prime.copy_(torch.tensor(Z_opt, dtype=torch.float32))
        return result

    def cvxpy_solving(self, x: np.ndarray, model: sklearn.linear_model.LogisticRegression,
                      y_prime, y_prime_confidence, K=1.1, solver=cp.MOSEK) -> cp.Problem:
        """
        Solves the Independent optimization problem using CVXPY.
        Since points are independent, this vectorizes extremely efficiently.
        """
        n, d, w_model, b_model, margin_logit = init_solving(x, model, y_prime, y_prime_confidence)

        # Decision Variables: The counterfactual points themselves
        Z = cp.Variable((n, d))

        # Objective: Minimize Sum of Squared Euclidean Distances
        # L2^2 = sum((Z - X)^2)
        objective = cp.Minimize(cp.sum_squares(Z - x))

        # Constraints: Classification
        constraints = []

        # Vectorized classification constraint: Z @ w + b
        logits = Z @ w_model.T + b_model

        if y_prime == 1:
            constraints.append(logits >= margin_logit)
        else:
            constraints.append(logits <= margin_logit)

        # No Bilipschitz constraints added here if K = None.
        # This is what makes it "Wachter" and not "Group CF".

        # If K is specified, add only Lipschitz constraints
        if K is not None:
            # Lipschitz constraints (independent points)
            # ||Z_i - Z_j||_2 <= K * ||X_i - X_j||_2
            for i in range(n):
                for j in range(i + 1, n):
                    diffX = x[i] - x[j]
                    distX = np.linalg.norm(diffX)
                    constraints.append(cp.norm(Z[i, :] - Z[j, :], 2) <= K * distX)

        # Solve
        prob = cp.Problem(objective, constraints)

        # CVXPY detects the diagonal structure and solves this very fast
        prob.solve(solver=solver, verbose=False)

        if prob.status != cp.OPTIMAL:
            print(f"Warning: DirectOptimization CVXPY status: {prob.status}")

        # Update parameters
        with torch.no_grad():
            self.X_prime.copy_(torch.tensor(Z.value, dtype=torch.float32))

        return prob