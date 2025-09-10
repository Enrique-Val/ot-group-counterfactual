import torch
import numpy as np
from pymoo.optimize import minimize
from pymoo.core.problem import ElementwiseProblem

from group_cfx.solver.abstract_solver import Solver


# -------------------
# Generic PyMOO Solver
# -------------------------
class PyMooSolver(Solver):
    def __init__(self, algorithm, termination, min_acc=0.9, verbose=False, **kwargs):
        """
        algorithm: pymoo algorithm object (e.g. NSGA2(...), CMAES(...))
        termination: pymoo termination object (e.g. get_termination("n_gen", 100))
        kwargs: extra args forwarded to pymoo.minimize
        """
        self.algorithm = algorithm
        self.termination = termination
        self.min_acc = min_acc
        self.verbose = verbose
        self.kwargs = kwargs

    def solve(self, model, classifier, density_estimator, X_orig, y_target):
        X_np = X_orig.detach().cpu().numpy()
        n, d = X_np.shape
        flatten_params = np.concatenate([p.detach().numpy().flatten() for p in model.parameters()])

        class SubgroupProblem(ElementwiseProblem):
            def __init__(self, min_acc=0.9, xl=-2.0, xu=2.0):
                super().__init__(n_var=len(flatten_params), n_obj=2, n_constr=2, xl=xl, xu=xu)
                self.min_acc = min_acc

            def _evaluate(self, x, out, *args, **kwargs):
                # Substitute d**2 first params for 0s
                #x[:d * d] = 0.0

                model.load_parameters(x)

                X_prime_t = model(X_orig)

                # Wasserstein proxy: mean pairwise L2
                diff = X_prime_t.detach().cpu().numpy() - X_np
                wasserstein = np.mean(np.linalg.norm(diff, axis=-1, ord=2))

                # Lipschitz proxy
                f2 = model.lipschitz_proxy()

                # Objective 3: -density
                #log_probs = density_estimator.score_samples(X_prime_t.detach().cpu().numpy())
                #density = np.mean(log_probs)
                density = 0


                # Constraint 1: classification accuracy
                if isinstance(classifier, torch.nn.Module):
                    with torch.no_grad():
                        logits = classifier(X_prime_t)
                        preds = logits.argmax(1).cpu().numpy()
                    acc = (preds == y_target).mean()
                else:
                    preds = classifier.predict_proba(X_prime_t.detach().cpu().numpy())
                    probs_y_target = preds[:, y_target]
                    mse = np.mean((probs_y_target - 1) ** 2)

                g1 = mse - (1-self.min_acc)
                g2 = -density-148

                out["F"] = [wasserstein,f2]
                out["G"] = [g1,g2]

        problem = SubgroupProblem(min_acc=self.min_acc, xl = model.xl, xu =model.xu)

        res = minimize(
            problem,
            self.algorithm,
            termination=self.termination,
            seed=1,
            verbose=self.verbose,
            **self.kwargs
        )

        X_prime = model(X_orig).detach().cpu()

        return X_prime, res.F, res.X