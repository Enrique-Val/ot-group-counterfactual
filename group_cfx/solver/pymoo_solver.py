import torch
import numpy as np
from pymoo.optimize import minimize
from pymoo.core.problem import ElementwiseProblem

from group_cfx.solver.abstract_solver import Solver
from group_cfx.transforms.functional_transforms import BaseTransform
from group_cfx.transforms.gaussian_transforms import BaseGaussianTransform
from group_cfx.transforms.probabilistic_transforms import ProbabilisticTransform


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

    def solve(self, model : BaseTransform, classifier, X_orig, y_prime, y_prime_confidence, seed = 0):
        X_np = X_orig.detach().cpu().numpy()
        n, d = X_np.shape
        flatten_params = np.concatenate([p.detach().numpy().flatten() for p in model.parameters()])

        class SubgroupProblem(ElementwiseProblem):
            def __init__(self, min_acc=0.9, xl=-2.0, xu=2.0):
                super().__init__(n_var=len(flatten_params), n_obj=2, n_constr=X_orig.shape[0], xl=xl, xu=xu)
                self.min_acc = min_acc

            def _evaluate(self, x, out, *args, **kwargs):
                # Substitute d**2 first params for 0s
                #x[:d * d] = 0.0

                model.load_parameters(x)

                X_prime_t = model(X_orig)

                if isinstance(model, ProbabilisticTransform):
                    wasserstein = model.wasserstein_projection_distance()
                    # Lipschitz proxy
                    f2 = model.lipschitz_proxy()
                else :
                    wasserstein = model.wasserstein_projection_distance(X_orig)
                    # Lipschitz proxy
                    f2 = model.lipschitz_proxy(X_orig)

                # Constraint 1: classification accuracy
                if isinstance(classifier, torch.nn.Module):
                    with torch.no_grad():
                        logits = classifier(X_prime_t)
                        probs_y_target = torch.sigmoid(logits)
                else:
                    preds = classifier.predict_proba(X_prime_t.detach().cpu().numpy())
                    probs_y_target = preds[:, y_prime]

                g1 = self.min_acc - probs_y_target

                out["F"] = [wasserstein,f2]
                out["G"] = g1

        problem = SubgroupProblem(min_acc=self.min_acc, xl = model.xl, xu =model.xu)

        res = minimize(
            problem,
            self.algorithm,
            termination=self.termination,
            seed=seed,
            verbose=self.verbose,
            **self.kwargs
        )

        X_prime = model(X_orig).detach().cpu()

        return X_prime, res.F, res.X