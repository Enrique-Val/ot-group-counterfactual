from abc import ABC, abstractmethod

# -------------------------
# Abstract Solver
# -------------------------
class Solver(ABC):
    @abstractmethod
    def solve(self, model, classifier, density_estimator, X_orig, y_target):
        pass