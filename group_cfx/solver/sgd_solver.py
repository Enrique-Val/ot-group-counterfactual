from group_cfx.solver.abstract_solver import Solver
import torch
import torch.nn as nn
import torch.optim as optim


# -------------------------
# SGD Solver
# -------------------------
class SGDSolver(Solver):
    def __init__(self, lr=1e-2, epochs=1000, alpha=1.0, beta=100.0, gamma=1e-3):
        self.lr, self.epochs = lr, epochs
        self.alpha, self.beta, self.gamma = alpha, beta, gamma

    def solve(self, model, classifier, density_estimator, X_orig, y_target):
        ce = nn.CrossEntropyLoss()
        opt = optim.Adam(model.parameters(), lr=self.lr)

        for epoch in range(self.epochs):
            x_prime = model()
            logits = classifier(x_prime)
            y_t = torch.full((X_orig.shape[0],), y_target, dtype=torch.long)

            loss_closeness = ((x_prime - X_orig) ** 2).mean()
            loss_target = ce(logits, y_t)
            loss_reg = sum([(p**2).mean() for p in model.parameters()])

            loss = self.alpha*loss_closeness + self.beta*loss_target + self.gamma*loss_reg
            opt.zero_grad()
            loss.backward()
            opt.step()

        return model()

