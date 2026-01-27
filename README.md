# Optimal Transport Group Counterfactual Explanations
## Description
In this repository, we present an implementation of Optimal Transport Group Counterfactual Explanations, a method for generating counterfactual explanations for machine learning models using optimal transport theory.
This approach allows to model the actionable recourse as a function (or coupling), allowing for an easy handling of group constraint that for other methods are non-convex.

## Organization of the Code
The library (folder `group_cfx/` is structured as follows:
- `transforms/`: Contain the implementation for functional, Gaussian and probabilistic optimal transport, as defined in the paper.
- `solver/`: If mathematical optimization is not possible, this module provide implementation of numerical solvers to compute the optimal transport map.
- Right now, only NSGAII is fully implemented, but SGD will be added soon.

Other folders and files:
- `affine_experiment.py`: Script to reproduce the experiments from the paper.
- `analysis_scripts/`: Folder containing scripts to analyze the results of the experiments and produce the figures presented.
- `sensitivity_bn.py`: Script to reproduce the surrogate Bayesian network for sensitivity analysis.

## Example usage
Here is a simple example of how to use the library.
```python
import numpy as np
import torch
import cvxpy as cp
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification

# Replace with your actual import path
from group_cfx.transforms import PSDAffine

# ---------------------------------------------------------
# 1. Setup: Generate dummy data and a classifier
# ---------------------------------------------------------
d = 5   # Dimension of features
n = 50  # Number of samples
X, y = make_classification(n_samples=n, n_features=d, random_state=42)

# Train a standard linear classifier (e.g., Logistic Regression)
# Since the model is linear, we can use mathematical (convex) optimization
model = LogisticRegression()
model.fit(X, y)

# Select a subset of data points to transform
X_group = X[:10]
y_group = y[:10]

# ---------------------------------------------------------
# 2. Find the best OT counterfactual
# ---------------------------------------------------------
# We use an affine transform with PSD constraint on A
transform = PSDAffine(d=d)

target_class = 1            # The desired class label
confidence_margin = 0.5     # Margin for the decision boundary
lipschitz_bound = 1.1       # Max distortion allowed (K)

print("Solving SDP to find optimal transform parameters...")
transform.cvxpy_solving(
    x=X_group,
    model=model,
    y_prime=target_class,
    y_prime_confidence=confidence_margin,
    K=lipschitz_bound,
    solver=cp.SCS
)

# ---------------------------------------------------------
# 3. Apply the OT coutnerfactual to get the group counterfactual
# ---------------------------------------------------------
# The transform is now a standard PyTorch module
X_tensor = torch.tensor(X_group, dtype=torch.float32)

with torch.no_grad():
    X_transformed = transform(X_tensor)

print("\n--- Results ---")
print(f"Original Input:      {X_tensor[0].numpy()}")
print(f"Transformed Output:  {X_transformed[0].numpy()}")
print(f"PSD Matrix A:\n{transform.A.detach().numpy()}")
```

## Extending the Code
If new OT counterfactuals (transforms) are implemented, follow the guidelines:
- Inherit from the proper class. The base class is `group_cfx.transforms.BaseTransform`.
- Implement the mathematical optimization method `cvxpy_solving` if possible and make the function `is_cvx(self) -> bool` return `True`.
- If not, implement `pyomo_solving` and make the function `is_cvx(self) -> bool` return `False`.
- If mathematical optimization is not feasible at all and the existing numerical solvers are not suitable, implement a new solver in the `group_cfx.solver` module.