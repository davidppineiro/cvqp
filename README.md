# CVQP

[![PyPI version](https://img.shields.io/pypi/v/cvqp.svg)](https://pypi.org/project/cvqp/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

A Python implementation of an operator splitting method for solving large-scale CVaR-constrained quadratic programs, as described in our [paper](https://web.stanford.edu/~boyd/papers/cvar_qp.html).

## Installation

### From PyPI (recommended)

```bash
pip install cvqp
```

### From Source

Clone the repository and install using Poetry:

```bash
# Clone the repository
git clone https://github.com/cvxgrp/cvqp.git
cd cvqp

# Install dependencies
poetry install

# Compile C++ extensions
poetry run pip install -e .
```

## CVaR-Constrained Quadratic Programs

CVaR-constrained quadratic programs (CVQPs) are optimization problems of the form

$$
\begin{align}
\text{minimize} & \quad \frac{1}{2}x^TPx + q^Tx \\
\text{subject to} & \quad \phi_\beta(Ax) \leq \kappa \\
                  & \quad l \leq Bx \leq u
\end{align}
$$

where $x \in \mathbb{R}^n$ is the variable, $P \in \mathbb{R}^{n \times n}$ is positive semidefinite, $q \in \mathbb{R}^n$, $\phi_\beta$ is the conditional value-at-risk (CVaR) with parameter $\beta \in (0,1)$, $A \in \mathbb{R}^{m \times n}$, $B \in \mathbb{R}^{p \times n}$, and $l, u \in \mathbb{R}^p$ are lower and upper bounds (which can include infinity).

If $Ax$ represents a vector of scenario losses, the CVaR constraint $\phi_\beta(Ax) \leq \kappa$ bounds the average loss in the worst $(1-\beta)$ fraction of scenarios. This type of constraint is useful in risk-sensitive applications such as portfolio optimization, where we want to limit the expected loss in the worst-case scenarios.

### Example: Using CVQP Solver

```python
import numpy as np
from cvqp import CVQP, CVQPParams, CVQPConfig

# Define problem parameters
params = CVQPParams(
    P=np.eye(10),              # Quadratic cost matrix
    q=np.ones(10) * -0.1,      # Linear cost vector
    A=np.random.randn(100, 10) * 0.2 + 0.1,  # CVaR constraint matrix
    B=np.eye(10),              # Box constraint matrix
    l=-np.ones(10),            # Lower bounds
    u=np.ones(10),             # Upper bounds
    beta=0.9,                  # CVaR confidence level
    kappa=0.1,                 # CVaR limit
)

# Create solver with custom configuration
config = CVQPConfig(
    verbose=True,              # Print detailed progress
    max_iter=1000,             # Maximum iterations
    abstol=1e-4,               # Absolute tolerance
    reltol=1e-3,               # Relative tolerance
)

# Initialize and solve
cvqp = CVQP(params, config)
results = cvqp.solve()

# Access solution
print(f"Optimal value: {results.objval[-1]:.6f}")
print(f"Solver status: {results.problem_status}")
print(f"Solve time: {results.solve_time:.2f} seconds")
print(f"Iterations: {results.iter_count}")
```

### The Same Problem in CVXPY

For comparison, here's how to solve the same problem using CVXPY:

```python
import cvxpy as cp
import numpy as np

# Define parameters (same as above)
n = 10
m = 100
P = np.eye(n)
q = np.ones(n) * -0.1
A = np.random.randn(m, n) * 0.2 + 0.1
beta = 0.9
kappa = 0.1

# Define variables
x = cp.Variable(n)

# Define objective
objective = cp.Minimize(0.5 * cp.quad_form(x, P) + q.T @ x)

# Define constraints
cvar_constraint = cp.cvar(A @ x, beta) <= kappa
box_constraints = [-np.ones(n) <= x, x <= np.ones(n)]

# Define and solve problem
prob = cp.Problem(objective, [cvar_constraint] + box_constraints)
prob.solve(solver=cp.MOSEK)

print(f"Optimal value: {prob.value:.6f}")
print(f"Solver status: {prob.status}")
```

While CVXPY makes it easy to model this problem, our specialized CVQP solver is orders of magnitude faster than general-purpose solvers (like MOSEK or CLARABEL) for large-scale problems with many scenarios.

## Sum-k-Largest Projection

The CVQP package also provides an efficient function for projecting onto the set where the sum of k largest elements is at most d:

$$
\begin{align}
\text{minimize} & \quad \lVert v - z \rVert_2^2 \\
\text{subject to} & \quad f_k(z) \leq d
\end{align}
$$

where $v \in \mathbb{R}^m$ is the vector to be projected, $z \in \mathbb{R}^m$ is the optimization variable, $f_k(z) = \sum_{i=1}^k z_{[i]}$ is the sum of the $k$ largest elements of $z$, and $z_{[1]} \geq z_{[2]} \geq \cdots \geq z_{[m]}$ are the components of $z$ in non-increasing order.

This projection problem is closely related to the CVaR constraint, as we can express the CVaR constraint equivalently using the sum-k-largest function with $k = (1-\beta)m$ and $d = \kappa k$.

### Example: Using Sum-k-Largest Projection

```python
from cvqp import proj_sum_largest
import numpy as np

# Create a vector to project
v = np.array([6.0, 2.0, 5.0, 4.0, 1.0])
k = 2  # Number of largest elements to constrain
d = 7.0  # Upper bound on sum

# Apply projection
z = proj_sum_largest(v, k, d)

print(f"Original vector: {v}")
print(f"Projected vector: {z}")
print(f"Sum of {k} largest elements after projection: {sum(sorted(z, reverse=True)[:k]):.6f}")
```

### The Same Problem in CVXPY

For comparison, here's the same projection using CVXPY:

```python
import cvxpy as cp
import numpy as np

# Create a vector to project
v = np.array([6.0, 2.0, 5.0, 4.0, 1.0])
k = 2  # Number of largest elements to constrain
d = 7.0  # Upper bound on sum

# Define variable and problem
z = cp.Variable(len(v))
objective = cp.Minimize(cp.sum_squares(z - v))
constraint = cp.sum_largest(z, k) <= d

# Solve the problem
prob = cp.Problem(objective, [constraint])
prob.solve(solver=cp.MOSEK)

print(f"Original vector: {v}")
print(f"Projected vector: {z.value}")
print(f"Sum of {k} largest elements after projection: {sum(sorted(z.value, reverse=True)[:k]):.6f}")
```

Both approaches solve the same mathematical problem, but our specialized projection algorithm is much faster and scales to vectors with millions of elements, where general-purpose solvers become impractical.

## Examples and Benchmarks

The [`examples`](examples/) directory contains notebooks and benchmark scripts for:
- Portfolio optimization
- Quantile regression

See the [examples README](examples/README.md) for detailed benchmark results from our paper and instructions for reproducing them.

## Citation

If you use CVQP in your research, please consider citing our paper:

```bibtex
@misc{cvqp2025,
  title={An Operator Splitting Method for Large-Scale {CVaR}-Constrained Quadratic Programs},
  author={Luxenberg, Eric and P\'erez-Pi\~neiro, David and Diamond, Steven and Boyd, Stephen},
  year={2025},
  eprint={2504.10814},
  archivePrefix={arXiv},
  primaryClass={math.OC},
  url={https://arxiv.org/abs/2504.10814}
}
```