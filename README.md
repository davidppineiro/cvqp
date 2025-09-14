# CVQP

[![PyPI version](https://img.shields.io/pypi/v/cvqp.svg)](https://pypi.org/project/cvqp/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

CVQP is a Python library with fast, scalable methods for solving large-scale CVaR-constrained quadratic programs and Euclidean projection onto CVaR constraints (or equivalent sum-of-k-largest constraints). These methods handle problems where standard solvers are prohibitively slow or fail. For details, see our [paper](https://web.stanford.edu/~boyd/papers/cvar_qp.html).

## Installation

From PyPI:

```bash
pip install cvqp
```

From source:

```bash
git clone https://github.com/cvxgrp/cvqp.git
cd cvqp
poetry install
poetry run pip install -e .
```

## Usage

### Solve a CVaR-constrained quadratic program

CVQP solves problems of the form

$$
\begin{array}{ll}
\text{minimize} & \frac{1}{2}x^TPx + q^Tx \\
\text{subject to} & \phi_\beta(Ax) \leq \kappa \\
                  & l \leq Bx \leq u,
\end{array}
$$

where $\phi_\beta$ is the Conditional Value-at-Risk (CVaR) at level $\beta$.

```python
import numpy as np
from cvqp import solve_cvqp

# Problem data
np.random.seed(42)
n, m = 10, 100
P = np.eye(n) * 0.1                    # Quadratic cost matrix
q = np.ones(n) * -0.1                  # Linear cost vector
A = np.random.randn(m, n) * 0.2 + 0.1  # CVaR constraint matrix
B = np.eye(n)                          # Box constraint matrix
l = -np.ones(n)                        # Lower bounds
u = np.ones(n)                         # Upper bounds
beta = 0.9                             # CVaR confidence level
kappa = 0.1                            # CVaR limit

results = solve_cvqp(P=P, q=q, A=A, B=B, l=l, u=u,
                     beta=beta, kappa=kappa, verbose=True)

print(f"Optimal value: {results.value:.6f}")
print(f"Solution: {results.x}")
```

### Project onto a CVaR constraint

Project a vector $v$ onto the feasible set defined by a CVaR constraint:

$$
\begin{array}{ll}
\text{minimize} & \|v - x\|_2^2 \\
\text{subject to} & \phi_\beta(x) \leq \kappa.
\end{array}
$$

```python
from cvqp import proj_cvar
import numpy as np

# Create a vector representing scenario losses
v = np.array([4.0, 1.0, 6.0, 2.0, 5.0, 3.0, 7.0])
beta = 0.8  # Confidence level (80%)
kappa = 4.5  # CVaR threshold

x = proj_cvar(v, beta, kappa)

# Compute CVaR at confidence level beta
def cvar(losses, beta):
    sorted_losses = sorted(losses, reverse=True)
    k = int((1 - beta) * len(losses))
    return sum(sorted_losses[:k]) / k

print(f"Original losses: {v}")
print(f"Projected losses: {x}")
print(f"Original CVaR: {cvar(v, beta):.2f}")
print(f"Projected CVaR: {cvar(x, beta):.2f}")
```

### Project onto a sum-of-k-largest constraint

CVaR constraints are equivalent to sum-of-k-largest constraints. For a vector $x \in \mathbf{R}^m$, the CVaR constraint $\phi_\beta(x) \leq \kappa$ is equivalent to $f_k(x) \leq d$ where $f_k(x) = \sum_{i=1}^k x_{[i]}$ (sum of k largest components), $k = (1-\beta)m$, and $d = \kappa k$. The projection problem becomes

$$
\begin{array}{ll}
\text{minimize} & \|v - x\|_2^2 \\
\text{subject to} & f_k(x) \leq d.
\end{array}
$$

```python
from cvqp import proj_sum_largest
import numpy as np

# Create a vector to project
v = np.array([6.0, 2.0, 5.0, 4.0, 1.0])
k = 2  # Number of largest elements to constrain
d = 7.0  # Upper bound on sum

x = proj_sum_largest(v, k, d)

print(f"Original vector: {v}")
print(f"Projected vector: {x}")
print(f"Sum of {k} largest: {sum(sorted(x, reverse=True)[:k]):.2f}")
```

## Benchmarks

See `benchmarks/` for benchmark results against MOSEK and Clarabel on a suite of large-scale problems.

## Citation

If you use CVQP in your research, please cite:

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