"""
Type definitions for the CVQP solver.
"""

from dataclasses import dataclass
from typing import Union, Optional
import numpy as np
import scipy as sp


def _validate_positive(value: float, name: str):
    """Check that a value is positive."""
    if value <= 0:
        raise ValueError(f"{name} must be positive")


def _validate_range(value: float, name: str, min_val: float, max_val: float, exclusive: bool = False):
    """Check that a value is within the specified range."""
    if exclusive:
        if not (min_val < value < max_val):
            raise ValueError(f"{name} must be between {min_val} and {max_val} (exclusive)")
    else:
        if not (min_val <= value <= max_val):
            raise ValueError(f"{name} must be between {min_val} and {max_val}")


def _validate_dimensions(params):
    """Check dimensional compatibility of problem matrices."""
    n = params.q.shape[0]

    if params.P is not None:
        if params.P.shape[0] != params.P.shape[1]:
            raise ValueError("Cost matrix P must be square")
        if params.P.shape[1] != n:
            raise ValueError(f"Incompatible dimensions: P({params.P.shape}) and q({params.q.shape})")

    if params.A.shape[1] != n:
        raise ValueError(f"Incompatible dimensions: A({params.A.shape}) and q({params.q.shape})")
    if params.B.shape[1] != n:
        raise ValueError(f"Incompatible dimensions: B({params.B.shape}) and q({params.q.shape})")
    if params.l.shape[0] != params.B.shape[0] or params.u.shape[0] != params.B.shape[0]:
        raise ValueError(f"Incompatible dimensions: l({params.l.shape}), u({params.u.shape}), and B({params.B.shape})")


@dataclass
class CVQPParams:
    """Parameters defining a CVQP problem instance."""

    P: Optional[Union[np.ndarray, sp.sparse.spmatrix]]  # Quadratic cost matrix (or None for linear)
    q: np.ndarray  # Linear cost vector
    A: np.ndarray  # CVaR constraint matrix
    B: Union[np.ndarray, sp.sparse.spmatrix]  # Linear constraint matrix
    l: np.ndarray  # Lower bounds for Bx
    u: np.ndarray  # Upper bounds for Bx
    beta: float  # Probability level for CVaR (0 < beta < 1)
    kappa: float  # CVaR threshold

    def __post_init__(self):
        _validate_range(self.beta, "beta", 0, 1, exclusive=True)
        _validate_dimensions(self)


@dataclass
class CVQPConfig:
    """Configuration parameters for the CVQP solver."""

    max_iter: int = int(1e5)  # Maximum iterations
    rho: float = 1e-2  # Initial penalty parameter
    abstol: float = 1e-4  # Absolute tolerance
    reltol: float = 1e-3  # Relative tolerance
    alpha_over: float = 1.7  # Over-relaxation parameter [1.5, 1.8]
    print_freq: int = 50  # Status update frequency
    mu: float = 10  # Threshold for adaptive rho
    rho_incr: float = 2.0  # Factor for increasing rho
    rho_decr: float = 2.0  # Factor for decreasing rho
    time_limit: float = 7200  # Max runtime in seconds
    dynamic_rho: bool = True  # Adaptive penalty updates

    def __post_init__(self):
        _validate_positive(self.max_iter, "max_iter")
        _validate_positive(self.rho, "rho")
        _validate_positive(self.abstol, "abstol")
        _validate_positive(self.reltol, "reltol")
        _validate_positive(self.time_limit, "time_limit")
        _validate_range(self.alpha_over, "alpha_over", 1, 2)


@dataclass
class CVQPResults:
    """Results from the CVQP solver."""

    x: np.ndarray  # Optimal solution
    iter_count: int  # Iterations performed
    solve_time: float  # Total solve time (seconds)
    objval: list[float]  # Objective values
    r_norm: list[float]  # Primal residual norms
    s_norm: list[float]  # Dual residual norms
    eps_pri: list[float]  # Primal feasibility tolerances
    eps_dual: list[float]  # Dual feasibility tolerances
    rho: list[float]  # Penalty parameter values
    problem_status: str = "unknown"  # Final status

    @property
    def is_optimal(self) -> bool:
        """True if solver status is 'optimal'."""
        return self.problem_status == "optimal"

    @property
    def value(self) -> float:
        """Optimal objective value, or NaN if no iterations completed."""
        return self.objval[-1] if self.objval else float("nan")
