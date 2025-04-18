"""
Type definitions for the CVQP solver.
"""

from dataclasses import dataclass
import numpy as np
import scipy as sp


@dataclass
class CVQPParams:
    """Parameters defining a CVQP problem instance."""
    P: np.ndarray | sp.sparse.spmatrix | None  # Quadratic cost matrix (or None for linear)
    q: np.ndarray  # Linear cost vector
    A: np.ndarray  # CVaR constraint matrix
    B: np.ndarray | sp.sparse.spmatrix  # Linear constraint matrix
    l: np.ndarray  # Lower bounds for Bx
    u: np.ndarray  # Upper bounds for Bx
    beta: float  # Probability level for CVaR (0 < beta < 1)
    kappa: float  # CVaR threshold
    
    def __post_init__(self):
        """Validate problem parameters."""
        if self.beta <= 0 or self.beta >= 1:
            raise ValueError("beta must be between 0 and 1")
        
        # Check dimension compatibility
        if self.P is not None:
            if self.P.shape[0] != self.P.shape[1]:
                raise ValueError("Cost matrix P must be square")
            if self.P.shape[1] != self.q.shape[0]:
                raise ValueError(f"Incompatible dimensions: P({self.P.shape}) and q({self.q.shape})")
        
        if self.A.shape[1] != self.q.shape[0]:
            raise ValueError(f"Incompatible dimensions: A({self.A.shape}) and q({self.q.shape})")
        
        if self.B.shape[1] != self.q.shape[0]:
            raise ValueError(f"Incompatible dimensions: B({self.B.shape}) and q({self.q.shape})")
        
        if self.l.shape[0] != self.B.shape[0] or self.u.shape[0] != self.B.shape[0]:
            raise ValueError(f"Incompatible dimensions: l({self.l.shape}), u({self.u.shape}), and B({self.B.shape})")


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
    verbose: bool = False  # Print detailed information
    time_limit: float = 7200  # Max runtime in seconds
    dynamic_rho: bool = True  # Adaptive penalty updates
    
    def __post_init__(self):
        """Validate solver configuration."""
        if self.max_iter <= 0:
            raise ValueError("max_iter must be positive")
        if self.rho <= 0:
            raise ValueError("rho must be positive")
        if self.abstol <= 0:
            raise ValueError("abstol must be positive")
        if self.reltol <= 0:
            raise ValueError("reltol must be positive")
        if self.alpha_over < 1 or self.alpha_over > 2:
            raise ValueError("alpha_over should be in range [1, 2]")
        if self.time_limit <= 0:
            raise ValueError("time_limit must be positive")


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
        """Check if the solution is optimal."""
        return self.problem_status == "optimal"
    
    @property
    def final_objective(self) -> float:
        """Get the final objective value."""
        return self.objval[-1] if self.objval else float('nan')