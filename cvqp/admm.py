"""
CVQP: A solver for CVaR-constrained quadratic programs.
"""

import logging
import time
from typing import Union, Optional
import numpy as np
import scipy as sp

from .types import CVQPParams, CVQPConfig, CVQPResults
from .projection import proj_sum_largest

# Logging constants
_SEPARATOR_WIDTH = 83
_LOG_FORMAT = "%(asctime)s: %(message)s"
_LOG_DATE_FORMAT = "%b %d %H:%M:%S"

# Status constants
_STATUS_OPTIMAL = "optimal"
_STATUS_TIMEOUT = "timeout"
_STATUS_UNKNOWN = "unknown"

logger = logging.getLogger(__name__)


def _setup_cvqp_logging():
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(_LOG_FORMAT, _LOG_DATE_FORMAT)
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)


class CVQP:
    """Solver for CVaR-constrained quadratic programs using ADMM."""

    def __init__(self, params: CVQPParams):
        self.params = params

    def solve(self, warm_start: Optional[np.ndarray] = None, verbose: bool = False, options: Optional[CVQPConfig] = None) -> CVQPResults:
        """Solve the CVaR-constrained quadratic program using ADMM.

        Args:
            warm_start: Initial solution guess. If None, uses zero initialization.
            verbose: Whether to print detailed solver progress during optimization.
            options: Solver configuration parameters (tolerances, iteration limits, etc.).

        Returns:
            CVQPResults containing the optimal solution, convergence history, and solver status.

        Raises:
            ValueError: If problem parameters are incompatible or invalid.
        """
        self.options = options or CVQPConfig()
        self.verbose = verbose

        self._initialize_problem()

        start_time = time.time()

        z, u, z_tilde, u_tilde, results = self._initialize_variables(warm_start)
        rho = self.options.rho

        if self.verbose:
            _setup_cvqp_logging()
            self._log_header()
        for i in range(self.options.max_iter):
            z_old, z_tilde_old = z.copy(), z_tilde.copy()

            # ADMM primal updates
            x = self._x_update(z, u, z_tilde, u_tilde, rho)
            z = self._z_update(x, z, u, self.options.alpha_over)
            z_tilde = self._z_tilde_update(x, z_tilde, u_tilde, self.options.alpha_over)

            # ADMM dual updates
            u += self.options.alpha_over * (self.params.A @ x) + (1 - self.options.alpha_over) * z_old - z
            Bx = self._ensure_dense(self.params.B @ x)
            u_tilde += self.options.alpha_over * Bx + (1 - self.options.alpha_over) * z_tilde_old - z_tilde

            results.objval.append(self._compute_objective(x))
            results.rho.append(rho)

            should_terminate, status = self._check_convergence_and_log(i, start_time, x, z, z_tilde, z_old, z_tilde_old, rho, results)
            if should_terminate:
                results.problem_status = status
                break

            # Adaptive penalty parameter update
            if self.options.dynamic_rho and i % self.options.print_freq == 0:
                r_norm, s_norm, _, _ = self._compute_residuals(x, z, z_tilde, z_old, z_tilde_old, rho)
                rho, u, u_tilde = self._update_rho(rho, r_norm, s_norm, u, u_tilde)

        self._unscale_problem()
        results.x = x
        results.iter_count = i + 1
        results.solve_time = time.time() - start_time

        if self.verbose:
            self._log_final(results)

        return results

    def _initialize_problem(self):
        """Initialize problem by scaling and precomputing matrices."""
        self._scale_problem()
        self._setup_cvar_params()
        self._precompute_matrices()

    def _scale_problem(self):
        """Scale problem data for better numerical conditioning."""
        self.scale = max(np.max(np.abs(self.params.A)), 1.0)
        self.params.A = self._scale_matrix(self.params.A, self.scale)
        self.params.q = self._scale_matrix(self.params.q, self.scale)
        if self.params.P is not None:
            self.params.P = self._scale_matrix(self.params.P, self.scale, inplace=False)

    def _setup_cvar_params(self):
        """Initialize CVaR parameters from problem dimensions."""
        self.m = self.params.A.shape[0]
        # Number of scenarios in the (1-beta) tail for CVaR
        self.k = int((1 - self.params.beta) * self.m)
        # Scale-adjusted constraint threshold
        self.alpha = self.params.kappa * self.k / self.scale

    def _precompute_matrices(self):
        """Cache frequently used matrix products."""
        self.AtA = self.params.A.T @ self.params.A
        self.BtB = self.params.B.T @ self.params.B
        self._update_M_factor(self.options.rho)

    def _ensure_dense(self, matrix):
        """Convert sparse matrix to dense if needed."""
        if sp.sparse.issparse(matrix):
            return matrix.toarray().ravel() if matrix.shape[1] == 1 else matrix.toarray()
        return matrix

    def _scale_matrix(self, matrix, scale_factor, inplace=True):
        """Unified sparse/dense matrix scaling."""
        if sp.sparse.issparse(matrix):
            return matrix / scale_factor
        elif inplace:
            matrix /= scale_factor
            return matrix
        else:
            return matrix / scale_factor

    def _compute_system_matrix(self, rho: float):
        """Compute the system matrix M = P + rho*(A^T*A + B^T*B)."""
        penalty_term = rho * (self.AtA + self.BtB)

        if self.params.P is None:
            M = penalty_term
        else:
            M = self.params.P + penalty_term

        # Preserve sparsity unless all input matrices are dense
        if not sp.sparse.issparse(M):
            return M
        elif not sp.sparse.issparse(self.AtA) and not sp.sparse.issparse(self.BtB) and (self.params.P is None or not sp.sparse.issparse(self.params.P)):
            return M.toarray()
        else:
            return M

    def _update_M_factor(self, rho: float):
        """Update and factorize the linear system matrix."""
        self.M = self._compute_system_matrix(rho)

        if sp.sparse.issparse(self.M):
            try:
                from scikits.sparse.cholmod import cholesky

                # CHOLMOD is fastest for sparse SPD systems
                self.factor = cholesky(self.M)
                self.use_cholesky = True
                self.use_sparse = True
            except ImportError:
                self.factor = sp.sparse.linalg.splu(self.M)
                self.use_cholesky = False
                self.use_sparse = True
        else:
            self.use_sparse = False
            try:
                # Cholesky is 2x faster than LU for SPD systems
                self.factor = sp.linalg.cho_factor(self.M)
                self.use_cholesky = True
            except np.linalg.LinAlgError:
                self.factor = sp.linalg.lu_factor(self.M)
                self.use_cholesky = False

    def _initialize_variables(self, warm_start: Optional[np.ndarray]) -> tuple:
        """Set up initial optimization variables and results storage."""
        n = self.params.q.shape[0]
        B_rows = self.params.B.shape[0]

        if warm_start is None:
            x = np.zeros(n)
            z = np.zeros(self.m)
            z_tilde = np.zeros(B_rows)
        else:
            x = warm_start.copy()
            # Initialize auxiliary variables to be feasible
            z = self.params.A @ x
            z_tilde = self._ensure_dense(self.params.B @ x)

        u = np.zeros(self.m)
        u_tilde = np.zeros(B_rows)
        results = CVQPResults(x=x, iter_count=0, solve_time=0, objval=[], r_norm=[], s_norm=[], eps_pri=[], eps_dual=[], rho=[])

        return z, u, z_tilde, u_tilde, results

    def _x_update(
        self,
        z: np.ndarray,
        u: np.ndarray,
        z_tilde: np.ndarray,
        u_tilde: np.ndarray,
        rho: float,
    ) -> np.ndarray:
        """Perform x-minimization step of ADMM."""
        rhs = -self.params.q + rho * (self.params.A.T @ (z - u)) + rho * (self.params.B.T @ (z_tilde - u_tilde))

        if self.use_sparse:
            if self.use_cholesky:
                return self.factor.solve_A(rhs)
            else:
                return self.factor.solve(rhs)
        else:
            if self.use_cholesky:
                return sp.linalg.cho_solve(self.factor, rhs)
            else:
                return sp.linalg.lu_solve(self.factor, rhs)

    def _z_update(self, x: np.ndarray, z: np.ndarray, u: np.ndarray, alpha_over: float) -> np.ndarray:
        """Update z variable with projection onto sum-k-largest constraint."""
        # Over-relaxation step for faster convergence
        z_hat = alpha_over * (self.params.A @ x) + (1 - alpha_over) * z + u
        return proj_sum_largest(z_hat, self.k, self.alpha)

    def _z_tilde_update(self, x: np.ndarray, z_tilde: np.ndarray, u_tilde: np.ndarray, alpha_over: float) -> np.ndarray:
        """Update z_tilde variable with box projection."""
        Bx = self._ensure_dense(self.params.B @ x)
        z_hat_tilde = alpha_over * Bx + (1 - alpha_over) * z_tilde + u_tilde
        # Project onto box constraints [l, u]
        return np.clip(z_hat_tilde, self.params.l, self.params.u)

    def _compute_residuals(
        self,
        x: np.ndarray,
        z: np.ndarray,
        z_tilde: np.ndarray,
        z_old: np.ndarray,
        z_tilde_old: np.ndarray,
        rho: float,
    ) -> tuple:
        """Compute primal and dual residuals for convergence check."""
        Ax = self.params.A @ x
        Bx = self._ensure_dense(self.params.B @ x)

        # Primal residual (constraint violations)
        r = np.concatenate([Ax - z, Bx - z_tilde])
        r_norm = np.linalg.norm(r)

        # Dual residual from variable changes
        z_diff = z - z_old
        z_tilde_diff = z_tilde - z_tilde_old
        Bt_z = self.params.B.T @ z_tilde_diff
        At_z = self.params.A.T @ z_diff + self._ensure_dense(Bt_z)
        s_norm = np.linalg.norm(rho * At_z)

        return r_norm, s_norm, Ax, At_z

    def _check_convergence_and_log(self, i, start_time, x, z, z_tilde, z_old, z_tilde_old, rho, results):
        """Handle convergence checking, logging, and termination decisions.

        Returns:
            (should_terminate, status): Tuple indicating if solver should stop and final status.
        """
        # Skip expensive convergence checks between print intervals
        if i % self.options.print_freq != 0:
            return False, None

        # Compute residuals and tolerances
        r_norm, s_norm, Ax, At_z = self._compute_residuals(x, z, z_tilde, z_old, z_tilde_old, rho)
        eps_pri, eps_dual = self._compute_tolerances(Ax, z, z_tilde, At_z, rho)

        self._record_iteration(results, r_norm, s_norm, eps_pri, eps_dual)

        # Log progress if verbose
        if self.verbose:
            self._log_iteration(i, r_norm, eps_pri, s_norm, eps_dual, rho, results.objval[-1])

        # Check termination conditions
        if time.time() - start_time > self.options.time_limit:
            return True, _STATUS_TIMEOUT

        if self._check_convergence(r_norm, s_norm, eps_pri, eps_dual):
            return True, _STATUS_OPTIMAL

        return False, None

    def _compute_tolerances(self, Ax, z, z_tilde, At_z, rho) -> tuple:
        """Compute primal and dual feasibility tolerances."""
        sqrt_d0 = (self.m + self.params.B.shape[0]) ** 0.5
        sqrt_d1 = self.params.A.shape[1] ** 0.5
        eps_pri = sqrt_d0 * self.options.abstol + self.options.reltol * max(np.linalg.norm(Ax), np.linalg.norm(np.concatenate([z, z_tilde])))
        eps_dual = sqrt_d1 * self.options.abstol + self.options.reltol * np.linalg.norm(rho * self._ensure_dense(At_z))
        return eps_pri, eps_dual

    def _check_convergence(self, r_norm: float, s_norm: float, eps_pri: float, eps_dual: float) -> bool:
        """Check if convergence criteria are satisfied."""
        return r_norm <= eps_pri and s_norm <= eps_dual

    def _update_rho(self, rho: float, r_norm: float, s_norm: float, u: np.ndarray, u_tilde: np.ndarray) -> tuple:
        """Update penalty parameter adaptively based on residuals."""
        if r_norm > self.options.mu * s_norm:
            # Primal infeasibility dominates - increase penalty
            rho *= self.options.rho_incr
            u /= self.options.rho_incr
            u_tilde /= self.options.rho_incr
            self._update_M_factor(rho)
        elif s_norm > self.options.mu * r_norm:
            # Dual infeasibility dominates - decrease penalty
            rho /= self.options.rho_decr
            u *= self.options.rho_decr
            u_tilde *= self.options.rho_decr
            self._update_M_factor(rho)
        return rho, u, u_tilde

    def _log_header(self):
        """Print solver header and column titles."""
        sep = "=" * _SEPARATOR_WIDTH
        logger.info(sep)
        logger.info("CVQP solver".center(_SEPARATOR_WIDTH))
        logger.info(sep)
        logger.info("{:<6} {:<12} {:<12} {:<12} {:<12} {:<12} {:<12}".format("iter", "r_norm", "eps_pri", "s_norm", "eps_dual", "rho", "obj_val"))
        logger.info("-" * _SEPARATOR_WIDTH)

    def _log_iteration(self, i, r_norm, eps_pri, s_norm, eps_dual, rho, objval):
        """Print iteration progress."""
        logger.info("{:<6} {:<12.3e} {:<12.3e} {:<12.3e} {:<12.3e} {:<12.2e} {:<12.3e}".format(i, r_norm, eps_pri, s_norm, eps_dual, rho, objval))

    def _log_final(self, results):
        """Print final results."""
        sep = "=" * _SEPARATOR_WIDTH
        logger.info(sep)
        logger.info(f"Optimal value: {results.objval[-1]:.3e}")
        logger.info(f"Solver took {results.solve_time:.2f} seconds")
        logger.info(f"Problem status: {results.problem_status}")

    def _compute_objective(self, x: np.ndarray) -> float:
        """Calculate objective value with efficient handling of P."""
        if self.params.P is None:
            return (self.params.q @ x) * self.scale

        if sp.sparse.issparse(self.params.P):
            Px = self.params.P @ x
            return (0.5 * x.T @ Px + self.params.q @ x) * self.scale
        else:
            return (0.5 * np.dot(x, self.params.P @ x) + self.params.q @ x) * self.scale

    def _record_iteration(
        self,
        results: CVQPResults,
        r_norm: float,
        s_norm: float,
        eps_pri: float,
        eps_dual: float,
    ):
        """Record convergence metrics."""
        results.r_norm.append(r_norm)
        results.s_norm.append(s_norm)
        results.eps_pri.append(eps_pri)
        results.eps_dual.append(eps_dual)

    def _unscale_problem(self):
        """Restore original scaling for final results."""
        self.params.A = self._scale_matrix(self.params.A, 1 / self.scale)
        self.params.q = self._scale_matrix(self.params.q, 1 / self.scale)
        if self.params.P is not None:
            self.params.P = self._scale_matrix(self.params.P, 1 / self.scale, inplace=False)


def solve_cvqp(
    P: Optional[Union[np.ndarray, sp.sparse.spmatrix]] = None,
    q: np.ndarray = None,
    A: np.ndarray = None,
    B: Union[np.ndarray, sp.sparse.spmatrix] = None,
    l: np.ndarray = None,
    u: np.ndarray = None,
    beta: float = None,
    kappa: float = None,
    warm_start: Optional[np.ndarray] = None,
    verbose: bool = False,
    **solver_options,
) -> CVQPResults:
    """Solve CVaR-constrained QP (functional API).

    Args:
        P: Quadratic cost matrix (n x n), or None for linear problems.
        q: Linear cost vector (n,).
        A: CVaR constraint matrix (m x n).
        B: Linear constraint matrix for box constraints.
        l: Lower bounds for Bx.
        u: Upper bounds for Bx.
        beta: Probability level for CVaR (0 < beta < 1).
        kappa: CVaR threshold.
        warm_start: Initial solution guess.
        verbose: Whether to print solver progress.
        **solver_options: Additional solver configuration options (max_iter, abstol, reltol, etc.).

    Returns:
        CVQPResults containing the optimal solution and solver statistics.
    """
    # Validate required parameters
    required_params = [q, A, B, l, u, beta, kappa]
    if any(param is None for param in required_params):
        missing = [name for name, param in zip(["q", "A", "B", "l", "u", "beta", "kappa"], required_params) if param is None]
        raise ValueError(f"Missing required parameters: {missing}")

    # Create parameter and configuration objects
    params = CVQPParams(P=P, q=q, A=A, B=B, l=l, u=u, beta=beta, kappa=kappa)
    config = CVQPConfig(**solver_options)

    # Solve using the OOP API
    solver = CVQP(params)
    return solver.solve(warm_start=warm_start, verbose=verbose, options=config)
