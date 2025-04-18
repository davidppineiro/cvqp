"""
CVQP: A solver for conditional value-at-risk (CVaR) constrained quadratic programs.
"""

import logging
import time
import numpy as np
import scipy as sp

from .types import CVQPParams, CVQPConfig, CVQPResults
from .projection import proj_sum_largest

# Constants
_SEPARATOR_WIDTH = 83
_LOG_FORMAT = "%(asctime)s: %(message)s"
_LOG_DATE_FORMAT = "%b %d %H:%M:%S"
_STATUS_OPTIMAL = "optimal"
_STATUS_TIMEOUT = "timeout"
_STATUS_UNKNOWN = "unknown"

logging.basicConfig(
    level=logging.INFO, format=_LOG_FORMAT, datefmt=_LOG_DATE_FORMAT
)


class CVQP:
    """CVQP solver for CVaR-constrained quadratic programs.
    
    This solver implements the Alternating Direction Method of Multipliers (ADMM)
    algorithm to solve problems with CVaR constraints and box constraints.
    """

    def __init__(self, params: CVQPParams, options: CVQPConfig = CVQPConfig()):
        """Initialize solver with problem parameters and configuration options."""
        self.params = params
        self.options = options
        self.initialize_problem()

    def initialize_problem(self):
        """Initialize problem by scaling and precomputing matrices."""
        self.scale_problem()
        self.setup_cvar_params()
        self.precompute_matrices()

    def scale_problem(self):
        """Scale problem data for better numerical conditioning."""
        self.scale = max(-self.params.A.min(), self.params.A.max())
        self.params.A /= self.scale
        self.params.q /= self.scale
        if self.params.P is not None:
            if sp.sparse.issparse(self.params.P):
                self.params.P = self.params.P / self.scale
            else:
                self.params.P /= self.scale

    def setup_cvar_params(self):
        """Initialize CVaR parameters from problem dimensions."""
        self.m = self.params.A.shape[0]
        self.k = int((1 - self.params.beta) * self.m)
        self.alpha = self.params.kappa * self.k / self.scale

    def precompute_matrices(self):
        """Cache frequently used matrix products."""
        self.AtA = self.params.A.T @ self.params.A
        self.BtB = self.params.B.T @ self.params.B
        self.update_M_factor(self.options.rho)

    def _ensure_dense(self, matrix):
        """Convert sparse matrix to dense if needed."""
        if sp.sparse.issparse(matrix):
            if matrix.shape[1] == 1:  # Vector
                return matrix.toarray().ravel()
            else:  # Matrix
                return matrix.toarray()
        return matrix

    def update_M_factor(self, rho: float):
        """Update and factorize the linear system matrix M."""
        BtB_dense = self._ensure_dense(self.BtB)
        
        if self.params.P is None:
            self.M = rho * (self.AtA + BtB_dense)
        else:
            P_dense = self._ensure_dense(self.params.P)
            self.M = P_dense + rho * (self.AtA + BtB_dense)

        self.factor = sp.linalg.lu_factor(self.M)

    def initialize_variables(self, warm_start: np.ndarray | None) -> tuple:
        """Set up initial optimization variables and results storage."""
        n = self.params.q.shape[0]  # Problem dimension
        
        if warm_start is None:
            x = np.zeros(n)
            z = np.zeros(self.m)
            z_tilde = np.zeros(self.params.B.shape[0])
        else:
            x = warm_start.copy()
            z = self.params.A @ warm_start
            z_tilde = self._ensure_dense(self.params.B @ warm_start)

        u = np.zeros(self.m)
        u_tilde = np.zeros(self.params.B.shape[0])

        results = CVQPResults(
            x=x,
            iter_count=0,
            solve_time=0,
            objval=[],
            r_norm=[],
            s_norm=[],
            eps_pri=[],
            eps_dual=[],
            rho=[],
        )

        return z, u, z_tilde, u_tilde, results

    def x_update(
        self,
        z: np.ndarray,
        u: np.ndarray,
        z_tilde: np.ndarray,
        u_tilde: np.ndarray,
        rho: float,
    ) -> np.ndarray:
        """Perform x-minimization step of ADMM."""
        rhs = (
            -self.params.q
            + rho * (self.params.A.T @ (z - u))
            + rho * (self.params.B.T @ (z_tilde - u_tilde))
        )
        return sp.linalg.lu_solve(self.factor, rhs)

    def z_update(
        self, x: np.ndarray, z: np.ndarray, u: np.ndarray, alpha_over: float
    ) -> np.ndarray:
        """Update z variable with projection onto sum-k-largest constraint."""
        z_hat = alpha_over * (self.params.A @ x) + (1 - alpha_over) * z + u
        return proj_sum_largest(z_hat, self.k, self.alpha)

    def z_tilde_update(
        self, x: np.ndarray, z_tilde: np.ndarray, u_tilde: np.ndarray, alpha_over: float
    ) -> np.ndarray:
        """Update z_tilde variable with box projection."""
        Bx = self._ensure_dense(self.params.B @ x)
        z_hat_tilde = alpha_over * Bx + (1 - alpha_over) * z_tilde + u_tilde
        return np.clip(z_hat_tilde, self.params.l, self.params.u)

    def compute_residuals(
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

        # Changes in the variables for dual residual
        z_diff = z - z_old
        z_tilde_diff = z_tilde - z_tilde_old

        # Compute dual residual components
        Bt_z = self.params.B.T @ z_tilde_diff
        At_z = self.params.A.T @ z_diff + self._ensure_dense(Bt_z)
        s_norm = np.linalg.norm(rho * At_z)

        return r_norm, s_norm, Ax, At_z

    def compute_tolerances(
        self,
        Ax: np.ndarray,
        z: np.ndarray,
        z_tilde: np.ndarray,
        At_z: np.ndarray,
        rho: float,
    ) -> tuple:
        """Compute primal and dual feasibility tolerances."""
        d0 = self.params.A.shape[0] + self.params.B.shape[0]
        d1 = self.params.A.shape[1]
        At_z = self._ensure_dense(At_z)

        eps_pri = (d0**0.5) * self.options.abstol + self.options.reltol * max(
            np.linalg.norm(Ax), np.linalg.norm(np.concatenate([z, z_tilde]))
        )
        eps_dual = (d1**0.5) * self.options.abstol + self.options.reltol * np.linalg.norm(rho * At_z)

        return eps_pri, eps_dual

    def check_convergence(
        self, r_norm: float, s_norm: float, eps_pri: float, eps_dual: float
    ) -> bool:
        """Check if convergence criteria are satisfied."""
        return r_norm <= eps_pri and s_norm <= eps_dual

    def update_rho(
        self,
        rho: float,
        r_norm: float,
        s_norm: float,
        u: np.ndarray,
        u_tilde: np.ndarray,
    ) -> tuple:
        """Update penalty parameter adaptively based on residuals."""
        if r_norm > self.options.mu * s_norm:
            # Primal residual too large - increase rho
            rho *= self.options.rho_incr
            u /= self.options.rho_incr
            u_tilde /= self.options.rho_incr
            self.update_M_factor(rho)
        elif s_norm > self.options.mu * r_norm:
            # Dual residual too large - decrease rho
            rho /= self.options.rho_decr
            u *= self.options.rho_decr
            u_tilde *= self.options.rho_decr
            self.update_M_factor(rho)
        return rho, u, u_tilde

    def setup_progress_display(self):
        """Set up formatting for iteration progress display."""
        self.header_titles = [
            "iter", "r_norm", "eps_pri", "s_norm", "eps_dual", "rho", "obj_val"
        ]
        self.header_format = "{:<6} {:<12} {:<12} {:<12} {:<12} {:<12} {:<12}"
        self.row_format = "{:<6} {:<12.3e} {:<12.3e} {:<12.3e} {:<12.3e} {:<12.2e} {:<12.3e}"
        self.separator = "=" * _SEPARATOR_WIDTH

        logging.info(self.separator)
        title = "CVQP solver"
        logging.info(title.center(_SEPARATOR_WIDTH))
        logging.info(self.separator)
        logging.info(self.header_format.format(*self.header_titles))
        logging.info("-" * _SEPARATOR_WIDTH)

    def print_iteration(
        self,
        iteration: int,
        r_norm: float,
        eps_pri: float,
        s_norm: float,
        eps_dual: float,
        rho: float,
        objval: float,
    ):
        """Print current iteration status."""
        logging.info(
            self.row_format.format(
                iteration, r_norm, eps_pri, s_norm, eps_dual, rho, objval
            )
        )

    def print_final_results(self, results: CVQPResults):
        """Print optimization results summary."""
        logging.info(self.separator)
        logging.info(f"Optimal value: {results.objval[-1]:.3e}")
        logging.info(f"Solver took {results.solve_time:.2f} seconds")
        logging.info(f"Problem status: {results.problem_status}")

    def _compute_objective(self, x: np.ndarray) -> float:
        """Calculate objective value with efficient handling of P."""
        if self.params.P is None:
            return (self.params.q @ x) * self.scale
            
        if sp.sparse.issparse(self.params.P):
            Px = self.params.P @ x
            return (0.5 * x.T @ Px + self.params.q @ x) * self.scale
        else:
            return (0.5 * np.dot(x, self.params.P @ x) + self.params.q @ x) * self.scale

    def record_iteration(
        self,
        results: CVQPResults,
        x: np.ndarray,
        r_norm: float,
        s_norm: float,
        eps_pri: float,
        eps_dual: float,
        rho: float,
    ):
        """Record iteration data for convergence analysis."""
        results.objval.append(self._compute_objective(x))
        results.r_norm.append(r_norm)
        results.s_norm.append(s_norm)
        results.eps_pri.append(eps_pri)
        results.eps_dual.append(eps_dual)
        results.rho.append(rho)

    def unscale_problem(self):
        """Restore original scaling for final results."""
        self.params.A *= self.scale
        self.params.q *= self.scale
        if self.params.P is not None:
            if sp.sparse.issparse(self.params.P):
                self.params.P = self.params.P * self.scale
            else:
                self.params.P *= self.scale

    def solve(self, warm_start: np.ndarray | None = None) -> CVQPResults:
        """Solve CVaR-constrained QP using ADMM algorithm.
        
        Args:
            warm_start: Optional initial guess for the solution
            
        Returns:
            CVQPResults object containing the solution and solver statistics
        """
        start_time = time.time()

        # Initialize variables and results
        z, u, z_tilde, u_tilde, results = self.initialize_variables(warm_start)
        rho = self.options.rho

        if self.options.verbose:
            self.setup_progress_display()

        # Main iteration loop
        for i in range(self.options.max_iter):
            # Store previous values
            z_old, z_tilde_old = z.copy(), z_tilde.copy()

            # Update primal and dual variables
            x = self.x_update(z, u, z_tilde, u_tilde, rho)
            z = self.z_update(x, z, u, self.options.alpha_over)
            z_tilde = self.z_tilde_update(x, z_tilde, u_tilde, self.options.alpha_over)

            # Update dual variables
            u += (
                self.options.alpha_over * (self.params.A @ x)
                + (1 - self.options.alpha_over) * z_old
                - z
            )
            Bx = self._ensure_dense(self.params.B @ x)
            u_tilde += (
                self.options.alpha_over * Bx
                + (1 - self.options.alpha_over) * z_tilde_old
                - z_tilde
            )

            # Check convergence periodically
            if i % self.options.print_freq == 0:
                r_norm, s_norm, Ax, At_z = self.compute_residuals(
                    x, z, z_tilde, z_old, z_tilde_old, rho
                )
                eps_pri, eps_dual = self.compute_tolerances(Ax, z, z_tilde, At_z, rho)

                # Record iteration data
                self.record_iteration(
                    results, x, r_norm, s_norm, eps_pri, eps_dual, rho
                )

                if self.options.verbose:
                    self.print_iteration(
                        i, r_norm, eps_pri, s_norm, eps_dual, rho, results.objval[-1]
                    )

                # Check termination conditions
                if time.time() - start_time > self.options.time_limit:
                    results.problem_status = _STATUS_TIMEOUT
                    break

                if self.check_convergence(r_norm, s_norm, eps_pri, eps_dual):
                    results.problem_status = _STATUS_OPTIMAL
                    break

                # Update penalty parameter if needed
                if self.options.dynamic_rho:
                    rho, u, u_tilde = self.update_rho(rho, r_norm, s_norm, u, u_tilde)

        # Finalize results
        self.unscale_problem()
        results.x = x
        results.iter_count = i + 1
        results.solve_time = time.time() - start_time

        if self.options.verbose:
            self.print_final_results(results)

        return results