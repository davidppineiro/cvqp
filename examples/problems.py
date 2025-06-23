"""
Classes to generate CVQP problem instances.
"""

from abc import ABC, abstractmethod
import numpy as np
import cvxpy as cp
import scipy.sparse as sp

from cvqp import CVQPParams


class CVQPProblem(ABC):
    """Abstract base class to generate CVQP problem instances."""

    @abstractmethod
    def generate_instance(self, n_vars: int, n_scenarios: int, seed: int | None = None) -> tuple[CVQPParams, cp.Problem]:
        """Generate a problem instance with both CVQP and CVXPY representations."""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Problem name."""
        pass


class PortfolioOptimization(CVQPProblem):
    """Portfolio optimization problem with CVaR constraint."""

    # Constants
    NORMAL_VOLATILITY = 1.0
    LARGE_SCENARIO_THRESHOLD = 1e6

    def __init__(
        self,
        alpha: float = 0.8,
        gamma: float = 1.0,
        nu: float = 0.2,
        sigma: float = 2.0,
        beta: float = 0.95,
        kappa: float = 0.3,
    ):
        self.alpha = alpha
        self.gamma = gamma
        self.nu = nu
        self.sigma = sigma
        self.beta = beta
        self.kappa = kappa

    def generate_instance(self, n_vars: int, n_scenarios: int, seed: int | None = None) -> tuple[CVQPParams, cp.Problem]:
        """Generate portfolio optimization instance."""
        rng = np.random.default_rng(seed)

        # Generate return scenarios (mixture of normal and stress conditions)
        n_normal = int(self.alpha * n_scenarios)
        n_stress = n_scenarios - n_normal

        R = np.empty((n_scenarios, n_vars))
        R[:n_normal] = rng.normal(self.nu, self.NORMAL_VOLATILITY, size=(n_normal, n_vars))
        R[n_normal:] = rng.normal(-self.nu, self.sigma, size=(n_stress, n_vars))

        # Compute statistics
        mu = np.mean(R, axis=0)
        Sigma = self._compute_covariance(R, n_scenarios)

        # Build CVQP parameters
        P = self._create_covariance_matrix(Sigma, n_scenarios)
        q = -mu
        A = -R
        B = sp.vstack([sp.csr_matrix(np.ones(n_vars)), sp.eye(n_vars, format="csr")])
        l = np.concatenate([np.ones(1), np.zeros(n_vars)])
        u = np.concatenate([np.ones(1), np.full(n_vars, np.inf)])

        params = CVQPParams(P=P, q=q, A=A, B=B, l=l, u=u, beta=self.beta, kappa=self.kappa)

        # Build CVXPY problem
        x = cp.Variable(n_vars)
        objective = (self.gamma / 2) * cp.quad_form(x, Sigma, assume_PSD=True) - mu @ x
        constraints = [cp.sum(x) == 1, x >= 0]

        # Add CVaR constraint
        k = int((1 - self.beta) * n_scenarios)
        alpha = self.kappa * k
        constraints.append(cp.sum_largest(A @ x, k) <= alpha)

        problem = cp.Problem(cp.Minimize(objective), constraints)
        return params, problem

    def _compute_covariance(self, R: np.ndarray, n_scenarios: int) -> np.ndarray:
        """Compute covariance matrix, using diagonal approximation for large problems."""
        if n_scenarios >= self.LARGE_SCENARIO_THRESHOLD:
            return np.diag(np.var(R, axis=0))
        else:
            return np.cov(R.T)

    def _create_covariance_matrix(self, Sigma: np.ndarray, n_scenarios: int):
        """Create covariance matrix for optimization (sparse for large problems)."""
        if n_scenarios >= self.LARGE_SCENARIO_THRESHOLD:
            return self.gamma * sp.diags(np.diag(Sigma))
        else:
            return self.gamma * Sigma

    @property
    def name(self) -> str:
        return "portfolio"


class QuantileRegression(CVQPProblem):
    """Quantile regression problem."""

    # Constants
    NOISE_SCALE = 0.1
    NOISE_DOF = 5  # Degrees of freedom for t-distributed noise

    def __init__(self, tau: float = 0.9):
        self.tau = tau

    def generate_instance(self, n_vars: int, n_scenarios: int, seed: int | None = None) -> tuple[CVQPParams, cp.Problem]:
        """Generate quantile regression instance."""
        rng = np.random.default_rng(seed)

        # Generate synthetic regression data
        X = rng.standard_normal((n_scenarios, n_vars))
        beta_true = rng.standard_normal(n_vars) / np.sqrt(1 + np.arange(n_vars))
        y = X @ beta_true + self.NOISE_SCALE * rng.standard_t(df=self.NOISE_DOF, size=n_scenarios)

        # Build CVQP parameters
        A = np.hstack([-X, -np.ones((n_scenarios, 1)), y.reshape(-1, 1)])
        q = np.hstack([np.mean(X, axis=0), 1, 0])
        B = sp.csr_matrix(([1], ([0], [n_vars + 1])), shape=(1, n_vars + 2))
        l = np.ones(1)
        u = np.ones(1)

        params = CVQPParams(P=None, q=q, A=A, B=B, l=l, u=u, beta=self.tau, kappa=0.0)

        # Build CVXPY problem
        x = cp.Variable(n_vars + 2)
        k = int((1 - self.tau) * n_scenarios)
        alpha = 0.0  # kappa * k

        objective = cp.Minimize(q @ x)
        constraints = [cp.sum_largest(A @ x, k) <= alpha, x[-1] == 1]
        problem = cp.Problem(objective, constraints)

        return params, problem

    @property
    def name(self) -> str:
        return "quantile_regression"
