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
    def generate_data(
        self, n_vars: int, n_scenarios: int, seed: int | None = None
    ) -> None:
        """
        Generate and store problem data for an instance of a given size.

        Args:
            n_vars: Number of decision variables
            n_scenarios: Number of scenarios
            seed: Random seed for reproducibility
        """
        pass

    @abstractmethod
    def generate_instance(
        self, n_vars: int, n_scenarios: int, seed: int | None = None
    ) -> tuple[CVQPParams, cp.Problem]:
        """
        Generate a problem instance of a given size with both CVQP and CVXPY representations.

        Args:
            n_vars: Number of decision variables
            n_scenarios: Number of scenarios
            seed: Random seed for reproducibility

        Returns:
            Tuple containing:
                - CVQPParams instance with the generated problem parameters
                - CVXPY Problem instance with the equivalent optimization problem
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Problem name."""
        pass


class PortfolioOptimization(CVQPProblem):
    """
    Portfolio optimization problem.

    Args:
        alpha: Probability of normal market conditions
        gamma: Risk aversion parameter for variance term
        nu: Mean return in normal market conditions
        sigma: Volatility scaling factor for stress periods
        beta: CVaR probability level
        kappa: CVaR threshold
    """

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

    def generate_data(
        self, n_vars: int, n_scenarios: int, seed: int | None = None
    ) -> None:
        """
        Generates and stores problem data.

        Generates return scenarios from a mixture of two normal distributions
        representing normal and stress market conditions. Computes and stores
        the mean returns and covariance matrix.

        Args:
            n_vars: Number of assets
            n_scenarios: Number of return scenarios
            seed: Random seed for reproducibility
        """
        rng = np.random.default_rng(seed)

        # Calculate number of scenarios for normal and stress market conditions
        n_normal = int(self.alpha * n_scenarios)
        n_stress = n_scenarios - n_normal

        # Generate return matrix
        R = np.empty((n_scenarios, n_vars))
        R[:n_normal] = rng.normal(self.nu, 1.0, size=(n_normal, n_vars))
        R[n_normal:] = rng.normal(-self.nu, self.sigma, size=(n_stress, n_vars))

        # Compute mean and sample covariance matrix
        mu = np.mean(R, axis=0)
        Sigma = np.cov(R.T) if n_scenarios < 1e6 else np.diag(np.var(R, axis=0))

        self.R = R
        self.mu = mu
        self.Sigma = Sigma

    def generate_instance(
        self, n_vars: int, n_scenarios: int, seed: int | None = None
    ) -> tuple[CVQPParams, cp.Problem]:
        """
        Generates the CVQP parameter representation and the equivalent CVXPY
        problem formulation for the problem instance.

        Args:
            n_vars: Number of assets
            n_scenarios: Number of return scenarios
            seed: Random seed for reproducibility

        Returns:
            Tuple containing:
                - CVQPParams with the generated problem parameters
                - CVXPY Problem instance
        """
        self.generate_data(n_vars, n_scenarios, seed)

        # CVQP representation
        # Convert Sigma to sparse if it's diagonal (large scenario case)
        if n_scenarios >= 1e6:
            P = self.gamma * sp.diags(np.diag(self.Sigma))
        else:
            P = self.gamma * self.Sigma
        q = -self.mu
        A = -self.R
        B = sp.vstack([sp.csr_matrix(np.ones(n_vars)), sp.eye(n_vars, format="csr")])
        l = np.concatenate([np.ones(1), np.zeros(n_vars)])
        u = np.concatenate([np.ones(1), np.full(n_vars, np.inf)])

        params = CVQPParams(
            P=P, q=q, A=A, B=B, l=l, u=u, beta=self.beta, kappa=self.kappa
        )

        # CVXPY problem
        x = cp.Variable(n_vars)
        objective = (self.gamma / 2) * cp.quad_form(
            x, self.Sigma, assume_PSD=True
        ) - self.mu @ x
        constraints = [cp.sum(x) == 1, x >= 0]

        # Add CVaR constraint
        n_scenarios = params.A.shape[0]
        k = int((1 - params.beta) * n_scenarios)
        alpha = params.kappa * k
        constraints.append(cp.sum_largest(params.A @ x, k) <= alpha)
        # constraints.append(cp.cvar(-self.R @ x, self.beta) <= self.kappa)

        problem = cp.Problem(cp.Minimize(objective), constraints)

        return params, problem

    @property
    def name(self) -> str:
        """Return the name identifier for this problem."""
        return "portfolio"


class QuantileRegression(CVQPProblem):
    """
    Quantile regression problem.

    Args:
        tau: Quantile level (between 0 and 1)
    """

    def __init__(
        self,
        tau: float = 0.9,
    ):
        self.tau = tau

    def generate_data(
        self, n_vars: int, n_scenarios: int, seed: int | None = None
    ) -> None:
        """
        Generate and store synthetic regression data.

        Generates predictor variables and response values from a linear model
        with t-distributed noise. The true coefficients decay as 1/sqrt(1+k)
        where k is the predictor index.

        Args:
            n_vars: Number of predictor variables
            n_scenarios: Number of observations
            seed: Random seed for reproducibility
        """
        rng = np.random.default_rng(seed)

        # Generate synthetic data
        X = rng.standard_normal((n_scenarios, n_vars))
        beta_true = rng.standard_normal(n_vars) / np.sqrt(1 + np.arange(n_vars))
        y = X @ beta_true + 0.1 * rng.standard_t(df=5, size=n_scenarios)

        self.X = X
        self.y = y

    def generate_instance(
        self, n_vars: int, n_scenarios: int, seed: int | None = None
    ) -> tuple[CVQPParams, cp.Problem]:
        """
        Generates the CVQP parameter representation and the equivalent CVXPY
        problem formulation for the problem instance.

        Args:
            n_vars: Number of predictor variables
            n_scenarios: Number of observations
            seed: Random seed for reproducibility

        Returns:
            Tuple containing:
                - CVQPParams with the generated problem parameters
                - CVXPY Problem instance
        """
        self.generate_data(n_vars, n_scenarios, seed)

        # CVQP representation
        A = np.hstack([-self.X, -np.ones((n_scenarios, 1)), self.y.reshape(-1, 1)])
        q = np.hstack([np.mean(self.X, axis=0), 1, 0])
        B = sp.csr_matrix(([1], ([0], [n_vars + 1])), shape=(1, n_vars + 2))
        l = np.ones(1)
        u = np.ones(1)

        params = CVQPParams(P=None, q=q, A=A, B=B, l=l, u=u, beta=self.tau, kappa=0.0)

        # # CVXPY problem (natural tilted absolute formulation)
        # a = cp.Variable(self.X.shape[1])  # Regression coefficients
        # b = cp.Variable()  # Intercept term
        # residuals = self.y - (self.X @ a + b)
        # loss = (
        #     cp.sum(0.5 * cp.abs(residuals) + (self.tau - 0.5) * residuals) / n_scenarios
        # )
        # problem = cp.Problem(cp.Minimize(loss))

        # CVXPY problem (CVaR formulation)
        x = cp.Variable(n_vars + 2)  # [beta; t; s] where s is the auxiliary variable
        n_scenarios = params.A.shape[0]
        k = int((1 - params.beta) * n_scenarios)
        alpha = params.kappa * k

        objective = cp.Minimize(q @ x)
        constraints = [cp.sum_largest(params.A @ x, k) <= alpha, x[-1] == 1]
        problem = cp.Problem(objective, constraints)

        return params, problem

    @property
    def name(self) -> str:
        """Return the name identifier for this problem."""
        return "quantile_regression"