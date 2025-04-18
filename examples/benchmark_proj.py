"""
Script to benchmark sum-k-largest projection.
"""

import numpy as np
import cvxpy as cp
import time
import pickle
from dataclasses import dataclass
from pathlib import Path
import logging
import warnings

from cvqp.projection import proj_sum_largest

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s: %(message)s", datefmt="%b %d %H:%M:%S"
)

warnings.filterwarnings("ignore", module="cvxpy")

# Constants
MAX_TIME_LIMIT = 7200  # Maximum time limit in seconds
DEFAULT_TAU = 0.5      # Default hardness parameter

SOLVER_CONFIGS = {
    "MOSEK": {"mosek_params": {"MSK_DPAR_OPTIMIZER_MAX_TIME": MAX_TIME_LIMIT}},
    "CLARABEL": {"time_limit": MAX_TIME_LIMIT},
}


@dataclass
class Projection:
    """
    Single instance of a sum-k-largest projection problem.

    The problem involves projecting a vector onto the set where the sum of its
    k largest elements is bounded by d.

    The "hardness" of the problem is controlled by tau: as tau approaches 1.0,
    the problem becomes easier since the bound is looser. As tau approaches 0.0,
    the problem becomes harder since it forces more elements to be reduced.

    Args:
        v: Input vector to project
        k: Number of largest elements to consider in sum constraint
        d: Upper bound on sum of k largest elements
        tau: Hardness parameter in (0,1) controlling problem difficulty
        vector_size: Length of vector v (number of scenarios)
        seed: Random seed used to generate this instance for reproducibility
    """

    v: np.ndarray
    k: int
    d: float
    tau: float
    vector_size: int
    seed: int


@dataclass
class ProjectionResults:
    """
    Results from benchmarking a solver on multiple problem instances.

    Stores solve times and success information for a specific solver on
    multiple instances of a problem with fixed size (m) and hardness (tau).

    Args:
        solver: Name of the solver used
        m: Number of scenarios in the tested problems
        tau: Hardness parameter used in problem generation
        times: List of solve times (np.nan indicates failed solve)
        status: List of solver status strings for each attempt
    """

    solver: str
    m: int
    tau: float
    times: list[float]
    status: list[str]

    @property
    def success_rate(self) -> float:
        """Return fraction of successful solves."""
        return np.sum(~np.isnan(self.times)) / len(self.times)

    @property
    def avg_time(self) -> float:
        """Average time of successful solves."""
        return np.nanmean(self.times)

    @property
    def std_time(self) -> float:
        """Standard deviation of successful solve times."""
        return np.nanstd(self.times)

    @property
    def num_success(self) -> int:
        """Return the total number of successful solves."""
        return int(np.sum(~np.isnan(self.times)))

    @property
    def num_total(self) -> int:
        """Return the total number of solve attempts."""
        return len(self.times)


class ProjectionBenchmark:
    """
    Runner class for sum-k-largest projection benchmarks.
    
    This class manages benchmarking of different solvers on the sum-k-largest
    projection problem across different problem sizes and difficulty settings.
    For each configuration, multiple random instances are generated and solved,
    with timing and success rate statistics collected.

    Args:
        vector_sizes: List of vector sizes (dimensions) to test
        tau_list: List of hardness parameters to test (0 < tau < 1)
        n_instances: Number of random instances per configuration
        solvers: List of solvers to benchmark
        n_consecutive_failures: Number of consecutive failures before stopping (None to run all)
        base_seed: Base random seed for reproducibility
    """

    def __init__(
        self,
        vector_sizes: list[int],
        tau_list: list[float],
        n_instances: int = 50,
        solvers: list[str] = ["Ours", "MOSEK", "CLARABEL"],
        n_consecutive_failures: int | None = None,
        base_seed: int = 42,
    ):
        # Validate inputs
        if not vector_sizes:
            raise ValueError("vector_sizes list cannot be empty")
        if not tau_list:
            raise ValueError("tau_list cannot be empty")
        if any(tau <= 0 or tau >= 1 for tau in tau_list):
            raise ValueError("All tau values must be between 0 and 1")
        if n_instances <= 0:
            raise ValueError("n_instances must be positive")
            
        self.vector_sizes = vector_sizes
        self.tau_list = tau_list
        self.n_instances = n_instances
        self.solvers = solvers
        self.n_consecutive_failures = n_consecutive_failures
        self.base_seed = base_seed
        self.results = {size: [] for size in vector_sizes}
        self.failed_solvers = set()

    def solve_instance_cvxpy(
        self, instance: Projection, solver: str
    ) -> tuple[float, str]:
        """
        Solve a projection instance using a CVXPY-supported solver.

        Args:
            instance: Problem instance to solve
            solver: Name of CVXPY-supported solver to use

        Returns:
            Tuple of (solve_time, status). solve_time is np.nan if solve failed
        """
        try:
            x = cp.Variable(instance.v.shape)
            objective = cp.Minimize(cp.sum_squares(x - instance.v))
            constraints = [cp.sum_largest(x, instance.k) <= instance.d]
            prob = cp.Problem(objective, constraints)

            solver_opts = SOLVER_CONFIGS.get(solver, {})
            prob.solve(
                solver=solver,
                verbose=False,
                **(solver_opts or {}),
                canon_backend=cp.SCIPY_CANON_BACKEND,
            )

            if prob.status in ["optimal", "optimal_inaccurate"]:
                return prob._solve_time, prob.status
            
            logging.error(f"Solver {solver} status: {prob.status}")
            return np.nan, prob.status

        except cp.SolverError as e:
            logging.error(f"Solver {solver} error: {str(e)}")
            return np.nan, "solver_error"
        except Exception as e:
            logging.error(f"Unexpected error: {str(e)}")
            return np.nan, "failed"

    def solve_instance_proj(self, instance: Projection) -> tuple[float, str]:
        """
        Solve a projection instance using our custom algorithm.

        Args:
            instance: Problem instance to solve

        Returns:
            Tuple of (solve_time, status). solve_time is np.nan if solve failed
        """
        try:
            start_time = time.time()
            _ = proj_sum_largest(instance.v, int(instance.k), instance.d)
            solve_time = time.time() - start_time
            return solve_time, "optimal"
        except ValueError as e:
            logging.error(f"Input validation error: {str(e)}")
            return np.nan, "invalid_input"
        except Exception as e:
            logging.error(f"Unexpected error: {str(e)}")
            return np.nan, "failed"

    def generate_instance(self, vector_size: int, tau: float, seed: int) -> Projection:
        """
        Generate a random sum-k-largest projection problem instance.

        Args:
            vector_size: Vector length
            tau: Hardness parameter in (0,1)
            seed: Random seed for reproducibility

        Returns:
            Generated problem instance with specified parameters
        """
        rng = np.random.RandomState(seed)
        beta = 0.95  # CVaR level for determining k

        # Number of largest elements to bound
        k = int(np.ceil((1 - beta) * vector_size))
        
        # Random vector to project
        v = rng.uniform(0, 1, vector_size)
        
        # Bound value (tau controls how tight the constraint is)
        d = tau * cp.sum_largest(v, k).value

        return Projection(v=v, k=k, d=d, tau=tau, vector_size=vector_size, seed=seed)

    def get_reproducible_seed(self, vector_size: int, tau: float, instance: int) -> int:
        """
        Generate a reproducible seed for a specific problem configuration.

        Args:
            vector_size: Vector length
            tau: Hardness parameter
            instance: Instance number

        Returns:
            Deterministic seed value within valid range
        """
        param_str = f"{vector_size}_{tau}_{instance}"
        return abs(hash(param_str)) % (2**32 - 1)

    @staticmethod
    def format_time_s(t: float) -> str:
        """Format time in scientific notation."""
        return f"{t:.2e}s"

    def save_results(self):
        """Save results to a pickle file for later analysis."""
        results_dict = {
            "vector_sizes": self.vector_sizes,
            "tau_list": self.tau_list,
            "n_instances": self.n_instances,
            "results": self.results,
        }
        # Use data directory within examples folder
        data_dir = Path(__file__).parent / "data"
        filename = data_dir / "proj.pkl"
        with open(filename, "wb") as f:
            pickle.dump(results_dict, f)

    def run_experiments(self):
        """Run all experiments and store results."""
        # Create examples/data directory instead of data at root level
        data_dir = Path(__file__).parent / "data"
        print(f"Data directory path: {data_dir.absolute()}")
        data_dir.mkdir(exist_ok=True)
        print(f"Data directory exists: {data_dir.exists()}")

        logging.info("Running sum-k-largest projection benchmarks")
        logging.info(f"Testing vector sizes: {self.vector_sizes}")
        logging.info(f"Testing τ values: {self.tau_list}")
        logging.info(f"Testing solvers: {self.solvers}")
        logging.info(f"Running {self.n_instances} instances per configuration")
        if self.n_consecutive_failures:
            logging.info(
                f"Will stop after {self.n_consecutive_failures} consecutive failures"
            )

        for vector_size in self.vector_sizes:
            logging.info(f"Benchmarking vectors with {vector_size:.0e} elements")

            for tau in self.tau_list:
                logging.info(f"  τ={tau}:")
                size_results = {}

                for solver in self.solvers:
                    # Skip if solver has completely failed in previous scenario
                    if solver in self.failed_solvers:
                        logging.info(
                            f"    {solver:8s}: skipped (failed in previous scenario)"
                        )
                        continue

                    times = []
                    statuses = []
                    consecutive_failures = 0

                    for i in range(self.n_instances):
                        seed = self.get_reproducible_seed(vector_size, tau, i)
                        instance = self.generate_instance(vector_size, tau, seed)

                        if solver.upper() == "OURS":
                            solve_time, status = self.solve_instance_proj(instance)
                        else:
                            solve_time, status = self.solve_instance_cvxpy(
                                instance, solver
                            )

                        times.append(solve_time)
                        statuses.append(status)

                        if np.isnan(solve_time):
                            consecutive_failures += 1
                        else:
                            consecutive_failures = 0

                        if (
                            self.n_consecutive_failures is not None
                            and consecutive_failures >= self.n_consecutive_failures
                        ):
                            logging.info(
                                f"    {solver:8s}: stopping after {consecutive_failures} "
                                "consecutive failures"
                            )
                            break

                    result = ProjectionResults(
                        solver=solver, m=vector_size, tau=tau, times=times, status=statuses
                    )

                    # Check if all instances failed for this solver
                    if result.num_success == 0:
                        logging.info(
                            f"    {solver:8s}: all {result.num_total} attempts failed"
                        )
                        # Add solver to failed set so it's skipped for remaining scenarios
                        self.failed_solvers.add(solver)
                    else:
                        logging.info(
                            f"    {solver:8s}: {self.format_time_s(result.avg_time):>10s} ± "
                            f"{self.format_time_s(result.std_time):>9s} "
                            f"[{result.num_success}/{result.num_total} succeeded]"
                        )

                    self.results[vector_size].append(result)
                    size_results[solver] = result
                    # Save results after each solver completes
                    self.save_results()

                # Calculate and display speedups across solvers for the current problem size
                if "Ours" in size_results and size_results["Ours"].num_success > 0:
                    our_time = size_results["Ours"].avg_time
                    speedups = []
                    for solver, result in size_results.items():
                        if (
                            solver != "Ours"
                            and result.num_success > 0
                            and result.avg_time > 0
                        ):
                            speedup = int(round(result.avg_time / our_time))
                            speedups.append(f"{solver}: {speedup}x")
                    if speedups:
                        logging.info(f"    {'Speedup':<8s}: {', '.join(speedups)}")

        logging.info("All experiments completed!")


def main():
    """Run projection benchmark experiments."""

    runner = ProjectionBenchmark(
        vector_sizes=[int(x) for x in [1e4, 3e4, 1e5, 3e5, 1e6, 3e6, 1e7]],
        tau_list=[DEFAULT_TAU],
        n_instances=5,             
        solvers=["Ours", "MOSEK", "CLARABEL"],
        n_consecutive_failures=None,
    )
    runner.run_experiments()


if __name__ == "__main__":
    main()
