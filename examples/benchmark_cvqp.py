"""
Script to benchmark CVQP solver.
"""

from dataclasses import dataclass
import logging
import pickle
from pathlib import Path
import cvxpy as cp
import numpy as np
import warnings

from cvqp import CVQP, CVQPConfig, CVQPResults, CVQPParams
from examples.problems import *

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%b %d %H:%M:%S"
)

warnings.filterwarnings("ignore", module="cvxpy")

# Constants
MAX_TIME_LIMIT = 7200  # Maximum time limit in seconds

# Solver configurations
SOLVER_CONFIGS = {
    "MOSEK": {"mosek_params": {"MSK_DPAR_OPTIMIZER_MAX_TIME": MAX_TIME_LIMIT}},
    "CLARABEL": {"time_limit": MAX_TIME_LIMIT},
}


@dataclass
class BenchmarkResults:
    """
    Store benchmark results for a specific problem size and solver.

    Args:
        problem: Problem type name
        solver: Name of the solver used
        n_vars: Number of variables in the problem
        n_scenarios: Number of scenarios in the problem
        times: List of solve times for each random instance
        status: List of solver status for each random instance
        cvqp_results: List of detailed CVQP results, if used (None for other solvers)
    """

    problem: str
    solver: str
    n_vars: int
    n_scenarios: int
    times: list[float]
    status: list[str]
    cvqp_results: list[CVQPResults | None] = None

    @property
    def success_rate(self) -> float:
        """Return fraction of successful solves."""
        return np.sum(~np.isnan(self.times)) / len(self.times)

    @property
    def avg_time(self) -> float | None:
        """Average time of successful solves."""
        return np.nanmean(self.times)

    @property
    def std_time(self) -> float | None:
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


class CVQPBenchmark:
    """
    Runner class for CVQP benchmark experiments.
    
    This class manages benchmarking of different solvers on CVaR-constrained
    QP problems across different problem sizes and scenario counts. For each
    configuration, random problem instances are generated and solved, with
    timing and success rate statistics collected.

    Args:
        problems: List of problem instances to benchmark
        n_instances: Number of random instances to generate for each problem size
        n_vars_list: List of problem sizes (number of variables) to test
        n_scenarios_list: List of scenario counts to test
        solvers: List of solvers to benchmark
        base_seed: Base random seed for reproducibility
        n_consecutive_failures: Number of consecutive failures before stopping (None to run all)
    """

    def __init__(
        self,
        problems: list[CVQPProblem],
        n_instances: int,
        n_vars_list: list[int],
        n_scenarios_list: list[int],
        solvers: list[str],
        base_seed: int = 42,
        n_consecutive_failures: int | None = None,
    ):
        # Validate inputs
        if not problems:
            raise ValueError("problems list cannot be empty")
        if n_instances <= 0:
            raise ValueError("n_instances must be positive")
        if not n_vars_list:
            raise ValueError("n_vars_list cannot be empty")
        if not n_scenarios_list:
            raise ValueError("n_scenarios_list cannot be empty")
        if not solvers:
            raise ValueError("solvers list cannot be empty")
            
        self.problems = problems
        self.n_instances = n_instances
        self.n_vars_list = n_vars_list
        self.n_scenarios_list = n_scenarios_list
        self.solvers = solvers
        self.base_seed = base_seed
        self.n_consecutive_failures = n_consecutive_failures
        self.results = {p.name: [] for p in problems}

    def run_experiments(self):
        """Run all experiments and store results."""
        # Create examples/data directory instead of data at root level
        data_dir = Path(__file__).parent / "data"
        data_dir.mkdir(exist_ok=True)

        logging.info("Running CVQP benchmarks")
        logging.info(f"Testing n_vars values: {self.n_vars_list}")
        logging.info(f"Testing n_scenarios values: {self.n_scenarios_list}")
        logging.info(f"Testing solvers: {self.solvers}")
        logging.info(f"Testing {self.n_instances} random instances per problem size")

        for problem in self.problems:
            logging.info(f"Benchmarking problem: {problem.name}")

            for n_vars in self.n_vars_list:
                failed_solvers = set()
                logging.info(f"n_vars={n_vars:.0e}:")

                for n_scenarios in self.n_scenarios_list:
                    logging.info(f"  n_scenarios={n_scenarios:.0e}:")
                    size_results = {}
                    solver_done = {solver: False for solver in self.solvers}
                    solver_results = self.initialize_solver_results()

                    for i in range(self.n_instances):
                        seed = self.get_instance_seed(
                            problem.name, n_vars, n_scenarios, i
                        )
                        params = problem.generate_instance(
                            n_vars, n_scenarios, seed=seed
                        )

                        for solver in self.solvers:
                            if solver in failed_solvers:
                                if self.handle_solver_failure(
                                    solver, failed_solvers, i == 0
                                ):
                                    continue

                            if solver_done[solver]:
                                continue

                            # Run solver and collect results
                            if solver == "CVQP":
                                solve_time, status, result = self.solve_instance(
                                    params, solver
                                )
                                solver_results[solver]["cvqp_results"].append(result)
                            else:
                                solve_time, status = self.solve_instance(params, solver)

                            solver_results[solver]["times"].append(solve_time)
                            solver_results[solver]["statuses"].append(status)

                            # Check for consecutive failures
                            if np.isnan(solve_time):
                                if self.check_consecutive_failures(
                                    solver_results, solver, failed_solvers
                                ):
                                    solver_done[solver] = True
                                    break

                            # Process results if solver has completed all instances
                            if (
                                len(solver_results[solver]["times"]) == self.n_instances
                                and not solver_done[solver]
                            ):
                                result, failed = self.process_solver_results(
                                    solver,
                                    solver_results,
                                    problem.name,
                                    n_vars,
                                    n_scenarios,
                                )

                                if failed:
                                    failed_solvers.add(solver)
                                else:
                                    size_results[solver] = result

                                self.results[problem.name].append(result)
                                solver_done[solver] = True

                    self.save_problem_results(problem.name)
                    self.calculate_speedups(size_results)

            logging.info("All experiments completed!")

    def get_instance_seed(
        self,
        problem_name: str,
        n_vars: int,
        n_scenarios: int,
        instance_idx: int,
    ) -> int:
        """
        Generate reproducible seed for a specific problem instance.

        Args:
            problem_name: Name of the problem type
            n_vars: Number of variables
            n_scenarios: Number of scenarios
            instance_idx: Index of the instance

        Returns:
            Seed for random number generation
        """
        instance_str = f"{problem_name}_{n_vars}_{n_scenarios}_{instance_idx}"
        return self.base_seed + hash(instance_str) % (2**32)

    @staticmethod
    def format_time_s(t: float) -> str:
        """Format time in scientific notation."""
        return f"{t:.2e}s"

    def solve_instance(
        self, params: tuple[CVQPParams, cp.Problem], solver: str
    ) -> tuple[float | None, str] | tuple[float | None, str, CVQPResults | None]:
        """
        Solve a CVQP instance with specified solver.

        Args:
            params: Tuple of (CVQPParams, CVXPY Problem)
            solver: Name of solver to use

        Returns:
            Tuple of (solve_time, status) or (solve_time, status, results) for CVQP solver
        """
        if solver in ["MOSEK", "CLARABEL"]:
            return self.solve_cvxpy(params[1], solver)
        else:
            return self.solve_cvqp(params[0])

    def solve_cvxpy(
        self, prob: cp.Problem, solver: str, verbose: bool = False
    ) -> tuple[float | None, str]:
        """
        Solve using CVXPY with specified solver.

        Args:
            prob: CVXPY Problem instance
            solver: Name of solver to use
            verbose: Whether to print solver output

        Returns:
            Tuple of (solve_time, status)
        """
        solver_map = {"MOSEK": cp.MOSEK, "CLARABEL": cp.CLARABEL}
        if solver not in solver_map:
            raise ValueError(f"Unsupported solver: {solver}")

        solver_name = solver
        solver = solver_map[solver]
        solver_opts = SOLVER_CONFIGS.get(solver_name, {})

        try:
            prob.solve(
                solver=solver,
                verbose=verbose,
                **(solver_opts or {}),
                canon_backend=cp.SCIPY_CANON_BACKEND,
            )
            solve_time = prob._solve_time
            status = prob.status

            if status != "optimal":
                logging.warning(f"Solver {solver_name} status: {status}")
                return np.nan, status

            return solve_time, status

        except cp.SolverError as e:
            logging.warning(f"Solver {solver_name} error: {str(e)}")
            return np.nan, "solver_error"
        except Exception as e:
            logging.warning(f"Unexpected error: {str(e)}")
            return np.nan, "error"

    def solve_cvqp(
        self, params: CVQPParams
    ) -> tuple[float | None, str, CVQPResults | None]:
        """
        Solve using CVQP solver.

        Args:
            params: CVQP problem parameters

        Returns:
            Tuple of (solve_time, status, results)
        """
        try:
            solver = CVQP(params, CVQPConfig())
            results = solver.solve()
            return results.solve_time, results.problem_status, results
        except ValueError as e:
            logging.warning(f"CVQP solver validation error: {str(e)}")
            return np.nan, "invalid_input", None
        except Exception as e:
            logging.warning(f"CVQP solver error: {str(e)}")
            return np.nan, "error", None

    def initialize_solver_results(self) -> dict[str, dict[str, list]]:
        """
        Initialize data structures for tracking solver results.

        Returns:
            Dictionary mapping solver names to their results storage structure
        """
        return {
            solver: {
                "times": [],
                "statuses": [],
                "cvqp_results": [] if solver == "CVQP" else None,
            }
            for solver in self.solvers
        }

    def handle_solver_failure(
        self, solver: str, failed_solvers: set[str], is_first_instance: bool
    ) -> bool:
        """
        Handle logging for failed solvers.

        Args:
            solver: Name of the solver
            failed_solvers: Set of solvers that have failed
            is_first_instance: Whether this is the first instance for this solver

        Returns:
            True to indicate the solver should be skipped
        """
        if is_first_instance:
            logging.info(
                f"    {solver:<8s} : skipping due to failure at smaller n_scenarios"
            )
        return True

    def check_consecutive_failures(
        self, solver_results: dict[str, list], solver: str, failed_solvers: set[str]
    ) -> bool:
        """
        Check if solver has hit consecutive failure limit.

        Args:
            solver_results: Dictionary containing solver results
            solver: Name of the solver to check
            failed_solvers: Set to track failed solvers

        Returns:
            True if solver has hit consecutive failure limit, False otherwise
        """
        if self.n_consecutive_failures is None:
            return False

        times = solver_results[solver]["times"]
        if len(times) >= self.n_consecutive_failures:
            all_recent_failed = all(
                np.isnan(t) for t in times[-self.n_consecutive_failures :]
            )
            if all_recent_failed:
                logging.info(
                    f"    {solver:<8s} : stopping after "
                    f"{self.n_consecutive_failures} consecutive failures"
                )
                failed_solvers.add(solver)
                return True
        return False

    def process_solver_results(
        self,
        solver: str,
        solver_results: dict[str, dict[str, list]],
        problem_name: str,
        n_vars: int,
        n_scenarios: int,
    ) -> tuple[BenchmarkResults, bool]:
        """
        Process results for a solver and create BenchmarkResults object.

        Args:
            solver: Name of the solver
            solver_results: Dictionary containing all solver results
            problem_name: Name of the problem being solved
            n_vars: Number of variables in the problem
            n_scenarios: Number of scenarios in the problem

        Returns:
            Tuple of (benchmark_results, failed_flag) where failed_flag indicates if all attempts failed
        """
        result = BenchmarkResults(
            problem=problem_name,
            solver=solver,
            n_vars=n_vars,
            n_scenarios=n_scenarios,
            times=solver_results[solver]["times"],
            status=solver_results[solver]["statuses"],
            cvqp_results=solver_results[solver]["cvqp_results"],
        )

        if result.num_success > 0:
            logging.info(
                f"    {solver:<8s} : "
                f"{self.format_time_s(result.avg_time)} ± "
                f"{self.format_time_s(result.std_time)} "
                f"[{result.num_success}/{result.num_total} succeeded]"
            )
            return result, False
        else:
            logging.info(
                f"    {solver:<8s} : all {result.num_total} attempts failed"
                f", skipping larger n_scenarios values"
            )
            return result, True

    def calculate_speedups(self, size_results: dict[str, BenchmarkResults]) -> None:
        """
        Calculate and log speedups relative to CVQP.

        Args:
            size_results: Dictionary mapping solver names to their benchmark results
        """
        if "CVQP" not in size_results or size_results["CVQP"].num_success == 0:
            return

        cvqp_time = size_results["CVQP"].avg_time
        speedups = []
        for solver, result in size_results.items():
            if solver != "CVQP" and result.num_success > 0 and result.avg_time > 0:
                speedup = int(round(result.avg_time / cvqp_time))
                speedups.append(f"{solver}: {speedup}x")
        if speedups:
            logging.info(f"    {'Speedup':<8s} : {', '.join(speedups)}")

    def save_problem_results(self, problem_name: str):
        """
        Save results for a specific problem to its own pickle file.

        Args:
            problem_name: Name of the problem whose results should be saved
        """
        results_dict = {
            "base_seed": self.base_seed,
            "n_instances": self.n_instances,
            "n_vars_list": self.n_vars_list,
            "n_scenarios_list": self.n_scenarios_list,
            "results": self.results[problem_name],
        }
        # Use data directory within examples folder
        data_dir = Path(__file__).parent / "data"
        filename = data_dir / f"{problem_name.lower()}.pkl"
        with open(filename, "wb") as f:
            pickle.dump(results_dict, f)


def main():
    """Run CVQP benchmark experiments."""

    portfolio_runner = CVQPBenchmark(
        problems=[PortfolioOptimization()],
        n_instances=1,
        n_vars_list=[int(x) for x in [2e3]],
        n_scenarios_list=[int(x) for x in [1e3, 3e3, 1e4, 3e4, 1e5, 3e5, 1e6]],
        solvers=["CVQP", "MOSEK", "CLARABEL"],       
        n_consecutive_failures=1,
    )
    portfolio_runner.run_experiments()

    qr_runner = CVQPBenchmark(
        problems=[QuantileRegression()],
        n_instances=1,
        n_vars_list=[int(x) for x in [500]],
        n_scenarios_list=[int(x) for x in [1e4, 3e4, 1e5, 3e5, 1e6, 3e6, 1e7]],
        solvers=["CVQP", "MOSEK", "CLARABEL"],
        n_consecutive_failures=1,
    )
    qr_runner.run_experiments()


if __name__ == "__main__":
    main()
