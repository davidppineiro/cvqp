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
from problems import *

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%b %d %H:%M:%S"
)
warnings.filterwarnings("ignore", module="cvxpy")

MAX_TIME_LIMIT = 7200
SOLVER_CONFIGS = {
    "MOSEK": {"mosek_params": {"MSK_DPAR_OPTIMIZER_MAX_TIME": MAX_TIME_LIMIT}},
    "CLARABEL": {"time_limit": MAX_TIME_LIMIT},
}


@dataclass
class BenchmarkResults:
    """Store benchmark results for a specific problem size and solver."""
    problem: str
    solver: str
    n_vars: int
    n_scenarios: int
    times: list[float]
    status: list[str]
    cvqp_results: list[CVQPResults | None] = None

    @property
    def success_rate(self) -> float:
        return np.sum(~np.isnan(self.times)) / len(self.times)

    @property
    def avg_time(self) -> float | None:
        return np.nanmean(self.times)

    @property
    def std_time(self) -> float | None:
        return np.nanstd(self.times)

    @property
    def num_success(self) -> int:
        return int(np.sum(~np.isnan(self.times)))

    @property
    def num_total(self) -> int:
        return len(self.times)
    

def solve_cvqp(params: CVQPParams) -> tuple[float | None, str, CVQPResults | None]:
    """Solve using CVQP solver."""
    try:
        solver = CVQP(params, CVQPConfig())
        results = solver.solve()
        return results.solve_time, results.problem_status, results
    except Exception as e:
        # Use debug level to reduce clutter
        logging.debug(f"CVQP solver error: {str(e)}")
        return np.nan, "error", None


def solve_cvxpy(prob: cp.Problem, solver: str) -> tuple[float | None, str]:
    """Solve using CVXPY with specified solver."""
    solver_map = {"MOSEK": cp.MOSEK, "CLARABEL": cp.CLARABEL}
    
    try:
        solver_opts = SOLVER_CONFIGS.get(solver, {})
        prob.solve(
            solver=solver_map[solver],
            verbose=False,
            **solver_opts,
            canon_backend=cp.SCIPY_CANON_BACKEND,
        )
        
        if prob.status == "optimal":
            return prob._solve_time, prob.status
        else:
            logging.debug(f"Solver {solver} status: {prob.status}")
            return np.nan, prob.status
            
    except Exception as e:
        logging.debug(f"Solver {solver} error: {str(e)}")
        return np.nan, "error"


def solve_instance(params: tuple[CVQPParams, cp.Problem], solver: str):
    """Solve a single problem instance."""
    if solver == "CVQP":
        return solve_cvqp(params[0])
    else:
        return solve_cvxpy(params[1], solver)


def get_instance_seed(problem_name: str, n_vars: int, n_scenarios: int, instance_idx: int, base_seed: int = 42) -> int:
    """Generate reproducible seed for a problem instance."""
    instance_str = f"{problem_name}_{n_vars}_{n_scenarios}_{instance_idx}"
    return base_seed + hash(instance_str) % (2**32)


class CVQPBenchmark:
    """Runner class for CVQP benchmark experiments."""

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
        self.problems = problems
        self.n_instances = n_instances
        self.n_vars_list = n_vars_list
        self.n_scenarios_list = n_scenarios_list
        self.solvers = solvers
        self.base_seed = base_seed
        self.n_consecutive_failures = n_consecutive_failures
        self.results = {p.name: [] for p in problems}
        self.failed_solvers = set()
        
        # Progress tracking
        self.total_combinations = len(problems) * len(n_vars_list) * len(n_scenarios_list)
        self.current_combination = 0
        self.total_benchmarks = 0
        self.successful_benchmarks = 0

    def run_experiments(self):
        """Run all experiments and store results."""
        self._setup_results_dir()
        self._log_experiment_info()
        
        for problem in self.problems:
            logging.info(f"Problem: {problem.name}")
            self._run_problem_experiments(problem)
            self._save_problem_results(problem.name)
        
        self._log_final_summary()

    def _setup_results_dir(self):
        """Create results directory if it doesn't exist."""
        results_dir = Path(__file__).parent / "results"
        results_dir.mkdir(exist_ok=True)

    def _log_experiment_info(self):
        """Log experiment setup information."""
        logging.info("=" * 60)
        logging.info("CVQP BENCHMARK SUITE")
        logging.info("=" * 60)
        logging.info(f"Problems: {', '.join(p.name for p in self.problems)}")
        logging.info(f"Solvers: {', '.join(self.solvers)}")
        logging.info(f"Number of variables (n_vars): {', '.join(map(str, self.n_vars_list))}")
        logging.info(f"Number of scenarios (n_scenarios): {', '.join(map(str, self.n_scenarios_list))}")
        logging.info(f"Instances per combination: {self.n_instances}")
        logging.info(f"Parameter combinations: {self.total_combinations}")
        logging.info("-" * 60)

    def _run_problem_experiments(self, problem: CVQPProblem):
        """Run experiments for a single problem type."""
        for n_vars in self.n_vars_list:
            for n_scenarios in self.n_scenarios_list:
                self.current_combination += 1
                
                # Progress and combination info
                logging.info(f"  [{self.current_combination}/{self.total_combinations}] "
                           f"n_vars={n_vars:.0e}, n_scenarios={n_scenarios:.0e}")
                
                # Run all solvers for this problem size
                size_results = {}
                for solver in self.solvers:
                    if solver in self.failed_solvers:
                        logging.info(f"    {solver:<8s}: SKIPPED (previous failures)")
                        continue
                    
                    result = self._benchmark_solver(problem, solver, n_vars, n_scenarios)
                    if result:
                        size_results[solver] = result
                        self.results[problem.name].append(result)
                        self.successful_benchmarks += 1
                    
                    self.total_benchmarks += 1
                
                if size_results:
                    self._log_speedups(size_results)

    def _benchmark_solver(self, problem: CVQPProblem, solver: str, n_vars: int, n_scenarios: int) -> BenchmarkResults | None:
        """Benchmark a single solver on a problem size."""
        times = []
        statuses = []
        cvqp_results = [] if solver == "CVQP" else None
        
        for i in range(self.n_instances):
            # Generate problem instance
            seed = get_instance_seed(problem.name, n_vars, n_scenarios, i, self.base_seed)
            params = problem.generate_instance(n_vars, n_scenarios, seed=seed)
            
            # Solve instance
            result = solve_instance(params, solver)
            
            if solver == "CVQP":
                solve_time, status, cvqp_result = result
                cvqp_results.append(cvqp_result)
            else:
                solve_time, status = result
            
            times.append(solve_time)
            statuses.append(status)
            
            # Simplified consecutive failure check
            if (self.n_consecutive_failures and 
                len(times) >= self.n_consecutive_failures and
                all(np.isnan(t) for t in times[-self.n_consecutive_failures:])):
                logging.warning(f"    {solver:<8s}: stopping after {self.n_consecutive_failures} consecutive failures")
                self.failed_solvers.add(solver)
                break

        # Create benchmark result
        result = BenchmarkResults(
            problem=problem.name,
            solver=solver,
            n_vars=n_vars,
            n_scenarios=n_scenarios,
            times=times,
            status=statuses,
            cvqp_results=cvqp_results,
        )

        # Log results
        if result.num_success > 0:
            logging.info(
                f"    {solver:<8s}: {result.avg_time:.2e}s ± {result.std_time:.2e}s "
                f"({result.num_success}/{result.num_total} OK)"
            )
            return result
        else:
            logging.error(f"    {solver:<8s}: FAILED (all {result.num_total} attempts)")
            self.failed_solvers.add(solver)
            return None

    def _log_speedups(self, size_results: dict[str, BenchmarkResults]):
        """Calculate and log speedups relative to CVQP."""
        if "CVQP" not in size_results:
            return
        
        cvqp_time = size_results["CVQP"].avg_time
        speedups = []
        
        for solver, result in size_results.items():
            if solver != "CVQP" and result.avg_time > 0:
                speedup = result.avg_time / cvqp_time
                speedups.append(f"{solver}: {speedup:.1f}x")
        
        if speedups:
            logging.info(f"    {'Speedup':<8s}: {', '.join(speedups)}")
    
    def _log_final_summary(self):
        """Log final benchmark summary."""
        logging.info("-" * 60)
        logging.info("BENCHMARK SUMMARY")
        logging.info("-" * 60)
        logging.info(f"Total benchmarks: {self.total_benchmarks}")
        logging.info(f"Successful: {self.successful_benchmarks}")
        logging.info(f"Failed: {self.total_benchmarks - self.successful_benchmarks}")
        logging.info(f"Success rate: {self.successful_benchmarks/self.total_benchmarks*100:.1f}%")
        
        if self.failed_solvers:
            logging.info(f"Failed solvers: {sorted(self.failed_solvers)}")
        else:
            logging.info("All solvers completed successfully!")
        
        logging.info("=" * 60)

    def _save_problem_results(self, problem_name: str):
        """Save results for a problem to pickle file."""
        results_dict = {
            "base_seed": self.base_seed,
            "n_instances": self.n_instances,
            "n_vars_list": self.n_vars_list,
            "n_scenarios_list": self.n_scenarios_list,
            "results": self.results[problem_name],
        }
        
        results_dir = Path(__file__).parent / "results"
        filename = results_dir / f"{problem_name.lower()}.pkl"
        with open(filename, "wb") as f:
            pickle.dump(results_dict, f)


def main():
    """Run CVQP benchmark experiments."""
    portfolio_runner = CVQPBenchmark(
        problems=[PortfolioOptimization()],
        n_instances=3,
        n_vars_list=[2000],
        n_scenarios_list=[int(x) for x in [1e3, 3e3, 1e4, 3e4, 1e5, 3e5, 1e6]],
        solvers=["CVQP", "MOSEK", "CLARABEL"],       
        n_consecutive_failures=1,
    )
    portfolio_runner.run_experiments()

    qr_runner = CVQPBenchmark(
        problems=[QuantileRegression()],
        n_instances=3,
        n_vars_list=[500],
        n_scenarios_list=[int(x) for x in [1e4, 3e4, 1e5, 3e5, 1e6, 3e6, 1e7]],
        solvers=["CVQP", "MOSEK", "CLARABEL"],
        n_consecutive_failures=1,
    )
    qr_runner.run_experiments()


if __name__ == "__main__":
    main()