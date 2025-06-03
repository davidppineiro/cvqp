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
    level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%b %d %H:%M:%S"
)
warnings.filterwarnings("ignore", module="cvxpy")

MAX_TIME_LIMIT = 7200
DEFAULT_TAU = 0.5
SOLVER_CONFIGS = {
    "MOSEK": {"mosek_params": {"MSK_DPAR_OPTIMIZER_MAX_TIME": MAX_TIME_LIMIT}},
    "CLARABEL": {"time_limit": MAX_TIME_LIMIT},
}


@dataclass
class Projection:
    """Single instance of a sum-k-largest projection problem."""
    v: np.ndarray
    k: int
    d: float
    tau: float
    vector_size: int
    seed: int


@dataclass
class ProjectionResults:
    """Results from benchmarking a solver on multiple problem instances."""
    solver: str
    m: int
    tau: float
    times: list[float]
    status: list[str]

    @property
    def success_rate(self) -> float:
        return np.sum(~np.isnan(self.times)) / len(self.times)

    @property
    def avg_time(self) -> float:
        return np.nanmean(self.times)

    @property
    def std_time(self) -> float:
        return np.nanstd(self.times)

    @property
    def num_success(self) -> int:
        return int(np.sum(~np.isnan(self.times)))

    @property
    def num_total(self) -> int:
        return len(self.times)
    

def solve_instance_cvxpy(instance: Projection, solver: str) -> tuple[float, str]:
    """Solve a projection instance using a CVXPY-supported solver."""
    try:
        x = cp.Variable(instance.v.shape)
        objective = cp.Minimize(cp.sum_squares(x - instance.v))
        constraints = [cp.sum_largest(x, instance.k) <= instance.d]
        prob = cp.Problem(objective, constraints)

        solver_opts = SOLVER_CONFIGS.get(solver, {})
        prob.solve(
            solver=solver,
            verbose=False,
            **solver_opts,
            canon_backend=cp.SCIPY_CANON_BACKEND,
        )

        if prob.status in ["optimal", "optimal_inaccurate"]:
            return prob._solve_time, prob.status
        
        logging.debug(f"Solver {solver} status: {prob.status}")
        return np.nan, prob.status

    except Exception as e:
        logging.debug(f"Solver {solver} error: {str(e)}")
        return np.nan, "error"


def solve_instance_proj(instance: Projection) -> tuple[float, str]:
    """Solve a projection instance using our custom algorithm."""
    try:
        start_time = time.time()
        _ = proj_sum_largest(instance.v, int(instance.k), instance.d)
        solve_time = time.time() - start_time
        return solve_time, "optimal"
    except Exception as e:
        logging.debug(f"Projection solver error: {str(e)}")
        return np.nan, "error"


def solve_instance(instance: Projection, solver: str) -> tuple[float, str]:
    """Solve a single projection instance."""
    if solver.upper() == "OURS":
        return solve_instance_proj(instance)
    else:
        return solve_instance_cvxpy(instance, solver)


def generate_instance(vector_size: int, tau: float, seed: int) -> Projection:
    """Generate a random sum-k-largest projection problem instance."""
    rng = np.random.RandomState(seed)
    beta = 0.95  # CVaR level for determining k

    # Number of largest elements to bound
    k = int(np.ceil((1 - beta) * vector_size))
    
    # Random vector to project
    v = rng.uniform(0, 1, vector_size)
    
    # Bound value (tau controls how tight the constraint is)
    d = tau * cp.sum_largest(v, k).value

    return Projection(v=v, k=k, d=d, tau=tau, vector_size=vector_size, seed=seed)


def get_reproducible_seed(vector_size: int, tau: float, instance: int, base_seed: int = 42) -> int:
    """Generate a reproducible seed for a specific problem configuration."""
    param_str = f"{vector_size}_{tau}_{instance}"
    return base_seed + abs(hash(param_str)) % (2**32 - 1)


class ProjectionBenchmark:
    """Runner class for sum-k-largest projection benchmarks."""

    def __init__(
        self,
        vector_sizes: list[int],
        tau_list: list[float],
        n_instances: int = 50,
        solvers: list[str] = ["Ours", "MOSEK", "CLARABEL"],
        base_seed: int = 42,
    ):
        self.vector_sizes = vector_sizes
        self.tau_list = tau_list
        self.n_instances = n_instances
        self.solvers = solvers
        self.base_seed = base_seed
        self.results = {size: [] for size in vector_sizes}
        self.failed_solvers = set()
        
        # Progress tracking
        self.total_combinations = len(vector_sizes) * len(tau_list)
        self.current_combination = 0
        self.total_benchmarks = 0
        self.successful_benchmarks = 0

    def run_experiments(self):
        """Run all experiments and store results."""
        self._setup_results_dir()
        self._log_experiment_info()
        
        for vector_size in self.vector_sizes:
            logging.info(f"Vector size: {vector_size}")
            self._run_size_experiments(vector_size)
            
        self._log_final_summary()

    def _setup_results_dir(self):
        """Create results directory if it doesn't exist."""
        results_dir = Path(__file__).parent / "results"
        results_dir.mkdir(exist_ok=True)

    def _log_experiment_info(self):
        """Log experiment setup information."""
        logging.info("=" * 60)
        logging.info("PROJECTION BENCHMARK SUITE")
        logging.info("=" * 60)
        logging.info(f"Solvers: {', '.join(self.solvers)}")
        logging.info(f"Vector sizes: {', '.join(map(str, self.vector_sizes))}")
        logging.info(f"Tau values: {', '.join(map(str, self.tau_list))}")
        logging.info(f"Instances per combination: {self.n_instances}")
        logging.info(f"Parameter combinations: {self.total_combinations}")
        logging.info("-" * 60)

    def _run_size_experiments(self, vector_size: int):
        """Run experiments for a single vector size."""
        for tau in self.tau_list:
            self.current_combination += 1
            
            # Progress and combination info
            logging.info(f"  [{self.current_combination}/{self.total_combinations}] "
                       f"vector_size={vector_size:.0e}, tau={tau}")
            
            # Run all solvers for this combination
            size_results = {}
            for solver in self.solvers:
                if solver in self.failed_solvers:
                    logging.info(f"    {solver:<8s}: SKIPPED (previous failures)")
                    continue
                
                result = self._benchmark_solver(vector_size, tau, solver)
                if result:
                    size_results[solver] = result
                    self.results[vector_size].append(result)
                    self.successful_benchmarks += 1
                
                self.total_benchmarks += 1
            
            if size_results:
                self._log_speedups(size_results)
            
            # Save results after each combination
            self._save_results()

    def _benchmark_solver(self, vector_size: int, tau: float, solver: str) -> ProjectionResults | None:
        """Benchmark a single solver on a problem combination."""
        times = []
        statuses = []
        
        for i in range(self.n_instances):
            # Generate problem instance
            seed = get_reproducible_seed(vector_size, tau, i, self.base_seed)
            instance = generate_instance(vector_size, tau, seed)
            
            # Solve instance
            solve_time, status = solve_instance(instance, solver)
            
            times.append(solve_time)
            statuses.append(status)

        # Create benchmark result
        result = ProjectionResults(
            solver=solver,
            m=vector_size,
            tau=tau,
            times=times,
            status=statuses,
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

    def _log_speedups(self, size_results: dict[str, ProjectionResults]):
        """Calculate and log speedups relative to Ours."""
        if "Ours" not in size_results:
            return
        
        our_time = size_results["Ours"].avg_time
        speedups = []
        
        for solver, result in size_results.items():
            if solver != "Ours" and result.avg_time > 0:
                speedup = result.avg_time / our_time
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

    def _save_results(self):
        """Save results to a pickle file for later analysis."""
        results_dict = {
            "vector_sizes": self.vector_sizes,
            "tau_list": self.tau_list,
            "n_instances": self.n_instances,
            "results": self.results,
        }
        results_dir = Path(__file__).parent / "results"
        filename = results_dir / "proj.pkl"
        with open(filename, "wb") as f:
            pickle.dump(results_dict, f)


def main():
    """Run projection benchmark experiments."""
    runner = ProjectionBenchmark(
        vector_sizes=[int(x) for x in [1e4, 3e4, 1e5, 3e5, 1e6, 3e6, 1e7]],
        tau_list=[DEFAULT_TAU],
        n_instances=5,             
        solvers=["Ours", "MOSEK", "CLARABEL"],
    )
    runner.run_experiments()


if __name__ == "__main__":
    main()