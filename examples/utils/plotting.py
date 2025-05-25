"""
Utility functions for plotting benchmark results.
"""

import pickle
import matplotlib.pyplot as plt
from pathlib import Path

# Import benchmark result types
try:
    from benchmark_proj import ProjectionResults
    from benchmark_cvqp import BenchmarkResults
except ImportError:
    try:
        from examples.benchmark_proj import ProjectionResults
        from examples.benchmark_cvqp import BenchmarkResults
    except ImportError:
        pass  # Allow import in package context


def setup_plotting_style():
    """Configure matplotlib plotting style."""
    plt.rcParams.update({
        "font.size": 14,
        "axes.labelsize": 14, 
        "axes.titlesize": 16,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
        "figure.figsize": (6, 4.5),
        "lines.linewidth": 2,
        "lines.markersize": 8,
        "axes.grid": True,
        "grid.alpha": 0.35,
        "grid.linestyle": ":",
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "backend": "ps",
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "text.latex.preamble": (r"\usepackage{gensymb} " r"\usepackage{amsmath}"),
    })


def load_proj_results(filename="proj.pkl", data_dir=None):
    """Load projection benchmark results from pickle file."""
    if data_dir is None:
        data_dir = Path(__file__).parent.parent / "data"
    else:
        data_dir = Path(data_dir)
        
    file_path = data_dir / filename
    if not file_path.exists():
        print(f"File not found: {file_path}")
        return []
        
    try:
        with open(file_path, "rb") as f:
            data = pickle.load(f)
            # Flatten the dict of lists into a single list
            all_results = []
            for m_results in data["results"].values():
                all_results.extend(m_results)
        return all_results
    except Exception as e:
        print(f"Error loading projection results: {e}")
        return []


def load_cvqp_results(problem_name, data_dir=None):
    """Load CVQP benchmark results for a specific problem."""
    if data_dir is None:
        data_dir = Path(__file__).parent.parent / "data"
    else:
        data_dir = Path(data_dir)
        
    data_path = data_dir / f"{problem_name.lower()}.pkl"
    if not data_path.exists():
        print(f"File not found: {data_path}")
        return []

    try:
        with open(data_path, "rb") as f:
            data = pickle.load(f)
        return data["results"]
    except Exception as e:
        print(f"Error loading {problem_name} results: {e}")
        return []


def plot_proj_benchmarks(results, save_figures=False):
    """Create plots comparing solver performance for projection benchmarks."""
    if not results:
        print("No benchmark results found.")
        return {}
        
    markers = {"Ours": "o", "MOSEK": "s", "CLARABEL": "^"}
    figs_dir = Path(__file__).parent.parent / "figs"
    if save_figures:
        figs_dir.mkdir(exist_ok=True)

    figures = {}
    tau_values = sorted(set(r.tau for r in results))

    for tau in tau_values:
        setup_plotting_style()
        fig, ax = plt.subplots()

        # Filter results for this tau
        tau_results = [r for r in results if r.tau == tau]
        solvers = sorted(set(r.solver for r in tau_results))

        # Plot solution times
        for solver in solvers:
            solver_results = sorted([r for r in tau_results if r.solver == solver], 
                                  key=lambda x: x.m)
            
            x_values = [r.m for r in solver_results]
            times = [r.avg_time for r in solver_results]

            ax.plot(x_values, times,
                   label=solver if solver == "Ours" else solver.upper(),
                   marker=markers[solver], markersize=8, linestyle="-", 
                   markeredgewidth=1)

        # Configure plot
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Number of scenarios")
        ax.set_ylabel("Solve time (seconds)")
        ax.grid(True, which="major", linestyle="-", alpha=0.35)
        ax.grid(True, which="minor", linestyle=":", alpha=0.2)
        ax.legend()

        plt.tight_layout()
        figures[tau] = fig

        if save_figures:
            fig.savefig(figs_dir / "proj.pdf")
        plt.show()

    return figures


def plot_cvqp_benchmarks(results, save_figures=False):
    """Create plots comparing solver performance for CVQP benchmarks."""
    if not results:
        print("No benchmark results found.")
        return {}
        
    markers = {"CVQP": "o", "MOSEK": "s", "CLARABEL": "^"}
    problem_name = results[0].problem.lower()
    
    figs_dir = Path(__file__).parent.parent / "figs"
    if save_figures:
        figs_dir.mkdir(exist_ok=True)

    figures = {}
    n_vars_values = sorted(set(r.n_vars for r in results))

    for n_vars in n_vars_values:
        setup_plotting_style()
        fig, ax = plt.subplots()

        # Filter results for this n_vars
        nvar_results = [r for r in results if r.n_vars == n_vars]
        solvers = ["CLARABEL", "MOSEK", "CVQP"]  # Fixed order

        # Plot solution times
        for solver in solvers:
            solver_results = sorted([r for r in nvar_results if r.solver == solver],
                                  key=lambda x: x.n_scenarios)
            
            x_values = [r.n_scenarios for r in solver_results]
            times = [r.avg_time for r in solver_results]

            ax.plot(x_values, times,
                   label=solver if solver == "Ours" else solver.upper(),
                   marker=markers[solver], markersize=8, linestyle="-",
                   markeredgewidth=1)

        # Configure plot
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Number of scenarios")
        ax.set_ylabel("Solve time (seconds)")
        ax.grid(True, which="major", linestyle="-", alpha=0.35)
        ax.grid(True, which="minor", linestyle=":", alpha=0.2)
        ax.legend()

        plt.tight_layout()
        figures[n_vars] = fig

        if save_figures:
            fig.savefig(figs_dir / f"{problem_name}.pdf")
        plt.show()

    return figures