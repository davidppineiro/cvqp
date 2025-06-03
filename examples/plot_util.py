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
        "lines.markersize": 6,
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


def load_results(filename, data_dir=None, flatten=False):
    """Load benchmark results from pickle file."""
    if data_dir is None:
        data_dir = Path(__file__).parent / "results"
    else:
        data_dir = Path(data_dir)
        
    file_path = data_dir / filename
    
    with open(file_path, "rb") as f:
        data = pickle.load(f)
        results = data["results"]
        
        # Flatten dict of lists if needed (for projection results)
        if flatten:
            all_results = []
            for m_results in results.values():
                all_results.extend(m_results)
            return all_results
        
        return results


def plot_proj_benchmarks(results, save_figures=False):
    """Create plots comparing solver performance for projection benchmarks."""
    if not results:
        print("No benchmark results found.")
        return {}
        
    markers = {"Ours": "o", "MOSEK": "o", "CLARABEL": "o"}
    colors = {"Ours": "#2E86AB", "MOSEK": "#A23B72", "CLARABEL": "#F18F01"}
    figs_dir = Path(__file__).parent / "figs"
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
                   marker=markers[solver], color=colors[solver], linestyle="-", 
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
        
    markers = {"CVQP": "o", "MOSEK": "o", "CLARABEL": "o"}
    colors = {"CVQP": "#2E86AB", "MOSEK": "#A23B72", "CLARABEL": "#F18F01"}
    problem_name = results[0].problem.lower()
    
    figs_dir = Path(__file__).parent / "figs"
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
                   marker=markers[solver], color=colors[solver], linestyle="-",
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