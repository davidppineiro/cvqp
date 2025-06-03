# CVQP Examples and Benchmarks

This directory contains examples and benchmarks that reproduce the results from our paper "[An Operator Splitting Method for Large-Scale CVaR-Constrained Quadratic Programs](https://web.stanford.edu/~boyd/papers/cvar_qp.html)".

## Benchmark Results

We compare our methods against state-of-the-art solvers: MOSEK (commercial) and CLARABEL (open-source) on two tasks:

1. Our specialized CVaR projection algorithm on random problem instances
2. Our ADMM-based method on large-scale CVQPs from portfolio optimization and quantile regression

## Running the Benchmarks

To reproduce the benchmark results from our paper:

```bash
# Run projection benchmarks
python benchmark_proj.py

# Run CVQP solver benchmarks
python benchmark_cvqp.py
```

Results are saved to the `data/` directory and can be visualized using the `benchmarks.ipynb` notebook:

```bash
# Open the visualization notebook
jupyter notebook benchmarks.ipynb
```

> **Note**: The benchmarks compare against MOSEK and CLARABEL. MOSEK requires a license (free for academic use), while CLARABEL is open-source.

## Utilities

The `plot_util.py` file contains visualization utilities for generating benchmark plots.