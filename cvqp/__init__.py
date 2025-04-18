"""
CVQP: A Python solver for CVaR-constrained quadratic programs.
"""

__version__ = "0.1.0"

from .types import CVQPParams, CVQPConfig, CVQPResults
from .solver import CVQP
from .projection import proj_sum_largest

__all__ = ["CVQP", "CVQPParams", "CVQPResults", "CVQPConfig", "proj_sum_largest"]
