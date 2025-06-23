"""
Projection onto sum-of-k-largest and CVaR constraints.
"""

import numpy as np
from .libs import proj_sum_largest_cpp


def proj_sum_largest(x: np.ndarray, k: int, alpha: float) -> np.ndarray:
    """Compute the Euclidean projection onto the sum-of-k-largest constraint.

    Given a vector x, finds the closest point (in Euclidean distance) that
    satisfies the constraint that the sum of its k largest elements is at most alpha.

    Args:
        x: Input vector to project.
        k: Number of largest elements to constrain (1 <= k <= len(x)).
        alpha: Upper bound on sum of k largest elements.

    Returns:
        Projected vector with same shape as x, satisfying the sum-of-k-largest constraint.

    Raises:
        TypeError: If x is not a numpy array.
        ValueError: If x is not 1D, k is out of bounds, or alpha is negative.
    """
    # Input validation
    if not isinstance(x, np.ndarray):
        raise TypeError(f"Input x must be a numpy array, got {type(x)}")

    if x.ndim != 1:
        raise ValueError(f"Input x must be a 1D array, got shape {x.shape}")

    if not 0 < k <= len(x):
        raise ValueError(f"k must be between 0 and len(x), got {k}")

    if alpha < 0:
        raise ValueError(f"alpha must be non-negative, got {alpha}")

    # Sort indices in descending order
    sorted_inds = np.argsort(x)[::-1]
    x_sorted = x[sorted_inds]

    # Early return if constraint is already satisfied
    if np.sum(x_sorted[:k]) <= alpha:
        return x.copy()

    # Call C++ implementation
    x_projected, *_ = proj_sum_largest_cpp(x_sorted, k, alpha, k, 0, len(x), False)

    # Restore original ordering
    result = np.empty_like(x)
    result[sorted_inds] = x_projected
    return result


def proj_cvar(x: np.ndarray, beta: float, kappa: float) -> np.ndarray:
    """Compute the Euclidean projection onto the CVaR constraint.

    Given a vector x, finds the closest point (in Euclidean distance) that
    satisfies the constraint CVaR_beta(x) <= kappa, where CVaR_beta is the
    Conditional Value at Risk at confidence level beta.

    Args:
        x: Input vector to project.
        beta: Confidence level for CVaR (0 < beta < 1).
        kappa: Upper bound on CVaR value.

    Returns:
        Projected vector with same shape as x, satisfying CVaR_beta(result) <= kappa.

    Raises:
        TypeError: If x is not a numpy array.
        ValueError: If x is not 1D, beta is out of bounds, or kappa is negative.
    """
    # Input validation
    if not isinstance(x, np.ndarray):
        raise TypeError(f"Input x must be a numpy array, got {type(x)}")

    if x.ndim != 1:
        raise ValueError(f"Input x must be a 1D array, got shape {x.shape}")

    if not 0 < beta < 1:
        raise ValueError(f"beta must be between 0 and 1 (exclusive), got {beta}")

    if kappa < 0:
        raise ValueError(f"kappa must be non-negative, got {kappa}")

    # Convert CVaR parameters to sum-of-k-largest parameters
    n_scenarios = x.shape[0]
    k = int((1 - beta) * n_scenarios)
    alpha = kappa * k

    return proj_sum_largest(x, k, alpha)
