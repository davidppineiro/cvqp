"""
Efficient projection onto sum-of-largest-elements constraint sets.
"""

import numpy as np
from .libs import proj_sum_largest_cpp


def proj_sum_largest(z: np.ndarray, k: int, alpha: float) -> np.ndarray:
    """
    Project vector onto {x | sum(k largest elements) ≤ α}.
    
    Efficiently computes the Euclidean projection of a vector z onto the set
    where the sum of its k largest elements does not exceed alpha.
    
    Args:
        z: Input vector to project
        k: Number of largest elements (must be 0 < k < len(z))
        alpha: Upper bound on sum (must be alpha ≥ 0)
        
    Returns:
        Projected vector with same shape as input
        
    Note:
        Uses fast C++ implementation
    """
    # Input validation
    if not isinstance(z, np.ndarray):
        raise TypeError(f"Input z must be a numpy array, got {type(z)}")
    
    if z.ndim != 1:
        raise ValueError(f"Input z must be a 1D array, got shape {z.shape}")
        
    if not 0 < k < len(z):
        raise ValueError(f"k must be between 0 and len(z), got {k}")
        
    if alpha < 0:
        raise ValueError(f"alpha must be non-negative, got {alpha}")
    
    # Simple case: if sum of all elements is less than alpha, return z unchanged
    if np.sum(z) <= alpha:
        return z.copy()
        
    # Sort indices in descending order
    sorted_inds = np.argsort(z)[::-1]
    z_sorted = z[sorted_inds]

    # Call C++ implementation (discarding extra return values)
    z_projected, *_ = proj_sum_largest_cpp(
        z_sorted, k, alpha, k, 0, len(z), False
    )

    # Restore original ordering
    x = np.empty_like(z)
    x[sorted_inds] = z_projected
    return x