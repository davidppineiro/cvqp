# C++ extension imports

# Try different import paths for the C++ extension
try:
    # When installed as package
    from cvqp.libs.mybindings import sum_largest_proj as proj_sum_largest_cpp
except ImportError:
    try:
        # For relative imports within package
        from .mybindings import sum_largest_proj as proj_sum_largest_cpp
    except ImportError:
        # Fallback to direct import (development mode)
        from mybindings import sum_largest_proj as proj_sum_largest_cpp  # noqa: F401
