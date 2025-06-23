# C++ extension imports

# Try different import paths for the C++ extension
try:
    # When installed as package
    from cvqp.libs.proj_sum_largest_cpp import proj_sum_largest_cpp
except ImportError:
    try:
        # For relative imports within package
        from .proj_sum_largest_cpp import proj_sum_largest_cpp
    except ImportError:
        # Fallback to direct import (development mode)
        from proj_sum_largest_cpp import proj_sum_largest_cpp  # noqa: F401
