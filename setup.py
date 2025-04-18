from setuptools import setup, Extension, find_packages
import pybind11

ext_modules = [
    Extension(
        "cvqp.libs.mybindings",  # Changed to be part of the cvqp package
        ["cvqp/libs/mybindings.cpp", "cvqp/libs/sum_largest_proj.cpp"],
        include_dirs=[pybind11.get_include()],
        extra_compile_args=["-std=c++11", "-O3"],
        language="c++",
    ),
]

setup(
    name="cvqp",
    version="0.1.0",
    description="A Python implementation of the CVQP solver for CVaR-constrained quadratic programs",
    author="David Perez Pineiro, Eric Luxenberg",
    packages=find_packages(),  # Added to include Python packages
    package_data={
        "cvqp": ["libs/*.cpp", "libs/*.h"],  # Include C++ files
    },
    include_package_data=True,
    ext_modules=ext_modules,
    install_requires=["pybind11>=2.10.0", "numpy>=2.1.3", "scipy>=1.14.1"],
    python_requires=">=3.11,<4.0",
)