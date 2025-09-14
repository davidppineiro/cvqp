from setuptools import setup, Extension
import pybind11

ext_modules = [
    Extension(
        "cvqp.libs.proj_sum_largest_cpp",
        ["cvqp/libs/bindings.cpp", "cvqp/libs/sum_largest_proj.cpp"],
        include_dirs=[pybind11.get_include()],
        extra_compile_args=["-std=c++11", "-O3"],
        language="c++",
    ),
]

setup(ext_modules=ext_modules)
