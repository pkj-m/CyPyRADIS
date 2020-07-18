from setuptools import setup
from Cython.Build import cythonize
import numpy as np

setup(
    ext_modules = cythonize("pointer_to_cupy_test.pyx"),
    include_dirs = [np.get_include()]
)