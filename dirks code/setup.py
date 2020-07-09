from setuptools import setup
from Cython.Build import cythonize
import numpy as np

setup(
    ext_modules = cythonize("cycupy_test.pyx"),
    include_dirs = [np.get_include()]
)

