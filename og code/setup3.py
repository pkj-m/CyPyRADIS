from setuptools import setup
from Cython.Build import cythonize
import numpy as np

setup(
    ext_modules = cythonize("constant_memory_read_write_check.pyx"),
    include_dirs = [np.get_include()]
)