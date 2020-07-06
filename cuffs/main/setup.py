##from setuptools import setup
##from Cython.Build import cythonize
##
##setup(
##    ext_modules = cythonize("py_cuffs.pyx")
##)

from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import os
import sys
from pathlib import Path

pypath = Path(os.path.dirname(sys.executable))
includepath = Path(pypath.as_posix()+'/Lib/site-packages/numpy/core/include')
print(includepath)

ext_modules = [Extension('py_cuffs',
                   sources=['py_cuffs.pyx'],
                   include_dirs=[includepath],
                   language='c++',
##                   extra_compile_args=['/openmp',
##                       '/O2', '/favor:INTEL64', '/fp:fast'],
                   extra_link_args=[],
                   )]
setup(name = 'py_cuffs',
      cmdclass = {'build_ext': build_ext},
      ext_modules = ext_modules)
