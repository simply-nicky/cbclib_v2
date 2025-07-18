import os
import sys
from setuptools import setup, find_namespace_packages
from pybind11.setup_helpers import Pybind11Extension
import numpy

__version__ = '0.10.7'

extension_args = {'extra_compile_args': ['-fopenmp', '-std=c++20'],
                  'extra_link_args': ['-lgomp'],
                  'library_dirs': ['/usr/local/lib',
                                   os.path.join(sys.prefix, 'lib')],
                  'include_dirs': [numpy.get_include(),
                                   os.path.join(sys.prefix, 'include')]}

extensions = [
      Pybind11Extension("cbclib_v2._src.src.bresenham",
                        sources=["cbclib_v2/_src/src/bresenham.cpp"],
                        define_macros = [('VERSION_INFO', __version__)],
                        **extension_args),
      Pybind11Extension("cbclib_v2._src.src.fft_functions",
                        sources=["cbclib_v2/_src/src/fft_functions.cpp"],
                        define_macros = [('VERSION_INFO', __version__)],
                        libraries = ['fftw3', 'fftw3f', 'fftw3l', 'fftw3_omp',
                                    'fftw3f_omp', 'fftw3l_omp'],
                        **extension_args),
      Pybind11Extension("cbclib_v2._src.src.index",
                        sources=["cbclib_v2/_src/src/index.cpp"],
                        define_macros = [('VERSION_INFO', __version__)],
                        **extension_args),
      Pybind11Extension("cbclib_v2._src.src.label",
                        sources=["cbclib_v2/_src/src/label.cpp"],
                        define_macros = [('VERSION_INFO', __version__)],
                        **extension_args),
      Pybind11Extension("cbclib_v2._src.src.median",
                        sources=["cbclib_v2/_src/src/median.cpp"],
                        define_macros = [('VERSION_INFO', __version__)],
                        **extension_args),
      Pybind11Extension("cbclib_v2._src.src.signal_proc",
                        sources=["cbclib_v2/_src/src/signal_proc.cpp"],
                        define_macros = [('VERSION_INFO', __version__)],
                        **extension_args),
      Pybind11Extension("cbclib_v2._src.src.streak_finder",
                        sources=["cbclib_v2/_src/src/streak_finder.cpp"],
                        define_macros = [('VERSION_INFO', __version__)],
                        **extension_args)
]

setup(version=__version__,
      packages=find_namespace_packages(),
      include_package_data=True,
      install_requires=['numpy', 'pybind11'],
      ext_modules=extensions)
