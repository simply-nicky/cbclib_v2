import os
import sys
from setuptools import setup, find_namespace_packages
from pybind11.setup_helpers import Pybind11Extension
import numpy

__version__ = '0.8.1'

extension_args = {'extra_compile_args': ['-fopenmp', '-std=c++20'],
                  'extra_link_args': ['-lgomp'],
                  'library_dirs': ['/usr/local/lib',
                                   os.path.join(sys.prefix, 'lib')],
                  'include_dirs': [numpy.get_include(),
                                   os.path.join(sys.prefix, 'include'),
                                   os.path.join(os.path.dirname(__file__), 'cbclib/include')]}

extensions = [Pybind11Extension("cbclib_v2.src.fft_functions",
                                sources=["cbclib_v2/src/fft_functions.cpp"],
                                define_macros = [('VERSION_INFO', __version__)],
                                libraries = ['fftw3', 'fftw3f', 'fftw3l', 'fftw3_omp',
                                             'fftw3f_omp', 'fftw3l_omp'],
                                **extension_args),
              Pybind11Extension("cbclib_v2.src.kd_tree",
                                sources=["cbclib_v2/src/kd_tree.cpp"],
                                define_macros = [('VERSION_INFO', __version__)],
                                **extension_args),
              Pybind11Extension("cbclib_v2.src.image_proc",
                                sources=["cbclib_v2/src/image_proc.cpp"],
                                define_macros = [('VERSION_INFO', __version__)],
                                **extension_args),
              Pybind11Extension("cbclib_v2.src.label",
                                sources=["cbclib_v2/src/label.cpp"],
                                define_macros = [('VERSION_INFO', __version__)],
                                **extension_args),
              Pybind11Extension("cbclib_v2.src.median",
                                sources=["cbclib_v2/src/median.cpp"],
                                define_macros = [('VERSION_INFO', __version__)],
                                **extension_args),
              Pybind11Extension("cbclib_v2.src.signal_proc",
                                sources=["cbclib_v2/src/signal_proc.cpp"],
                                define_macros = [('VERSION_INFO', __version__)],
                                **extension_args),
              Pybind11Extension("cbclib_v2.src.streak_finder",
                                sources=["cbclib_v2/src/streak_finder.cpp"],
                                define_macros = [('VERSION_INFO', __version__)],
                                **extension_args)]

with open('README.md', 'r') as readme:
    long_description = readme.read()

setup(name='cbclib_v2',
      version=__version__,
      author='Nikolay Ivanov',
      author_email="nikolay.ivanov@desy.de",
      long_description=long_description,
      long_description_content_type='text/markdown',
      url="https://github.com/simply-nicky/cbclib_v2",
      packages=find_namespace_packages(),
      include_package_data=True,
      install_requires=['h5py', 'numpy', 'scipy'],
      ext_modules=extensions,
      classifiers=[
          "Programming Language :: Python",
          "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
          "Operating System :: OS Independent"
      ],
      python_requires='>=3.10')
