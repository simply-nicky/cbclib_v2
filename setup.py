import os
import platform
import shutil
import sys
from typing import Iterable, Protocol
from setuptools import setup, find_namespace_packages, Extension
from setuptools.command.build_ext import build_ext
from numpy import get_include as numpy_get_include
from pybind11 import get_include as pybind11_get_include

IS_WINDOWS = sys.platform == 'win32'
IS_MACOS = sys.platform.startswith('darwin')
IS_LINUX = sys.platform.startswith('linux')

__version__ = '0.13.1'

def find_conda_home() -> str:
    """Find the Conda install path."""
    conda_home = os.environ.get('CONDA_PREFIX')
    if conda_home is None:
        raise RuntimeError("Could not find Conda installation home folder.")
    return conda_home

def find_cuda_home() -> str | None:
    """Find the CUDA install path."""
    # Guess #1
    cuda_home = os.environ.get('CUDA_HOME') or os.environ.get('CUDA_PATH')
    if cuda_home is None:
        # Guess #2
        nvcc_path = shutil.which("nvcc")
        if nvcc_path is not None:
            cuda_home = os.path.dirname(os.path.dirname(nvcc_path))

    return cuda_home

SKIP_CUDA = os.environ.get('CBCLIB_SKIP_CUDA', '0').lower() in ('1', 'true', 'yes')
CUDA_HOME_FOUND = find_cuda_home() is not None and not SKIP_CUDA

def cuda_include() -> str:
    """ Return the CUDA include path. """
    conda_home = find_conda_home()

    arch = platform.machine() # e.g., 'x86_64' or 'aarch64'
    system = platform.system().lower() # e.g., 'linux' or 'windows'
    include_path = os.path.join(conda_home, 'targets', f'{arch}-{system}', 'include')
    if not os.path.exists(include_path):
        raise RuntimeError(f"CUDA include path does not exist: {include_path}")

    return include_path

def cuda_library_path() -> str:
    """ Return the CUDA library path. """
    conda_home = find_conda_home()

    lib_path = os.path.join(conda_home, 'lib')
    if not os.path.exists(lib_path):
        raise RuntimeError(f"CUDA library path does not exist: {lib_path}")

    return lib_path

class CCompiler(Protocol):
    compiler_so : list[str]
    src_extensions : list[str]

    def _compile(self, obj: str, src: str, ext: str, cc_args: list[str],
                 extra_postargs: list[str], pp_opts: list[str]) -> None: ...

    def set_executable(self, key: str, value: list[str]) -> None: ...

class CPPExtension(Extension):
    """
    Build a C++11+ Extension module with pybind11. This automatically adds the
    recommended flags when you init the extension and assumes C++ sources - you
    can further modify the options yourself.

    The customizations are:

    * ``/EHsc`` and ``/bigobj`` on Windows
    * ``stdlib=libc++`` on macOS
    * ``visibility=hidden`` and ``-g0`` on Unix

    Finally, you can set ``cxx_std`` via constructor or afterwards to enable
    flags for C++ std, and a few extra helper flags related to the C++ standard
    level. It is _highly_ recommended you either set this, or use the provided
    ``build_ext``, which will search for the highest supported extension for
    you if the ``cxx_std`` property is not set. Do not set the ``cxx_std``
    property more than once, as flags are added when you set it. Set the
    property to None to disable the addition of C++ standard flags.

    If you want to add pybind11 headers manually, for example for an exact
    git checkout, then set ``include_pybind11=False``.
    """
    STD_TMPL = "/std:c++{}" if IS_WINDOWS else "-std=c++{}"
    extra_compile_args: dict[str, list[str]]

    def __init__(
        self,
        name: str,
        sources: Iterable[str],
        include_dirs: list[str] | None = None,
        define_macros: list[tuple[str, str | None]] | None = None,
        undef_macros: list[str] | None = None,
        library_dirs: list[str] | None = None,
        libraries: list[str] | None = None,
        runtime_library_dirs: list[str] | None = None,
        extra_objects: list[str] | None = None,
        extra_compile_args: list[str] | None = None,
        extra_link_args: list[str] | None = None,
        export_symbols: list[str] | None = None,
        swig_opts: list[str] | None = None,
        depends: list[str] | None = None,
        language: str | None = None,
        optional: bool | None = None,
        *,
        cxx_std: int = 17,
        include_pybind11: bool = True,
        include_numpy: bool = True,
        py_limited_api: bool = False,
    ) -> None:
        if language is None:
            language = "c++"

        if isinstance(extra_compile_args, dict):
            cxx_flags = extra_compile_args.get('cxx', [])
            nvcc_flags = extra_compile_args.get('nvcc', [])
        else:
            cxx_flags = extra_compile_args or []
            nvcc_flags = []

        super().__init__(name, sources, include_dirs, define_macros, undef_macros,
                         library_dirs, libraries, runtime_library_dirs, extra_objects,
                         cxx_flags, extra_link_args, export_symbols, swig_opts,
                         depends, language, optional, py_limited_api=py_limited_api)

        self.extra_compile_args = {'cxx': cxx_flags, 'nvcc': nvcc_flags}

        # Include the installed package pybind11 headers
        if include_pybind11:
            self.include_dirs.append(pybind11_get_include())
        if include_numpy:
            self.include_dirs.append(numpy_get_include())

        # Setting extra compile args, flags are prepended, so that they can be further
        # overridden, e.g. by ``extra_compile_args=["-g"]``.

        # Set C++ standard flags
        self.extra_compile_args['cxx'][:0] += [self.STD_TMPL.format(cxx_std)]
        self.extra_compile_args['nvcc'][:0] += [self.STD_TMPL.format(cxx_std)]

        if IS_MACOS and "MACOSX_DEPLOYMENT_TARGET" not in os.environ:
            # C++17 requires a higher min version of macOS. An earlier version
            # (10.12 or 10.13) can be set manually via environment variable if
            # you are careful in your feature usage, but 10.14 is the safest
            # setting for general use. However, never set higher than the
            # current macOS version!
            current_macos = tuple(int(x) for x in platform.mac_ver()[0].split(".")[:2])
            desired_macos = (10, 9) if cxx_std < 17 else (10, 14)
            macos_string = ".".join(str(x) for x in min(current_macos, desired_macos))
            macosx_min = f"-mmacosx-version-min={macos_string}"
            self.extra_compile_args['cxx'][:0] += [macosx_min]
            self.extra_link_args[:0] += [macosx_min]

        if IS_WINDOWS:
            self.extra_compile_args['cxx'][:0] += ["/EHsc", "/bigobj"]
        if IS_MACOS:
            self.extra_compile_args['cxx'][:0] += ["-stdlib=libc++"]

class AnyExtension(Protocol):
    name                    : str
    sources                 : list[str]
    include_dirs            : list[str]
    define_macros           : list[tuple[str, str | None]]
    undef_macros            : list[str]
    library_dirs            : list[str]
    libraries               : list[str]
    runtime_library_dirs    : list[str]
    extra_objects           : list[str]
    extra_compile_args      : dict[str, list[str]] | list[str]
    extra_link_args         : list[str]

class BuildCPPExp(build_ext):
    compiler    : CCompiler
    extensions  : list[AnyExtension]

    def add_cxx_extra_args(self, extension: AnyExtension, args: list[str]) -> None:
        if isinstance(extension.extra_compile_args, dict):
            extension.extra_compile_args['cxx'] += args
        else:
            extension.extra_compile_args += args

    def add_nvcc_extra_args(self, extension: AnyExtension, args: list[str]) -> None:
        if isinstance(extension.extra_compile_args, dict):
            extension.extra_compile_args['nvcc'] += args

    def build_extensions(self):
        # You can detect --debug via self.debug
        self.compiler.src_extensions += ['.cu']
        original_compile = self.compiler._compile

        for ext in self.extensions:
            if self.debug:
                # Add your debug flags here
                self.add_cxx_extra_args(ext, ["-g", "-O0", "-D_FORTIFY_SOURCE=0"])
                self.add_nvcc_extra_args(ext, ["-G"])
            else:
                self.add_cxx_extra_args(ext, ["-O3"])
                self.add_nvcc_extra_args(ext, ["-O3"])
                if IS_LINUX:
                    self.add_cxx_extra_args(ext, ["-fvisibility=hidden", "-g0"])

            ext.library_dirs += [os.path.join(sys.prefix, 'lib')]
            ext.include_dirs += [os.path.join(sys.prefix, 'include')]

            ext.define_macros += [('VERSION_INFO', __version__)]

        def wrap_single_compile(obj: str, src: str, ext: str, cc_args: list[str],
                                extra_postargs: list[str] | dict[str, list[str]] | None,
                                pp_opts: list[str]) -> None:
            # Copy before we make any modifications.
            original_compiler = self.compiler.compiler_so
            if extra_postargs is None:
                extra_postargs = []

            try:
                if src.endswith('.cu'):
                    nvcc = [os.path.join(find_conda_home(), 'bin', 'nvcc')]
                    self.compiler.set_executable('compiler_so', nvcc)

                    # Handle both list and dict formats
                    if isinstance(extra_postargs, dict):
                        cuda_flags = extra_postargs.get('nvcc', [])
                        host_flags = extra_postargs.get('cxx', [])
                    else:
                        # Fallback to heuristic for list format
                        cuda_flags = []
                        host_flags = extra_postargs

                    if '-fPIC' not in host_flags:
                        host_flags[:0] += ['-fPIC']

                    cflags = cuda_flags
                    if host_flags:
                        cflags += ['--compiler-options', ','.join(host_flags)]

                elif isinstance(extra_postargs, dict):
                    cflags = extra_postargs.get('cxx', [])
                else:
                    cflags = extra_postargs

                original_compile(obj, src, ext, cc_args, cflags, pp_opts)
            finally:
                # Put the original compiler back in place.
                self.compiler.set_executable('compiler_so', original_compiler)

        self.compiler._compile = wrap_single_compile

        super().build_extensions()

extensions = [
    CPPExtension("cbclib_v2._src.src.bresenham",
                 sources=["cbclib_v2/_src/src/bresenham.cpp"],
                 cxx_std=17,
                 extra_compile_args=['-fopenmp'],
                 extra_link_args=['-lgomp']),
    CPPExtension("cbclib_v2._src.src.index",
                 sources=["cbclib_v2/_src/src/index.cpp"],
                 cxx_std=17),
    CPPExtension("cbclib_v2._src.src.label",
                 sources=["cbclib_v2/_src/src/label.cpp"],
                 cxx_std=17,
                 extra_compile_args=['-fopenmp'],
                 extra_link_args=['-lgomp']),
    CPPExtension("cbclib_v2._src.src.median",
                 sources=["cbclib_v2/_src/src/median.cpp"],
                 cxx_std=17,
                 extra_compile_args=['-fopenmp'],
                 extra_link_args=['-lgomp']),
    CPPExtension("cbclib_v2._src.src.streak_finder",
                 sources=["cbclib_v2/_src/src/streak_finder.cpp"],
                 cxx_std=17,
                 extra_compile_args=['-fopenmp'],
                 extra_link_args=['-lgomp']),
    CPPExtension("cbclib_v2._src.src.test",
                 sources=["cbclib_v2/_src/src/test.cpp"],
                 cxx_std=17)
]

if CUDA_HOME_FOUND:
    extensions += [
        CPPExtension("cbclib_v2._src.src.cuda_draw_lines",
                     sources=["cbclib_v2/_src/src/cuda_draw_lines.cu",],
                     cxx_std=17,
                     include_dirs=[cuda_include()],
                     library_dirs=[cuda_library_path()],
                     libraries=['cudart']),
        CPPExtension("cbclib_v2._src.src.cuda_label",
                     sources=["cbclib_v2/_src/src/cuda_label.cu",],
                     cxx_std=17,
                     include_dirs=[cuda_include()],
                     library_dirs=[cuda_library_path()],
                     libraries=['cudart']),
        CPPExtension("cbclib_v2._src.src.cuda_median",
                     sources=["cbclib_v2/_src/src/cuda_median.cu",],
                     cxx_std=17,
                     include_dirs=[cuda_include()],
                     library_dirs=[cuda_library_path()],
                     libraries=['cudart'])
    ]

setup(
    version=__version__,
    packages=find_namespace_packages(),
    include_package_data=True,
    ext_modules=extensions,
    cmdclass={"build_ext": BuildCPPExp}
)
