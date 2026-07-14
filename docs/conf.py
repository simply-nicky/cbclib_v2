import ast as _ast
import importlib
import inspect
import os
import subprocess
import sys
from importlib.metadata import version as _pkg_version
from typing import Any
import cbclib_v2

project = 'cbclib_v2'
author = 'Nikolay Ivanov'
copyright = '2026, Nikolay Ivanov'

try:
    release = _pkg_version('cbclib_v2')
except Exception:
    release = 'unknown'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon',
    'sphinx.ext.intersphinx',
    'sphinx.ext.linkcode',
    'sphinx_copybutton',
]

napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_use_param = True
napoleon_use_rtype = False
napoleon_use_ivar = True

autodoc_docstring_signature = False
autodoc_typehints = 'description'
autodoc_typehints_description_target = 'documented'

autodoc_type_aliases = {
    'Array':       '~cbclib_v2.annotations.Array',
    'RealArray':   '~cbclib_v2.annotations.RealArray',
    'IntArray':    '~cbclib_v2.annotations.IntArray',
    'BoolArray':   '~cbclib_v2.annotations.BoolArray',
    'ArrayLike':   '~cbclib_v2.annotations.ArrayLike',
    'NDArray':     '~cbclib_v2.annotations.NDArray',
    'NDRealArray': '~cbclib_v2.annotations.NDRealArray',
    'ShapeLike':   '~cbclib_v2.annotations.ShapeLike',
    'DTypeLike':   '~cbclib_v2.annotations.DTypeLike',
    'Indices':     '~cbclib_v2.annotations.Indices',
}
autodoc_typehints_format = 'short'
autoclass_content = 'class'
autodoc_class_signature = 'mixed'
autosummary_generate = True

# Show members inherited from cbclib_v2 base classes but suppress stdlib noise
# (e.g. str methods surfacing on Kinds, object methods on dataclasses).
def _autodoc_skip_stdlib_member(app: Any, what: str, name: str, obj: Any,
                                skip: bool, options: Any) -> bool:
    """Hide members inherited from low-level stdlib bases."""
    owner = getattr(obj, '__objclass__', None)
    if owner in (object, str):
        return True

    qualname = getattr(obj, '__qualname__', '')
    if qualname.startswith(('object.', 'str.')):
        return True

    return skip

intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable', None),
    'jax': ('https://jax.readthedocs.io/en/latest', None),
    'pandas': ('https://pandas.pydata.org/docs', None),
    'h5py': ('https://docs.h5py.org/en/latest/', None),
}

templates_path = ['_templates']

html_static_path = ['_static']
html_logo = '_static/logo.svg'

html_theme = 'pydata_sphinx_theme'
html_theme_options = {
    'github_url': 'https://github.com/simply-nicky/cbclib_v2',
    'logo': {
        'image_light': '_static/logo.svg',
        'image_dark': '_static/logo.svg',
    },
    'navigation_depth': 3,
}

# GitHub source links — pinned to the exact commit the docs were built from.
_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

try:
    _commit = subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'],
        cwd=_repo_root,
        stderr=subprocess.DEVNULL,
    ).decode().strip()
except Exception:
    _commit = 'main'

def linkcode_resolve(domain: str, info: dict[str, Any]) -> str | None:
    if domain != 'py' or not info['module']:
        return None

    submod = sys.modules.get(info['module'])
    if submod is None:
        try:
            submod = importlib.import_module(info['module'])
        except Exception:
            return None

    obj = submod
    for part in info['fullname'].split('.'):
        try:
            obj = getattr(obj, part)
        except Exception:
            return None

    while hasattr(obj, '__wrapped__'):
        obj = obj.__wrapped__

    try:
        fn = inspect.getsourcefile(obj)
    except Exception:
        fn = None

    # Fallback for compiled extensions: look for a .pyi stub next to the .so
    if fn is None:
        module_path = info['module'].replace('.', '/')
        pyi = os.path.join(_repo_root, module_path + '.pyi')
        if os.path.exists(pyi):
            fn = pyi

    if fn is None:
        return None

    try:
        source, lineno = inspect.getsourcelines(obj)
        linespec = f'#L{lineno}-L{lineno + len(source) - 1}'
    except Exception:
        linespec = ''

    startdir = os.path.abspath(os.path.join(os.path.dirname(cbclib_v2.__file__), '..'))
    fn = os.path.relpath(fn, start=startdir).replace(os.path.sep, '/')
    if not fn.startswith('cbclib_v2/'):
        return None

    return f'https://github.com/simply-nicky/cbclib_v2/blob/{_commit}/{fn}{linespec}'

def _load_stub_docs() -> dict[str, str]:
    """Parse .pyi stubs and return a {ClassName / ClassName.method: docstring} mapping."""
    result: dict[str, str] = {}
    stubs = [
        os.path.join(_repo_root, 'cbclib_v2', '_src', 'src', 'label.pyi'),
        os.path.join(_repo_root, 'cbclib_v2', '_src', 'src', 'streak_finder.pyi'),
    ]
    for path in stubs:
        if not os.path.exists(path):
            continue
        with open(path) as fh:
            tree = _ast.parse(fh.read())
        for node in tree.body:
            if isinstance(node, _ast.ClassDef):
                doc = _ast.get_docstring(node)
                if doc:
                    result[node.name] = doc
                for item in node.body:
                    if isinstance(item, (_ast.FunctionDef, _ast.AsyncFunctionDef)):
                        fdoc = _ast.get_docstring(item)
                        if fdoc:
                            result[f'{node.name}.{item.name}'] = fdoc
    return result

_STUB_DOCS = _load_stub_docs()

def _autodoc_process_pybind11(app: Any, what: str, name: str, obj: Any,
                               options: Any, lines: list[str]) -> None:
    """Fill empty docstrings for C++ extension objects from .pyi stubs.

    With disable_function_signatures() active in every pybind11 module,
    __doc__ is None for objects without an explicit C++ docstring.  This
    hook injects the matching stub docstring so Napoleon can process it.
    Runs at priority 100, before Napoleon (500).
    """
    if lines:
        return
    mod = getattr(obj, '__module__', None)
    if mod is None or '_src.src' not in mod:
        return
    parts = name.split('.')
    for key in (
        f'{parts[-2]}.{parts[-1]}' if len(parts) >= 2 else None,
        parts[-1],
    ):
        if key and key in _STUB_DOCS:
            lines[:] = _STUB_DOCS[key].splitlines()
            return

def setup(app: Any) -> None:
    app.connect('autodoc-skip-member', _autodoc_skip_stdlib_member)
    app.connect('autodoc-process-docstring', _autodoc_process_pybind11, priority=100)
