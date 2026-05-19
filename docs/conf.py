import importlib
import inspect
import os
import subprocess
from importlib.metadata import version as _pkg_version
from typing import Any

project = 'cbclib_v2'
author = 'Nikolay Ivanov'
copyright = '2024, Nikolay Ivanov'

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
napoleon_use_param = False   # types already in annotations — don't duplicate
napoleon_use_rtype = False

autodoc_typehints = 'description'
autodoc_typehints_format = 'short'
autodoc_default_options = {
    'members': True,
    'show-inheritance': True,
}
autosummary_generate = True

intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable', None),
    'jax': ('https://jax.readthedocs.io/en/latest', None),
    'pandas': ('https://pandas.pydata.org/docs', None),
}

html_theme = 'pydata_sphinx_theme'
html_theme_options = {
    'github_url': 'https://github.com/simply-nicky/cbclib_v2',
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
    module_path = info['module'].replace('.', '/')
    # Prefer .pyi stub (C++ extensions) over .py when both could exist.
    ext = None
    for candidate in ('.pyi', '.py'):
        if os.path.exists(os.path.join(_repo_root, module_path + candidate)):
            ext = candidate
            break
    if ext is None:
        return None
    # For pure-Python files, resolve line numbers so the link lands on the
    # right definition rather than the top of the file.
    linespec = ''
    if ext == '.py':
        try:
            mod = importlib.import_module(info['module'])
            obj = mod
            for part in info['fullname'].split('.'):
                obj = getattr(obj, part)
            source, lineno = inspect.getsourcelines(obj)
            linespec = f'#L{lineno}-L{lineno + len(source) - 1}'
        except Exception:
            pass
    return (
        f'https://github.com/simply-nicky/cbclib_v2/blob/{_commit}'
        f'/{module_path}{ext}{linespec}'
    )
