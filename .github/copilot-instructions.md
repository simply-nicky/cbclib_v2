# Copilot Instructions for cbclib_v2

## Collaboration Policy

**Do not jump to editing source code.** Any change — whether fixing a bug or adding a feature — must be preceded by a conversation that explores the root cause or weighs alternative solutions. Ask clarifying questions before touching files. Follow the existing code style precisely: strict Python type hints throughout, OOP design, and dataclass-based containers.

## Build

Two conda environments are used:

| Environment | Use |
|-------------|-----|
| `cbc`       | CPU-only development (default) |
| `cbc_cuda`  | GPU-accelerated development with CuPy and CUDA extensions |

```bash
# CPU-only install
conda activate cbc
pip install -e ".[dev]"

# Optional: build native extensions in-place only (skips full install)
python setup.py build_ext -i

# CUDA install
conda activate cbc_cuda
python setup.py build_ext -i
pip install -e ".[dev]"

# Skip CUDA extensions even when CUDA is present
CBCLIB_SKIP_CUDA=1 pip install -e ".[dev]"
```

## Tests

```bash
# Full suite
pytest tests/ -v

# Single file
pytest tests/streak_finder_test.py -v

# With stdout captured and verbose
pytest tests/ -vvs
```

Test files live under `tests/`, follow `*_test.py` naming, and are discovered by pytest with `--import-mode=importlib`.

## Architecture

### Three-layer structure

```
cbclib_v2/          ← Public API: re-exports, high-level data classes, subpackages
cbclib_v2/_src/     ← Private implementation: data containers, parsing, processing logic
cbclib_v2/_src/src/ ← Native extensions: C++17 (pybind11) and CUDA kernels
```

The public layer ([cbclib_v2/\_\_init\_\_.py](cbclib_v2/__init__.py)) is thin — it only re-exports from `_src`. New Python-only functionality always goes into `_src/` first and is exposed via `__init__.py`.

### Container hierarchy (`_src/data_container.py`)

All data abstractions are frozen-ish dataclasses that inherit from a clear hierarchy:

```
Container            ← dict/JSON serialisable; to_dict / from_dict / replace
  DataContainer      ← holds arrays; knows its array namespace; to_numpy/to_jax/to_cupy
    ArrayContainer   ← uniform-shape arrays; concatenate / stack / __getitem__ / reshape
      IndexedContainer  ← adds an integer `index` field grouping rows into frames
```

`replace(**kwargs)` is the idiomatic way to produce a modified copy (mirrors `dataclasses.replace` but works on the concrete subclass).

### Array-namespace portability

Every array operation uses the `ArrayNamespace` protocol ([`_src/annotations.py`](cbclib_v2/_src/annotations.py)) rather than importing numpy/jax directly. The active namespace for a container or array is resolved at runtime via `array_namespace()` from [`_src/array_api.py`](cbclib_v2/_src/array_api.py):

- `NumPy` — CPU, `np.ndarray`
- `JaxNumPy` — CPU or GPU via XLA, `jax.Array`
- `CuPy` — GPU, `cupy.ndarray` (optional, import-guarded)

Always write `xp = array_namespace(self.data)` then call `xp.zeros(...)`, `xp.concat(...)`, etc. Never import `numpy` or `jax.numpy` directly inside a function that must stay device-agnostic. Use `asnumpy`, `asjax`, `ascupy` helpers for explicit conversions.

### JAX pytree / `State` classes

`State` subclasses (in `_src/state.py`) are JAX-compatible dataclasses. Use `field(static=True)` for geometry/config attributes that must not be differentiated. **JAX registration happens lazily on first instantiation**, not at import time — this avoids the ~12 GB GPU context preallocation during import.

### Native extensions (`_src/src/`)

- C++ CPU kernels: `bresenham`, `label`, `median`, `streak_finder`, `index`; compiled with `-fopenmp` where applicable.
- CUDA GPU kernels: `cuda_draw_lines`, `cuda_label`, `cuda_median`, `cuda_streak_finder`; compiled with `nvcc`.
- `_src/src/__init__.py` does a conditional import and sets `CUDA_AVAILABLE`.

`CPPExtension` in [`setup.py`](setup.py) handles the custom nvcc dispatch (routes `.cu` files through nvcc, `.cpp` through the system C++ compiler).

#### C++ / CUDA conventions

- **Allman/BSD brace style**: opening brace on its own line for functions, classes, namespaces, and control flow. Inline braces only for single-statement lambdas.
- **Indentation**: tabs in C++/CUDA, 4 spaces in Python.
- **Precision dispatch**: use `math_traits<float>` / `math_traits<double>` specialisations (defined in `cuda_geometry.hpp`) inside kernels — never call `sqrtf` vs `::sqrt` directly.
- **pybind11 binding pattern**: register one overload per type (`m.def("func", &func<float>, ...); m.def("func", &func<double>, ...)`).
- **CPU arrays**: pass via non-owning `array<T>` view (from `array_view.hpp`); supports strided layouts.
- **CUDA arrays**: extract raw pointers from `py::array_t<T>` and pass to kernels alongside `PointND`/`LineND` geometry types.

### HDF5 / CXI protocol

[`_src/cxi_protocol.py`](cbclib_v2/_src/cxi_protocol.py) defines the HDF5 dataset abstraction used for reading/writing CBC datasets. **Do not modify this file unless explicitly instructed.**

### SLURM integration

[`cbclib_v2/slurm/`](cbclib_v2/slurm/) provides batch-job orchestration for compute clusters. Entry point: `cbclib_cli` (see `pyproject.toml`).

## Adding extensions

| What | Where |
|------|-------|
| Python-only feature | `_src/<module>.py`, expose in `__init__.py` |
| C++ CPU kernel | `_src/src/<name>.cpp` + `.pyi`, add `CPPExtension` entry to `setup.py` |
| CUDA GPU kernel | `_src/src/cuda_<name>.cu` + `.pyi`, add to CUDA block in `setup.py` |
| Tests | `tests/<name>_test.py` |

## Environment variables

| Variable | Effect |
|----------|--------|
| `CBCLIB_SKIP_CUDA` | Set to `1` to skip all CUDA extension builds |
| `XLA_PYTHON_CLIENT_PREALLOCATE` | Set to `false` to disable JAX GPU memory preallocation |
| `XLA_PYTHON_CLIENT_MEM_FRACTION` | Fraction of GPU memory JAX may use (e.g. `0.6`) |

## Type checking

```bash
pyright cbclib_v2/
```

Config is in [`pyrightconfig.json`](pyrightconfig.json): mode `standard`, target Python 3.10, private import usage suppressed (internal `_src` imports are intentional).
