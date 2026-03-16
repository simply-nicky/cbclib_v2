# Copilot Instructions for cbclib_v2

## Project Overview
**cbclib_v2** is a Python library for convergent beam crystallography (CBC) data processing. It combines **PyData ecosystem** (NumPy, Pandas, JAX) with **C++17 extensions** (pybind11) and optional **CUDA acceleration** for high-performance array operations on crystallography datasets.

## Critical Constraints

### Environment Setup
- **Development uses conda environments**:
  - `cbc`: CPU-only setup (default environment, see [environment.yml](environment.yml))
  - `cbc_cuda`: CUDA-supported setup with GPU acceleration
- Always verify which environment is active when debugging GPU/CPU issues

### Code Modification Policy
- **DO NOT modify CXI protocol or HDF5 data handling code** ([cxi_protocol.py](cbclib_v2/_src/cxi_protocol.py)) unless explicitly instructed
- **DO NOT create extra files** (scripts, docs, configs) unless explicitly requested
- When asked to implement features, modify existing files rather than creating new ones

### Formatting Rules (C++/CUDA)
Codebase uses **Allman/BSD style** with strict whitespace conventions:
- **Opening braces on new lines** for functions, classes, namespaces, control flow:
  ```cpp
  void function()
  {
      if (condition)
      {
          // code
      }
  }
  ```
- **Inline braces** only for single-statement lambdas: `[](auto x){ return x + 1; }`
- **NO trailing whitespace or tabs** at end of lines
- **Indentation**: Use tabs for C++ (inferred from existing code), spaces for Python (4 spaces)
- **Line length**: Aim for <100 chars where practical (not strictly enforced)

## Architecture

### Layered Structure
- **Python API layer** ([cbclib_v2/](cbclib_v2/)): User-facing data classes and workflows
- **Native extensions** ([cbclib_v2/_src/src/](cbclib_v2/_src/src/)): C++17 and CUDA kernels compiled via setuptools
- **Intermediate Python** ([cbclib_v2/_src/](cbclib_v2/_src/)): Data containers, state management, parsing

### Key Components
1. **Data I/O**: [crystfel.py](cbclib_v2/_src/crystfel.py) (geometry parsing), [cxi_protocol.py](cbclib_v2/_src/cxi_protocol.py) (HDF5 dataset abstraction)
2. **Geometry & Indexing**: [indexer/](cbclib_v2/indexer/) subpackage (crystal orientation optimization via JAX)
3. **Streak Detection**: [streak_finder/](cbclib_v2/) (signal processing + label analysis)
4. **Metadata Extraction**: [data_processing.py](cbclib_v2/_src/data_processing.py) (PCA, whitefield, variance estimation)
5. **SLURM Integration**: [slurm/scripts.py](cbclib_v2/slurm/scripts.py) (batch processing orchestration for compute clusters)
6. **Device Management**: [device.py](cbclib_v2/_src/device.py) (CPU/CUDA backend switching, JAX-like context API)

### JAX Integration (Critical Gotcha)
- **JAX is lazily initialized**: `State` subclasses (e.g., `XtalState`, `FixedSetup`) register as JAX pytrees **only on first instance creation**, not on import
- **Why**: JAX's `register_pytree_with_keys()` triggers GPU context init and ~12GB preallocation on first call. Lazy registration defers this until actually needed.
- **Impact**: Package imports are GPU-safe; GPU memory is allocated only when JAX computations run.
- **In [test_util.py](cbclib_v2/test_util.py)**: `get_random_xtal()` / `get_random_setup()` use lazy getters for module-level test fixtures

## Developer Workflows

### Build
CPU-only (cbc):
```bash
conda activate cbc
pip install -e ".[dev]"
# Or build native extensions in-place first (optional)
python setup.py build_ext -i
```

With CUDA (cbc_cuda):
```bash
conda activate cbc_cuda
# Ensure CUDA toolkit is visible to the build
python setup.py build_ext -i
pip install -e ".[dev]"
```

### Test
CPU-only (cbc):
```bash
conda activate cbc
pytest tests/ -v
```

With CUDA (cbc_cuda):
```bash
conda activate cbc_cuda
pytest tests/ -v
```

Extras:
```bash
# Verbose output + capture disabled
pytest tests/ -vvs

# Single file
pytest tests/label_test.py -v
```

### Common Issues
- **CUDA not found at build time**: Set `CBCLIB_SKIP_CUDA=1` to disable CUDA extension.
- **JAX GPU preallocation**: Control via environment variables before import:
  ```bash
  export XLA_PYTHON_CLIENT_PREALLOCATE=false
  export XLA_PYTHON_CLIENT_MEM_FRACTION=0.6  # Use 60% of GPU
  ```

## Critical Patterns

### State & JAX Pytrees
Classes inheriting from `State` (in `_src/state.py`) are JAX-compatible dataclasses with **dynamic fields** (pytree leaves for differentiation) and **static fields** (hashed for control flow). Example:
```python
class XtalState(ArrayContainer, State):
    lattice: RealArray  # Dynamic (traced by JAX)
    cell: CrystCell = field(static=True)  # Static (compared by JAX)
```
Subclasses automatically register with JAX on first `__init__` call via `_register_pytree()`.

### C++/CUDA Templating
Native extensions use **math_traits specialization** for float/double precision dispatch:
- **Template kernel**: Single generic template `template <typename T> void kernel(T* data, ...)`
- **Type-specific math**: Define `math_traits<float>` and `math_traits<double>` structs in header (e.g., `cuda_geometry.hpp`):
  ```cpp
  template <> struct math_traits<float> {
      static HOST_DEVICE float sqrt(float x) { return sqrtf(x); }
      static HOST_DEVICE float max(float x, float y) { return fmaxf(x, y); }
  };
  template <> struct math_traits<double> {
      static HOST_DEVICE double sqrt(double x) { return ::sqrt(x); }
      static HOST_DEVICE double max(double x, double y) { return ::fmax(x, y); }
  };
  ```
- **Multiple pybind11 definitions**: Register separate Python functions for each type (same name):
  ```cpp
  m.def("draw_lines", &cuda::draw_lines<float>, ...);
  m.def("draw_lines", &cuda::draw_lines<double>, ...);
  ```
- **Usage in kernel**: `math_traits<T>::sqrt(x)` automatically selects float or double variant.
- Example: `cbclib_v2/_src/src/cuda_geometry.hpp` (lines 20–56) defines traits; `cuda_functions.cu:287` uses them.

### DataContainer & ArrayNamespace
All data abstractions inherit from `DataContainer` and accept an `xp: ArrayNamespace` parameter (NumPy or JAX):
```python
class Detector(DataContainer):
  def indices(self) -> 'PixelIndices':  # Return CPU-friendly indices
  def to_patterns(self, streaks: Streaks, xp: ArrayNamespace = NumPy) -> Patterns:
    # Use xp.array(), xp.dot(), etc. for GPU/JAX compatibility
```
This pattern enables seamless CPU → GPU portability.

### Lightweight Array Class (`array<T>`) — CPU Only
The `array<T>` template class (in [array.hpp](cbclib_v2/_src/src/array.hpp)) provides a non-owning view wrapper for multi-dimensional arrays with full support for **both contiguous and non-contiguous layouts**:
- **Shape & stride metadata**: Tracks array dimensions, strides, and itemsize
- **Iterator support**: `array_iterator<T, IsConst>` implements random-access iteration over both C-contiguous and strided arrays
- **Indexing helpers**: Methods like `at()`, `slice()`, `slice_front()`, `slice_back()` work universally across any memory layout
- **CPU-only**: All C++ CPU functions use `array<T>`; CUDA kernels use device-compatible types from `cuda_geometry.hpp`
- **Python interop**: Implicit conversion to `py::array_t<T>` for pybind11; accepts NumPy arrays with arbitrary strides

### Device-Friendly Geometry Types (CUDA)
CUDA kernels use lightweight device-compatible types from [cuda_geometry.hpp](cbclib_v2/_src/src/cuda_geometry.hpp):
- **`PointND<T, N>`**: N-dimensional point with arithmetic operations (addition, scaling, dot product)
- **`LineND<T, N>`**: Line segment with distance/projection calculations
- **`ThickLineND<T, N>`**: Line segment with thickness (width) for rendering operations
- **`math_traits<float>` / `math_traits<double>`**: Type-specific math dispatch (sqrtf vs ::sqrt, fmaxf vs ::fmax, etc.)
- All types support HOST_DEVICE compilation for both CPU and GPU targets

### pybind11 Conventions
- Module name matches extension name: `PYBIND11_MODULE(label, m)` → `cbclib_v2._src.src.label`
- Use `.def()` for functions, `.def_readonly()` for C++ struct fields
- NumPy arrays: bind via `py::array_t<T>` with format `c_style | forcecast`
- Keep bindings minimal; use pybind11-generated type annotations (`.pyi` files auto-generated)
- **Array wrapping**: Pass NumPy arrays to C++ via `array<T>` constructor accepting `py::buffer_info` for full stride support (CPU functions only)
- **CUDA binding pattern**: Accept `py::array_t<T>` in bindings, extract raw pointers and metadata, pass to templated kernels expecting `PointND`, `LineND`, or raw pointers

## Performance & Optimization

### Spatial Indexing (CUDA Example)
The [cuda_functions.cu](cbclib_v2/_src/src/cuda_functions.cu) `draw_thick_lines_3d()` kernel demonstrates:
1. **CPU preprocessing**: Build spatial grid (CSR format) to cull candidate line segments
2. **GPU point-parallel**: Each thread processes one voxel, queries grid, computes distance
3. **Result**: 10–100× speedup vs naive all-pairs brute force

When adding new CUDA kernels: preprocess on CPU to reduce device kernel branching.

### Memory Management
- CUDA device memory is **not explicitly freed** in Python bindings (relies on scope cleanup via pybind11)
- For long-running processes: explicitly call `cudaFreeAll()` or Python's `gc.collect()` if needed.

## File Organization

```
cbclib_v2/
├── __init__.py              # Imports main API (Detector, State, etc.)
├── _src/                    # "Private" implementation (but public imports)
│   ├── state.py             # JAX State base class + pytree registration
│   ├── crystfel.py          # Detector/Panel parsing
│   ├── data_processing.py   # High-level workflows
│   └── src/                 # C++/CUDA extensions
│       ├── *.cpp            # CPU kernels (pybind11-wrapped)
│       ├── cuda_functions.cu # GPU kernels
│       └── *.hpp            # Template definitions (included in .cpp/.cu)
├── indexer/                 # JAX-based crystal orientation optimization
├── slurm/                   # Batch job orchestration
└── streak_finder/           # Streak detection pipelines
```

## Editing Guidelines

### When Adding Features
1. **Python-only**: Add to [_src/](cbclib_v2/_src/) module, expose via [__init__.py](cbclib_v2/__init__.py)
2. **C++ CPU kernel**: Create `_src/src/feature.cpp`, use pybind11, add to [setup.py](setup.py) extensions list
3. **CUDA GPU kernel**: Add to [cuda_functions.cu](cbclib_v2/_src/src/cuda_functions.cu), template for float/double, expose in `PYBIND11_MODULE(cuda_functions, m)`
4. **Tests**: Place in `tests/feature_test.py`, name functions `test_*` for pytest discovery

### Modifying State Subclasses
- Always use `State` base class (not raw `@dataclass`) for JAX compatibility
- Use `field(static=True)` for non-differentiable attributes (geometry, config)
- Do **not** manually call `_register_pytree()`; happens automatically on first instance

### GPU Memory Control
- Document environment variables in docstrings: e.g., "Respects `XLA_PYTHON_CLIENT_MEM_FRACTION`"
- Test locally with `XLA_PYTHON_CLIENT_PREALLOCATE=false` before committing GPU-heavy code.

## References
- **State/JAX**: [state.py](cbclib_v2/_src/state.py) (lines 87–183)
- **Pytree registration**: `_register_pytree()` method defers JAX init
- **Array class**: [array.hpp](cbclib_v2/_src/src/array.hpp) (non-owning view with full stride support)
- **CUDA templating**: [cuda_functions.cu](cbclib_v2/_src/src/cuda_functions.cu) (lines 1–50, 178–461)
- **pybind11 example**: [label.cpp](cbclib_v2/_src/src/label.cpp) (line 464+)
- **DataContainer pattern**: [data_container.py](cbclib_v2/_src/data_container.py)
- **Build system**: [setup.py](setup.py) (CPPExtension class, conda CUDA paths)
