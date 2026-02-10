# cbclib_v2
Convergent beam crystallography project (**cbclib**) is a library for
data processing of convergent beam crystallography datasets.

## Features

- **Hybrid CPU/CUDA backend**: Seamlessly switch between CPU and GPU computation using JAX-like device management
- **High-performance streak detection**: Optimized algorithms for detecting diffraction streaks
- **Crystal indexing**: JAX-based optimization for crystal orientation refinement
- **HDF5/CXI data handling**: Native support for crystallography data formats
- **Batch processing**: SLURM integration for cluster computing

## Dependencies

- [Python](https://www.python.org/) 3.10 or later (Python 2.x is **not** supported).
- [h5py](https://www.h5py.org) 2.10.0 or later.
- [NumPy](https://numpy.org) 1.19.0 or later.
- [Pandas](https://pandas.pydata.org) 2.1.0 or later.
- [SciPy](https://scipy.org) 1.5.2 or later.
- [JAX](https://github.com/google/jax) v0.4.20 or later.
- [PyBind11](https://github.com/pybind/pybind11) 2.10.3 or later.
- **Optional**: CUDA Toolkit 11.0+ for GPU acceleration

## Installation from source

### CPU-only installation
```bash
pip install .
```

### With CUDA support
```bash
# Ensure CUDA toolkit is installed and visible
python setup.py build_ext -i
pip install -e .
```

To skip CUDA compilation if unavailable:
```bash
export CBCLIB_SKIP_CUDA=1
pip install .
```

## Quick Start

### Device Management

```python
from cbclib_v2 import device
from cbclib_v2.ndimage import draw_lines
import numpy as np

# Check if CUDA is available
if device.is_cuda_available():
    print("CUDA backend is available!")

# Method 1: Set global device
device.set_device('cuda')
result = draw_lines(lines, shape)

# Method 2: Use context manager
with device.context('cuda'):
    result = draw_lines(lines, shape)
# Automatically returns to CPU
```
