# cbclib_v2

**cbclib_v2** is a Python library for processing serial crystallography datasets
measured at free-electron lasers (FELs) such as EuXFEL, SwissFEL, and LCLS.
It targets convergent beam crystallography (CBC) experiments on protein crystals
and small-molecule crystals, and covers the main stages of an X-ray
crystallography data-processing pipeline.

The library is inspired by [CrystFEL](https://www.desy.de/~twhite/crystfel/) and
extends its concepts to the specific geometry of convergent beam diffraction.

## Processing pipeline

- **Pattern pre-processing** — background estimation, variance analysis, and
  PCA-based whitefield correction.
- **Streak detection** — high-performance connected-component labelling and
  line-fitting algorithms implemented in C++17 with optional CUDA acceleration.
- **Indexing** — JAX-based optimisation of crystal orientation and unit-cell parameters.
- **Intensity scaling** — planned.

## Installation

CPU-only (default):
```bash
pip install cbclib_v2
```

With CUDA support:
```bash
python setup.py build_ext -i
pip install cbclib_v2
```

To skip CUDA compilation when CUDA headers are present but unwanted:
```bash
CBCLIB_SKIP_CUDA=1 pip install cbclib_v2
```

## Dependencies

- [Python](https://www.python.org/) 3.10 or later
- [NumPy](https://numpy.org) 2.1.0 or later
- [JAX](https://github.com/google/jax) 0.4.20 or later
- [pandas](https://pandas.pydata.org)
- [h5py](https://www.h5py.org)
- [hdf5plugin](https://github.com/silx-kit/hdf5plugin)
- [array-api-compat](https://github.com/data-apis/array-api-compat)
- [pybind11](https://github.com/pybind/pybind11) 2.10.0 or later
- **Optional**: [CuPy](https://cupy.dev) 13.0+ for CUDA acceleration

## Reference

If you use cbclib_v2 in your research, please cite:

C. Li, M. Zakharova, M. Prasciolu, J. C. Wong, H. Fleckenstein, N. Ivanov,
W. Zhang, M. Butola, J. L. Dresselhaus, I. De Gennaro Aquino, D. Egorov,
P. Middendorf, A. Henkel, B. Klopprogge, L. Klemeyer, T. Beck, O. Yefanov,
M. Barthelmess, J. Sprenger, D. Oberthuer, S. Bajt, and H. N. Chapman,
"Convergent-Beam X-ray Crystallography," arXiv:2602.14402 (2026).
https://arxiv.org/abs/2602.14402
