import numpy as np
import pytest
from jax import random
from cbclib_v2 import default_rng, device
from cbclib_v2.indexer import CBDIndexer, CBDModel
from cbclib_v2.annotations import CuPy, Device, Generator, KeyArray, NDArray, NumPy
from cbclib_v2.test_util import TestGenerator, TestNamespace

@pytest.fixture
def key() -> KeyArray:
    return random.key(42)

@pytest.fixture
def indexer() -> CBDIndexer:
    return CBDIndexer(10)

@pytest.fixture
def model() -> CBDModel:
    return CBDModel()

@pytest.fixture
def cpu_device():
    """CPU device with multiple threads."""
    return device.cpu(num_threads=1)

@pytest.fixture
def cpu_rng() -> Generator[NDArray]:
    return default_rng(42, NumPy)

@pytest.fixture(params=device.devices())
def test_device(request: pytest.FixtureRequest) -> Device:
    """Test CPU / GPU device with multiple threads."""
    return device.to_device(request.param)

@pytest.fixture
def test_xp(test_device: Device) -> TestNamespace:
    if test_device.platform == 'cpu':
        return NumPy
    if test_device.platform == 'gpu':
        if CuPy is None:
            pytest.skip("CuPy is not available")
        return CuPy
    raise ValueError(f"Unsupported device platform: {test_device.platform}")

@pytest.fixture
def test_rng(test_xp: TestNamespace) -> TestGenerator:
    return default_rng(42, test_xp)
