import numpy as np
import pytest
from jax import random
from cbclib_v2.annotations import KeyArray
from cbclib_v2.test_util import TestModel

@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(69)

@pytest.fixture
def key() -> KeyArray:
    return random.key(69)

@pytest.fixture
def model() -> TestModel:
    return TestModel()
