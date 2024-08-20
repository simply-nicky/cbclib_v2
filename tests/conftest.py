import numpy as np
import pytest
from jax import random
from cbclib_v2.annotations import KeyArray

@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(69)

@pytest.fixture
def jax_rng() -> KeyArray:
    return random.PRNGKey(69)
