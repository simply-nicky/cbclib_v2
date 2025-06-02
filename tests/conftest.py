import numpy as np
import pytest
from jax import random
from cbclib_v2.indexer import CBDIndexer, CBDModel
from cbclib_v2.annotations import KeyArray

@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(69)

@pytest.fixture
def key() -> KeyArray:
    return random.key(69)

@pytest.fixture
def indexer() -> CBDIndexer:
    return CBDIndexer(10)

@pytest.fixture
def model() -> CBDModel:
    return CBDModel()
