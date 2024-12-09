import numpy as np
import pytest
from jax import random
import cbclib_v2 as cbc
from cbclib_v2.annotations import KeyArray
from cbclib_v2.test_util import TestSetup, TestModel, TestState

@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(69)

@pytest.fixture
def key() -> KeyArray:
    return random.key(69)

@pytest.fixture
def state() -> TestState:
    return TestState(TestSetup.lens(), TestSetup.xtal(), TestSetup.z())

@pytest.fixture
def model() -> TestModel:
    return TestModel()

@pytest.fixture
def int_state(model: TestModel, state: TestState) -> cbc.jax.InternalState:
    return model.to_internal(state)
