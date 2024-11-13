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
    return TestState(TestSetup.xtal(), TestSetup.lens(), TestSetup.z())

@pytest.fixture
def model(state: TestState) -> TestModel:
    init = cbc.jax.init_from_bounds(state, default=lambda val: 0.25 * val)
    return TestModel(init)

@pytest.fixture
def int_state(model: TestModel, state: TestState) -> cbc.jax.InternalState:
    return model.to_internal(state)
