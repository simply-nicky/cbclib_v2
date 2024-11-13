from typing import Any, Callable, Tuple
import pytest
import jax.numpy as jnp
from jax import random, jit
import cbclib_v2 as cbc
from cbclib_v2.annotations import IntArray, KeyArray
from cbclib_v2.test_util import check_gradient, TestSetup, TestModel, Criterion

class TestCBDModel():
    EPS: float = 5e-7

    @pytest.fixture
    def patterns(self, key: KeyArray, model: TestModel, int_state: cbc.jax.InternalState,
                 num_lines: int) -> cbc.jax.Patterns:
        keys = random.split(key, 4)
        center = model.lens.zero_order(int_state.lens)

        length = random.uniform(keys[0], (num_lines,), jnp.float32, 1.5e-3, 1.5e-2)
        x = random.uniform(keys[2], (num_lines,), jnp.float32,
                           TestSetup.roi[2] * TestSetup.x_pixel_size,
                           TestSetup.roi[3] * TestSetup.x_pixel_size)
        y = random.uniform(keys[3], (num_lines,), jnp.float32,
                           TestSetup.roi[0] * TestSetup.y_pixel_size,
                           TestSetup.roi[1] * TestSetup.y_pixel_size)
        phi = jnp.arctan2(y - center[1], x - center[0])
        angles = phi + jnp.pi / 2 + random.uniform(keys[1], (num_lines,), jnp.float32,
                                                   -jnp.pi / 50, jnp.pi / 50)

        lines = jnp.stack((x - 0.5 * length * jnp.cos(angles),
                           y - 0.5 * length * jnp.sin(angles),
                           x + 0.5 * length * jnp.cos(angles),
                           y + 0.5 * length * jnp.sin(angles)), axis=-1)
        index = jnp.concatenate((jnp.full((num_lines // 2), 0),
                                 jnp.full((num_lines - num_lines // 2), 1)))
        return cbc.jax.Patterns(lines=lines, index=index)

    @pytest.fixture
    def offsets(self, patterns: cbc.jax.Patterns, model: TestModel,
                int_state: cbc.jax.InternalState) -> IntArray:
        return model.init_offsets(patterns, int_state)

    @pytest.fixture
    def dynamic_sampler(self, key: KeyArray, patterns: cbc.jax.Patterns, model: TestModel,
                       offsets: IntArray, num_points: int) -> cbc.jax.LaueSampler:
        return model.dynamic_sampler(key, patterns, offsets, num_points)

    @pytest.fixture
    def static_sampler(self, key: KeyArray, patterns: cbc.jax.Patterns, model: TestModel,
                       offsets: IntArray, num_points: int, int_state: cbc.jax.InternalState
                       ) -> cbc.jax.LaueSampler:
        return model.static_sampler(key, patterns, offsets, num_points, int_state)

    @pytest.fixture
    def dynamic_criterion(self, model: TestModel, dynamic_sampler: cbc.jax.LaueSampler,
                          num_lines: int) -> Criterion:
        return jit(model.criterion(dynamic_sampler, model.pupil_projector, num_lines))

    @pytest.fixture
    def static_criterion(self, model: TestModel, static_sampler: cbc.jax.LaueSampler,
                          num_lines: int) -> Criterion:
        return jit(model.criterion(static_sampler, model.line_projector, num_lines))

    def check_gradient(self, f: Callable, args: Tuple[Any, ...]):
        check_gradient(f, args, eps=self.EPS)

    @pytest.mark.slow
    @pytest.mark.parametrize('num_lines,num_points', [(200, 4)])
    def test_model_gradients(self, key: KeyArray, static_criterion: Criterion,
                             dynamic_criterion: Criterion, model: TestModel):
        self.check_gradient(static_criterion, (model.init(key),))
        self.check_gradient(dynamic_criterion, (model.init(key),))
