from typing import Any, Callable, Tuple
import pytest
import jax.numpy as jnp
from jax import config
from jax.test_util import check_grads
import cbclib_v2 as cbc
from cbclib_v2.jax import jax_dataclass
from cbclib_v2.annotations import KeyArray, RealArray

config.update("jax_enable_x64", True)

class Parameters():
    ChainState = Tuple[cbc.jax.RotationState, ...]

    a_vec           = [-0.00093604, -0.00893389, -0.00049815]
    b_vec           = [ 0.00688718, -0.00039195, -0.00573877]
    c_vec           = [ 0.01180108, -0.00199115,  0.01430421]
    foc_pos         = [ 0.14292289,  0.16409828, -0.39722229]
    frames          = (10, 20, 30)
    roi             = (1100, 3260, 1040, 3108)
    pupil_roi       = [0.16583517, 0.17700936, 0.14640569, 0.15699476]
    rot_axis        = [ 1.57123908, -1.56927065]
    rotations       = [[[ 9.9619448e-01,  3.2783555e-05, -8.7157689e-02],
                        [-4.4395445e-05,  1.0000000e+00, -1.3129010e-04],
                        [ 8.7157689e-02,  1.3465989e-04,  9.9619448e-01]],
                       [[ 9.8480743e-01,  5.3705335e-05, -1.7364986e-01],
                        [-1.0006334e-04,  9.9999994e-01, -2.5820805e-04],
                        [ 1.7364983e-01,  2.7166118e-04,  9.8480743e-01]],
                       [[ 9.6592510e-01,  6.2607127e-05, -2.5882185e-01],
                        [-1.6658218e-04,  9.9999988e-01, -3.7979320e-04],
                        [ 2.5882179e-01,  4.0996689e-04,  9.6592498e-01]]]
    smp_dist        = 0.006571637911728528
    dist            = -0.3906506481396024
    wavelength      = 7.09291721831675e-11
    x_pixel_size    = 7.5e-05
    y_pixel_size    = 7.5e-05

    @classmethod
    def xtal(cls) -> cbc.jax.XtalState:
        return cbc.jax.XtalState(jnp.stack([jnp.asarray(cls.a_vec),
                                            jnp.asarray(cls.b_vec),
                                            jnp.asarray(cls.c_vec)]))

    @classmethod
    def lens(cls) -> cbc.jax.LensState:
        return cbc.jax.LensState(jnp.asarray(cls.foc_pos), jnp.asarray(cls.pupil_roi))

    @classmethod
    def pixel_size(cls) -> Tuple[float, float]:
        return (cls.x_pixel_size, cls.y_pixel_size)

    @classmethod
    def tilt(cls) -> cbc.jax.TiltAxisState:
        theta, phi = cls.rot_axis
        axis = jnp.array([jnp.sin(theta) * jnp.cos(phi),
                          jnp.sin(theta) * jnp.sin(phi), jnp.cos(theta)])
        return cbc.jax.TiltAxisState(angles=-jnp.pi / 360 * jnp.array(cls.frames), axis=axis)

    @classmethod
    def transforms(cls) -> ChainState:
        return tuple(cbc.jax.RotationState(jnp.asarray(mat)) for mat in cls.rotations)

    @classmethod
    def z(cls) -> RealArray:
        return jnp.array(cls.smp_dist + cls.foc_pos[2])

@jax_dataclass
class TestState:
    transform       : cbc.jax.TiltAxisState
    xtal            : cbc.jax.XtalState
    lens            : cbc.jax.LensState
    z               : RealArray

@jax_dataclass
class TestModel(cbc.jax.CBDModel):
    init_state  : Callable[[KeyArray,], TestState]
    transform   : cbc.jax.TiltAxis = cbc.jax.TiltAxis()

    def init(self, rng) -> TestState:
        return self.init_state(rng)

    def to_internal(self, state: TestState) -> cbc.jax.InternalState:
        xtal = self.transform.apply(state.xtal, state.transform)
        z = state.z * jnp.ones(xtal.num)
        return cbc.jax.InternalState(xtal, state.lens, z)

class TestCBDModel():
    REL_TOL: float = 0.05
    Q_ABS: float = 0.25
    EPS: float = 1e-11

    @pytest.fixture(params=[1.0])
    def width(self, request: pytest.FixtureRequest) -> float:
        return request.param

    @pytest.fixture
    def model(self) -> TestModel:
        state = TestState(Parameters.tilt(), Parameters.xtal(), Parameters.lens(), Parameters.z())
        bounds = {'transform': {'angles': jnp.full(3, 1.0 / 180 * jnp.pi), 'axis': jnp.full(3, 0.05)}}
        init = cbc.jax.init_from_bounds(state, bounds=bounds, default=lambda val: self.REL_TOL * val)
        return TestModel(init)

    @pytest.fixture
    def miller(self, model: TestModel, jax_rng: KeyArray) -> cbc.jax.MillerIndices:
        return model.init_miller(self.Q_ABS, model.to_internal(model.init(jax_rng)))

    @pytest.fixture
    def criterion(self, model: TestModel, miller: cbc.jax.MillerIndices, width: float
                  ) -> Callable[[TestState,], RealArray]:

        def wrapper(state: TestState) -> RealArray:
            int_state = model.to_internal(state)
            streaks = model.init_streaks(miller, False, Parameters.pixel_size(), int_state)
            return jnp.mean(streaks.length)

        return wrapper

    def check_gradient(self, f: Callable, args: Tuple[Any, ...]):
        def wrapper(*args):
            return f(*args)
        check_grads(wrapper, args, order=1, modes='rev', eps=self.EPS)

    @pytest.mark.slow
    def test_model_gradients(self, criterion: Callable[[TestState,], RealArray], model: TestModel,
                             jax_rng: KeyArray):
        self.check_gradient(criterion, (model.init(jax_rng),))
