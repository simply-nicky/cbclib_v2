from typing import Any, Callable, Dict, Optional, Union
import numpy as np
import jax.numpy as jnp
from jax.test_util import check_grads
from .jax import CBDModel, InternalState, LensState, XtalState, jax_dataclass
from ._src.annotations import ComplexArray, KeyArray, RealArray

class TestSetup():
    basis           = [[[-0.00088935, -0.00893378, -0.00057904],
                        [ 0.00736113, -0.0003915 , -0.00511673],
                        [ 0.01050942, -0.00199355,  0.01527804]],
                       [[-0.0008358 , -0.00893367, -0.00065555],
                        [ 0.00777905, -0.00039116, -0.00445575],
                        [ 0.00913779, -0.00199602,  0.01613559]]]
    foc_pos         = [ 0.14292289,  0.16409828, -0.39722229]
    roi             = (1100, 3260, 1040, 3108)
    pupil_roi       = (0.16583517, 0.17700936, 0.14640569, 0.15699476)
    smp_dist        = 0.006571637911728528
    x_pixel_size    = 7.5e-05
    y_pixel_size    = 7.5e-05

    @classmethod
    def xtal(cls) -> XtalState:
        return XtalState(jnp.array(cls.basis))

    @classmethod
    def lens(cls) -> LensState:
        return LensState(jnp.asarray(cls.foc_pos), cls.pupil_roi)

    @classmethod
    def z(cls) -> RealArray:
        return jnp.full((len(cls.basis),), cls.smp_dist + cls.foc_pos[2])

TestState = InternalState
Criterion = Callable[[TestState,], RealArray]

@jax_dataclass
class TestModel(CBDModel):
    init_state  : Callable[[KeyArray,], TestState]

    def init(self, rng) -> TestState:
        return self.init_state(rng)

    def to_internal(self, state: TestState) -> InternalState:
        return state

_atol = {np.dtype(np.float32): 1e-4, np.dtype(np.float64): 1e-5,
         np.dtype(np.complex64): 1e-4, np.dtype(np.complex128): 1e-5}
_rtol = {np.dtype(np.float32): 1e-3, np.dtype(np.float64): 1e-4,
         np.dtype(np.complex64): 1e-3, np.dtype(np.complex128): 1e-4}

def default_tolerance() -> Dict[np.dtype, float]:
    return _atol

def tolerance(dtype: np.dtype, tol: Optional[float]=None) -> float:
    if tol is None:
        return default_tolerance()[dtype]
    return default_tolerance().get(dtype, tol)

def default_gradient_tolerance() -> Dict[np.dtype, float]:
    return _rtol

def gradient_tolerance(dtype: np.dtype, tol: Optional[float]=None) -> float:
    if tol is None:
        return default_gradient_tolerance()[dtype]
    return default_gradient_tolerance().get(dtype, tol)

def check_close(a: Union[RealArray, ComplexArray], b: Union[RealArray, ComplexArray],
                rtol: Optional[float]=None, atol: Optional[float]=None):
    if rtol is None:
        rtol = max(gradient_tolerance(a.dtype, rtol),
                   gradient_tolerance(b.dtype, rtol))
    if atol is None:
        atol = max(tolerance(a.dtype, atol), tolerance(b.dtype, atol))
    np.testing.assert_allclose(a, b, rtol=rtol, atol=atol)

def check_gradient(f: Callable, args: Any, atol: Optional[float]=None, rtol: Optional[float]=None,
                   eps: Optional[float]=None, **static_args: Any):
    def wrapper(*args):
        return f(*args, **static_args)
    check_grads(wrapper, args, order=1, modes='rev', atol=atol, rtol=rtol, eps=eps)
