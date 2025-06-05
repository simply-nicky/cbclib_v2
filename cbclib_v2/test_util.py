from typing import Any, Callable, Dict, Optional, Tuple, overload
import numpy as np
from jax import tree
from jax.test_util import check_grads
from .indexer import (BaseState, CBData, Detector, FixedLens, FixedPupilLens,
                      FixedPupilSetup, FixedSetup, XtalState, random_state)
from ._src.annotations import ArrayNamespace, ComplexArray, JaxNumPy, RealArray
from ._src.state import State, field

REL_TOL = 0.025

class TestSetup():
    basis           = [[[-0.00088935, -0.00893378, -0.00057904],
                        [ 0.00736113, -0.0003915 , -0.00511673],
                        [ 0.01050942, -0.00199355,  0.01527804]],
                       [[-0.0008358 , -0.00893367, -0.00065555],
                        [ 0.00777905, -0.00039116, -0.00445575],
                        [ 0.00913779, -0.00199602,  0.01613559]]]
    foc_pos         = ( 0.14292289,  0.16409828, -0.39722229)
    roi             = (1100, 3260, 1040, 3108)
    pupil_roi       = (0.16583517, 0.17700936, 0.14640569, 0.15699476)
    smp_dist        = 0.006571637911728528
    x_pixel_size    = 7.5e-05
    y_pixel_size    = 7.5e-05

    @classmethod
    def xtal(cls, xp: ArrayNamespace) -> XtalState:
        return XtalState(xp.array(cls.basis))

    @classmethod
    def detector(cls) -> Detector:
        return Detector(cls.x_pixel_size, cls.y_pixel_size)

    @classmethod
    def fixed_lens(cls) -> FixedLens:
        return FixedLens(cls.foc_pos, cls.pupil_roi)

    @classmethod
    def fixed_pupil_lens(cls, xp: ArrayNamespace) -> FixedPupilLens:
        return FixedPupilLens(xp.asarray(cls.foc_pos), cls.pupil_roi)

    @classmethod
    def fixed_setup(cls, size: int=1) -> FixedSetup:
        return FixedSetup(cls.fixed_lens(), cls.z(size=size))

    @classmethod
    def fixed_pupil_setup(cls, xp: ArrayNamespace, size: int=1) -> FixedPupilSetup:
        return FixedPupilSetup(cls.fixed_pupil_lens(xp), cls.z(xp, size))

    @overload
    @classmethod
    def z(cls, xp: None=None, size: int=1) -> Tuple[float, ...]: ...

    @overload
    @classmethod
    def z(cls, xp: ArrayNamespace, size: int=1) -> RealArray: ...

    @classmethod
    def z(cls, xp: ArrayNamespace | None=None, size: int=1) -> Tuple[float, ...] | RealArray:
        if xp is None:
            return tuple([cls.smp_dist + cls.foc_pos[2],] * size)
        return xp.array([cls.smp_dist + cls.foc_pos[2],] * size)

class FixedState(BaseState, State):
    xtal    : XtalState
    setup   : FixedSetup = field(default=TestSetup.fixed_setup(), static=True)

random_xtal = random_state(TestSetup.xtal(JaxNumPy),
                           tree.map(lambda val: REL_TOL * val, TestSetup.xtal(JaxNumPy)))
random_setup = random_state(TestSetup.fixed_pupil_setup(JaxNumPy),
                            tree.map(lambda val: REL_TOL * val, TestSetup.fixed_pupil_setup(JaxNumPy)))

class FullState(BaseState, State, random=True):
    xtal    : XtalState = field(random=random_xtal)
    setup   : FixedPupilSetup = field(random=random_setup)

Criterion = Callable[[CBData, BaseState,], RealArray]

_atol = {np.dtype(np.float32): 1e-4, np.dtype(np.float64): 1e-5,
         np.dtype(np.complex64): 1e-4, np.dtype(np.complex128): 1e-5}
_rtol = {np.dtype(np.float32): 1e-3, np.dtype(np.float64): 1e-4,
         np.dtype(np.complex64): 1e-3, np.dtype(np.complex128): 1e-4}

def default_tolerance() -> Dict[np.dtype, float]:
    return _atol

def tolerance(dtype: np.dtype, tol: float | None=None) -> float:
    if tol is None:
        return default_tolerance()[dtype]
    return default_tolerance().get(dtype, tol)

def default_gradient_tolerance() -> Dict[np.dtype, float]:
    return _rtol

def gradient_tolerance(dtype: np.dtype, tol: float | None=None) -> float:
    if tol is None:
        return default_gradient_tolerance()[dtype]
    return default_gradient_tolerance().get(dtype, tol)

def check_close(a: RealArray | ComplexArray, b: RealArray | ComplexArray,
                rtol: float | None=None, atol: float | None=None):
    if rtol is None:
        rtol = max(gradient_tolerance(a.dtype, rtol),
                   gradient_tolerance(b.dtype, rtol))
    if atol is None:
        atol = max(tolerance(a.dtype, atol), tolerance(b.dtype, atol))
    np.testing.assert_allclose(a, b, rtol=rtol, atol=atol)

def check_gradient(f: Callable, args: Any, atol: float | None=None, rtol: float | None=None,
                   eps: float | None=None, **static_args: Any):
    def wrapper(*args):
        return f(*args, **static_args)
    check_grads(wrapper, args, order=1, modes='rev', atol=atol, rtol=rtol, eps=eps)
