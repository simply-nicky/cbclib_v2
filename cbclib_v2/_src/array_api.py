from typing import TYPE_CHECKING, Any, Literal, NamedTuple, Tuple, Set, TypeVar, overload, cast
from jax import devices, device_put, dlpack as jdl, random
import numpy as np
from array_api_compat import array_namespace as get_array_namespace, device
from .annotations import (AnyFloat, Array, ArrayLike, AnyNamespace, ArrayNamespace, CPArray,
                          CPIntArray, CuPy, DTypeLike, Generator, IntArray, IntSequence, JaxArray,
                          JaxDevice, JaxIntArray, JaxNumPy, MultiIndices, NDArray, NDIntArray, NumPy,
                          RealArray, RealSequence, Scalar, Shape, ShapeLike, SupportsNamespace)

if TYPE_CHECKING:
    from .data_container import ArrayContainer

if CuPy is not None or TYPE_CHECKING:
    import cupy as cp
    from cupy import fromDlpack as from_dlpack

    class CuPyGenerator(cp.random.RandomState):
        def integers(self, low: int | CPIntArray, high: int | None=None,
                    size: Shape | None=None, dtype: DTypeLike = np.int64) -> IntArray:
            return super().randint(low, high=high, size=size, dtype=cast(type[int], dtype))

        def random(self, size: Shape | None=None, dtype: DTypeLike = np.float64) -> RealArray:
            if size is None:
                return super().rand(1, dtype=dtype)
            return super().rand(*size, dtype=dtype)

    def ascupy(array: Array) -> CPArray:
        """Convert an array to a CuPy array. If the input array is already a CuPy array,
        it will be returned as is.

        Args:
            array: The input array to convert.

        Returns:
            A CuPy array containing the same data as the input array.
        """
        xp = array_namespace(array)
        if xp is JaxNumPy:
            x = device_put(array, device=devices("gpu")[0])
            return from_dlpack(jdl.to_dlpack(x))
        if xp is NumPy:
            return CuPy.asarray(array)
        if xp is CuPy:
            return CuPy.asarray(array)
        raise ValueError(f"Unsupported array namespace: {xp}")
else:
    cp = None  # type: ignore
    CuPyGenerator = None  # type: ignore

    def ascupy(array: Array) -> CPArray:
        """Convert an array to a CuPy array. If the input array is already a CuPy array,
        it will be returned as is.

        Args:
            array: The input array to convert.

        Returns:
            A CuPy array containing the same data as the input array.
        """
        raise ValueError("CuPy is not available")

def to_shape(shape: ShapeLike) -> Tuple[int, ...]:
    if isinstance(shape, (int, np.integer)):
        return (int(shape),)
    return tuple(int(ax) for ax in shape)

class JaxGenerator:
    def __init__(self, seed: int | None = None):
        self.key = random.PRNGKey(seed if seed is not None else 0)

    def beta(self, a: AnyFloat, b: AnyFloat, size: ShapeLike | None = None) -> JaxArray:
        if size is not None:
            size = to_shape(size)
        xp = JaxNumPy
        return random.beta(self.key, xp.asarray(a), xp.asarray(b), shape=size)

    def binomial(self, n: IntSequence, p: AnyFloat, size: ShapeLike | None = None) -> JaxArray:
        if size is not None:
            size = to_shape(size)
        xp = JaxNumPy
        return random.binomial(self.key, xp.asarray(n), xp.asarray(p), shape=size)

    def chisquare(self, df: AnyFloat, size: ShapeLike | None = None) -> JaxArray:
        if size is not None:
            size = to_shape(size)
        return random.chisquare(self.key, JaxNumPy.asarray(df), shape=size)

    def choice(self, a: IntSequence | RealSequence | ArrayLike, size: ShapeLike | None = None,
               replace: bool = True, p: RealSequence | ArrayLike | None = None) -> JaxArray:
        if size is None:
            size = (1,)
        xp = JaxNumPy
        if p is not None:
            p = xp.asarray(p)
        return random.choice(self.key, xp.asarray(a), shape=to_shape(size), replace=replace, p=p)

    def dirichlet(self, alpha: RealSequence, size: ShapeLike | None = None) -> JaxArray:
        if size is not None:
            size = to_shape(size)
        return random.dirichlet(self.key, JaxNumPy.asarray(alpha), shape=size)

    def exponential(self, scale: AnyFloat = 1.0, size: ShapeLike | None = None) -> JaxArray:
        if size is None:
            size = (1,)
        xp = JaxNumPy
        return random.exponential(self.key, shape=to_shape(size)) * xp.asarray(scale)

    def f(self, dfnum: AnyFloat, dfden: AnyFloat, size: ShapeLike | None = None) -> JaxArray:
        if size is not None:
            size = to_shape(size)
        xp = JaxNumPy
        return random.f(self.key, xp.asarray(dfnum), xp.asarray(dfden), shape=size)

    def gamma(self, shape: AnyFloat, scale: AnyFloat = 1.0, size: ShapeLike | None = None
              ) -> JaxArray:
        if size is not None:
            size = to_shape(size)
        xp = JaxNumPy
        return random.gamma(self.key, xp.asarray(shape), shape=size) * xp.asarray(scale)

    def geometric(self, p: AnyFloat, size: ShapeLike | None = None) -> JaxArray:
        if size is not None:
            size = to_shape(size)
        return random.geometric(self.key, JaxNumPy.asarray(p), shape=size)

    def integers(self, low: IntSequence, high: IntSequence | None = None,
                 size: ShapeLike | None = None, dtype: DTypeLike = np.int64) -> JaxArray:
        if size is None:
            size = (1,)
        if high is None:
            low, high = 0, low
        xp = JaxNumPy
        return random.randint(self.key, to_shape(size), xp.asarray(low), xp.asarray(high),
                              dtype=dtype)

    def laplace(self, loc: AnyFloat = 0.0, scale: AnyFloat = 1.0,
                size: ShapeLike | None = None) -> JaxArray:
        if size is None:
            size = (1,)
        xp = JaxNumPy
        laplace = random.laplace(self.key, shape=to_shape(size))
        return laplace * xp.asarray(scale) + xp.asarray(loc)

    def logistic(self, loc: AnyFloat = 0.0, scale: AnyFloat = 1.0,
                 size: ShapeLike | None = None) -> JaxArray:
        if size is None:
            size = (1,)
        xp = JaxNumPy
        logistic = random.logistic(self.key, shape=to_shape(size))
        return logistic * xp.asarray(scale) + xp.asarray(loc)

    def lognormal(self, mean: AnyFloat = 0.0, sigma: AnyFloat = 1.0, size: ShapeLike | None = None
                  ) -> JaxArray:
        if size is None:
            size = (1,)
        xp = JaxNumPy
        normal = random.normal(self.key, shape=to_shape(size))
        return xp.exp(normal * xp.asarray(sigma) + xp.asarray(mean))

    def multivariate_normal(self, mean: RealSequence, cov: ArrayLike,
                            size: ShapeLike | None = None) -> JaxArray:
        if size is not None:
            size = to_shape(size)
        xp = JaxNumPy
        return random.multivariate_normal(self.key, xp.asarray(mean),
                                          xp.asarray(cov), shape=size)

    def normal(self, loc: AnyFloat = 0.0, scale: AnyFloat = 1.0,
               size: ShapeLike | None = None) -> JaxArray:
        if size is None:
            size = (1,)
        xp = JaxNumPy
        return random.normal(self.key, shape=to_shape(size)) * xp.asarray(scale) + xp.asarray(loc)

    def pareto(self, a: AnyFloat, size: ShapeLike | None = None) -> JaxArray:
        if size is not None:
            size = to_shape(size)
        xp = JaxNumPy
        return random.pareto(self.key, xp.asarray(a), shape=size)

    def permutation(self, x: IntSequence | RealSequence | ArrayLike) -> JaxArray:
        xp = JaxNumPy
        return random.permutation(self.key, xp.asarray(x))

    def poisson(self, lam: AnyFloat = 1.0, size: ShapeLike | None = None) -> JaxArray:
        if size is not None:
            size = to_shape(size)
        xp = JaxNumPy
        return random.poisson(self.key, xp.asarray(lam), shape=size)

    def random(self, size: ShapeLike | None = None, dtype: DTypeLike = np.float64
               ) -> JaxArray:
        if size is None:
            size = (1,)
        return random.uniform(self.key, shape=to_shape(size), dtype=dtype)

    def rayleigh(self, scale: AnyFloat = 1.0, size: ShapeLike | None = None) -> JaxArray:
        if size is None:
            size = (1,)
        return random.rayleigh(self.key, JaxNumPy.asarray(scale), shape=to_shape(size))

    def standard_exponential(self, size: ShapeLike | None = None, dtype: DTypeLike = np.float64
                             ) -> JaxArray:
        if size is None:
            size = (1,)
        return random.exponential(self.key, shape=to_shape(size), dtype=dtype)

    def standard_gamma(self, shape: AnyFloat, size: ShapeLike | None = None,
                       dtype: DTypeLike = np.float64) -> JaxArray:
        if size is None:
            size = (1,)
        return random.gamma(self.key, JaxNumPy.asarray(shape), shape=to_shape(size), dtype=dtype)

    def standard_normal(self, size: ShapeLike | None = None, dtype: DTypeLike = np.float64
                        ) -> JaxArray:
        if size is None:
            size = (1,)
        return random.normal(self.key, shape=to_shape(size), dtype=dtype)

    def uniform(self, low: AnyFloat = 0.0, high: AnyFloat = 1.0, size: ShapeLike | None = None
                ) -> JaxArray:
        if size is None:
            size = (1,)
        xp = JaxNumPy
        return random.uniform(self.key, shape=to_shape(size), minval=xp.asarray(low),
                              maxval=xp.asarray(high))

@overload
def add_at(a: NDArray, indices: IntArray | Tuple[IntArray, ...], b: Array | Scalar) -> NDArray: ...

@overload
def add_at(a: JaxArray, indices: IntArray | Tuple[IntArray, ...], b: Array | Scalar
           ) -> JaxArray: ...

@overload
def add_at(a: CPArray, indices: IntArray | Tuple[IntArray, ...], b: Array | Scalar) -> CPArray: ...

def add_at(a: Array, indices: IntArray | Tuple[IntArray, ...], b: Array | Scalar) -> Array:
    """Perform unbuffered in-place addition of `b` to `a` at the specified `indices`. This
    function works with all supported array APIs (NumPy, JAX, CuPy).

    Args:
        a: The input array to which values will be added.
        indices: The indices at which to add the values from `b`. This can be a single array
            of indices or a tuple of arrays for multi-dimensional indexing.
        b: The values to add to `a` at the specified indices. This can be a scalar or an array
            of values to add.

    Returns:
        An array with the same shape and type as `a`, where the values from `b` have been added
        to `a` at the specified `indices`.
    """
    xp = array_namespace(a)

    if xp is JaxNumPy:
        return JaxNumPy.asarray(a).at[indices].add(b)
    if xp is NumPy:
        np.add.at(np.asarray(a), indices, b)
        return a
    if CuPy is not None and xp is CuPy:
        a[indices] += b
        return a
    raise ValueError(f"Unsupported array namespace: {xp}")

@overload
def argmin_at(indices: IntArray, values: NDArray, size: int) -> NDIntArray: ...

@overload
def argmin_at(indices: IntArray, values: JaxArray, size: int) -> JaxIntArray: ...

@overload
def argmin_at(indices: IntArray, values: CPArray, size: int) -> CPIntArray: ...

def argmin_at(indices: IntArray, values: RealArray, size: int) -> IntArray:
    """Return packed indices of the minimum value in each indexed group.

    Args:
        indices: Group index for every value, with shape ``(N,)``.
        values: Values to minimize, with shape ``(N,)``.
        size: Number of groups in the output.

    Returns:
        Absolute indices into ``values``, with shape ``(size,)``. Ties select
        the first occurrence.

    Raises:
        ValueError: If any group has no values.
    """
    xp = array_namespace(values)
    candidate_id = xp.arange(values.size)
    sentinel = values.size

    minima = min_at(
        xp.full((size,), xp.inf, dtype=values.dtype),
        indices,
        values,
    )
    matching_id = xp.where(
        values == minima[indices],
        candidate_id,
        sentinel,
    )
    result = min_at(
        xp.full((size,), sentinel, dtype=int),
        indices,
        matching_id,
    )

    if xp.any(result == sentinel):
        raise ValueError('Cannot calculate argmin for an empty group')

    return result

@overload
def min_at(a: NDArray, indices: IntArray | Tuple[IntArray, ...], b: Array) -> NDArray: ...

@overload
def min_at(a: JaxArray, indices: IntArray | Tuple[IntArray, ...], b: Array) -> JaxArray: ...

@overload
def min_at(a: CPArray, indices: IntArray | Tuple[IntArray, ...], b: Array) -> CPArray: ...

def min_at(a: Array, indices: IntArray | Tuple[IntArray, ...], b: Array) -> Array:
    """Perform unbuffered in-place minimum of `b` and `a` at the specified `indices`. This
    function works with all supported array APIs (NumPy, JAX, CuPy).

    Args:
        a: The input array to which values will be compared.
        indices: The indices at which to compare the values from `b`. This can be a single array
            of indices or a tuple of arrays for multi-dimensional indexing.
        b: The values to compare with `a` at the specified indices. This can be a scalar or an array
            of values to compare.

    Returns:
        An array with the same shape and type as `a`, where the minimum values between `a` and `b`
        have been set at the specified `indices`.
    """
    xp = array_namespace(a)

    if xp is JaxNumPy:
        return JaxNumPy.asarray(a).at[indices].min(b)
    if xp is NumPy:
        np.minimum.at(np.asarray(a), indices, b)
        return a
    if CuPy is not None and xp is CuPy:
        cp.minimum.at(cp.asarray(a), indices, b)
        return a
    raise ValueError(f'Unsupported array namespace: {xp}')

@overload
def set_at(a: NDArray, indices: MultiIndices, b: Array | Scalar) -> NDArray: ...

@overload
def set_at(a: JaxArray, indices: MultiIndices, b: Array | Scalar
           ) -> JaxArray: ...

@overload
def set_at(a: CPArray, indices: MultiIndices, b: Array | Scalar) -> CPArray: ...

def set_at(a: Array, indices: MultiIndices, b: Array | Scalar) -> Array:
    """Perform unbuffered in-place assignment of `b` to `a` at the specified `indices`. This
    function works with all supported array APIs (NumPy, JAX, CuPy).

    Args:
        a: The input array to which values will be assigned.
        indices: The indices at which to assign the values from `b`. This can be a single array
            of indices or a tuple of arrays for multi-dimensional indexing.
        b: The values to assign to `a` at the specified indices. This can be a scalar or an array
            of values to assign.

    Returns:
        An array with the same shape and type as `a`, where the values from `b` have been assigned
        to `a` at the specified `indices`.
    """
    xp = array_namespace(a)

    if xp is JaxNumPy:
        return JaxNumPy.asarray(a).at[indices].set(b)
    if xp is NumPy:
        a[indices] = b
        return a
    if CuPy is not None and xp is CuPy:
        a[indices] = b
        return a
    raise ValueError(f"Unsupported array namespace: {xp}")

def default_rng(seed: int | None = None, xp: ArrayNamespace = NumPy) -> Generator:
    """Return a random number generator for the specified array API.

    This function provides a unified interface for creating random number
    generators across different array APIs (NumPy, JAX, CuPy).

    Args:
        seed: An optional seed for the random number generator. If `None`, a
            default seed will be used.
        xp: The array namespace for which to create the random number
            generator. This can be `NumPy`, `JaxNumPy`, or `CuPy`.

    Returns:
        A random number generator for the specified array namespace.
    """
    if xp is JaxNumPy:
        return cast(Generator, JaxGenerator(seed))
    if xp is NumPy:
        return cast(Generator, np.random.default_rng(seed))
    if CuPy is not None and xp is CuPy:
        return cast(Generator, CuPyGenerator(seed))
    raise ValueError(f"Unsupported array namespace: {xp}")

@overload
def array_namespace(*arrays: NDArray) -> ArrayNamespace[NDArray]: ...

@overload
def array_namespace(*arrays: JaxArray) -> ArrayNamespace[JaxArray]: ...

@overload
def array_namespace(*arrays: CPArray) -> ArrayNamespace[CPArray]: ...

@overload
def array_namespace(*arrays: SupportsNamespace | Any) -> AnyNamespace: ...

def array_namespace(*arrays: SupportsNamespace | Array | Any) -> AnyNamespace:
    """Determine the array API namespace of the given arrays. This function checks the underlying
    array types to determine which array API they belong to (NumPy, JAX, CuPy). If multiple arrays
    are provided, the precedence order for determining the namespace is CuPy > JAX > NumPy.

    Args:
        *arrays: One or more arrays for which to determine the array API namespace. These can be
            arrays from any supported array API (NumPy, JAX, CuPy) or objects that implement the
            `__array_namespace__` method.

    Returns:
        The array API namespace that the input arrays belong to. This will be one of `NumPy`,
        `JaxNumPy`, or `CuPy`.
    """
    def namespaces(*arrays: SupportsNamespace | Array | Any) -> Set[Any]:
        result = set()
        for array in arrays:
            if isinstance(array, dict):
                result |= namespaces(*array.values())
            elif isinstance(array, Array):
                result.add(get_array_namespace(array))
            elif isinstance(array, SupportsNamespace):
                result.add(array.__array_namespace__())
        return result

    nspaces = namespaces(*arrays)
    if len(nspaces) == 0:
        raise ValueError("namespace set should not be empty")
    if CuPy is not None:
        if any("cupy" in getattr(nspace, "__name__", "") for nspace in nspaces):
            return CuPy
    if any(getattr(nspace, "__name__", "").startswith("jax.numpy") for nspace in nspaces):
        return JaxNumPy
    if any(getattr(nspace, "__name__", "") == "numpy" or
           getattr(nspace, "__name__", "").startswith("array_api_compat.numpy")
           for nspace in nspaces):
        return NumPy
    raise ValueError(f"The array namespace {nspaces.pop()} is not supported")

Platform = Literal['cpu', 'gpu']

def get_platform(array: Array) -> Platform:
    """Determine the platform (CPU or GPU) on which the given array is located. This function
    checks the array's API and device information to determine whether it is on the CPU or
    GPU.

    Args:
        array: The input array for which to determine the platform. This can be an array from
            any supported array API (NumPy, JAX, CuPy).

    Returns:
        A string indicating the platform on which the array is located. This will be either
        'cpu' or 'gpu'.
    """
    xp = array_namespace(array)
    if xp is JaxNumPy:
        dev : JaxDevice = device(array)
        return 'gpu' if dev.platform == 'gpu' else 'cpu'
    if xp is NumPy:
        return 'cpu'
    if CuPy is not None and xp is CuPy:
        return 'gpu'
    raise ValueError(f"Unsupported array namespace: {xp}")

def asnumpy(array: Array) -> NDArray:
    """Convert an array to a NumPy array. If the input array is already a NumPy array, it will be
    returned as is.

    Args:
        array: The input array to convert. This can be an array from any supported array API
            (NumPy, JAX, CuPy).

    Returns:
        A NumPy array containing the same data as the input array.
    """
    xp = array_namespace(array)
    if xp is JaxNumPy:
        return np.asarray(array)
    if xp is NumPy:
        return np.asarray(array)
    if CuPy is not None and xp is CuPy:
        return cp.asnumpy(array)
    raise ValueError(f"Unsupported array namespace: {xp}")

def asjax(array: Array) -> JaxArray:
    """Convert an array to a JAX array. If the input array is already a JAX array, it will be
    returned as is.

    Args:
        array: The input array to convert. This can be an array from any supported array API
            (NumPy, JAX, CuPy).

    Returns:
        A JAX array containing the same data as the input array.
    """
    xp = array_namespace(array)

    if xp is JaxNumPy:
        return JaxNumPy.asarray(array)
    if xp is NumPy:
        return JaxNumPy.asarray(array)
    if CuPy is not None and xp is CuPy:
        return jdl.from_dlpack(cp.asarray(array).toDlpack())
    raise ValueError(f"Unsupported array namespace: {xp}")

def default_api(platform: Platform) -> AnyNamespace:
    """Get the default array API namespace for a given platform.

    Args:
        platform: The platform for which to get the default array API namespace. This can be
            either 'cpu' or 'gpu'.

    Returns:
        The default array API namespace for the specified platform. This will be either `NumPy`
        for CPU or `CuPy` for GPU.
    """
    if platform == 'cpu':
        return NumPy
    if platform == 'gpu':
        if CuPy is None:
            raise ValueError("CuPy is not available, cannot use GPU platform")
        return CuPy
    raise ValueError(f"Unsupported platform: {platform}")

def safe_divide(x: Array, y: Array, xp: AnyNamespace) -> Array:
    _y = xp.where(y == 0.0, 1.0, y)
    return xp.where(y == 0.0, 0.0, x / _y)

def safe_log(x: Array, xp: AnyNamespace) -> Array:
    _x = xp.where(x <= 0.0, 1.0, x)
    return xp.where(x <= 0.0, 0.0, xp.log(_x))

def safe_sqrt(x: Array, xp: AnyNamespace) -> Array:
    _x = xp.where(x <= 0.0, 0.0, x)
    return xp.where(x <= 0.0, 0.0, xp.sqrt(_x))

def kxy_to_k(kxy: RealArray, xp: AnyNamespace) -> RealArray:
    kz = safe_sqrt(1 - xp.sum(kxy[..., :2]**2, axis=-1), xp)
    return xp.stack((kxy[..., 0], kxy[..., 1], kz), axis=-1)

ArrCon_T = TypeVar('ArrCon_T', bound='ArrayContainer')

@overload
def broadcast_to(array: Array, idxs: IntArray, suffix: Shape, xp: AnyNamespace) -> Array: ...

@overload
def broadcast_to(array: ArrCon_T, idxs: IntArray, suffix: Shape, xp: AnyNamespace
                 ) -> ArrCon_T: ...

def broadcast_to(array: Array | ArrCon_T, idxs: IntArray, suffix: Shape, xp: AnyNamespace
                 ) -> Array | ArrCon_T:
    array = array.reshape((-1,) + suffix)
    if array.shape[0] == 1:
        return array[0]
    return array[xp.reshape(idxs, -1)].reshape(idxs.shape + suffix)

def euler_angles(rmats: RealArray, xp: AnyNamespace) -> RealArray:
    r"""Calculate Euler angles with Bunge convention [EUL]_.

    Args:
        rmats : A set of rotation matrices.

    Returns:
        A set of Euler angles with Bunge convention :math:`\phi_1, \Phi, \phi_2`.

    References:
        .. [EUL] Depriester, Dorian. (2018), "Computing Euler angles with Bunge convention from
                rotation matrix", 10.13140/RG.2.2.34498.48321/5.
    """
    beta = xp.acos(rmats[..., 2, 2])
    is_zero = xp.isclose(beta, 0)
    is_pi = xp.isclose(beta, xp.pi)
    alpha = xp.where(is_zero, xp.atan2(-rmats[..., 1, 0], rmats[..., 0, 0]), 0.0)
    alpha = xp.where(is_pi, xp.atan2(rmats[..., 1, 0], rmats[..., 0, 0]), alpha)
    alpha = xp.where(xp.invert(is_zero) & xp.invert(is_pi),
                     xp.atan2(rmats[..., 2, 0], -rmats[..., 2, 1]), alpha)
    gamma = xp.where(xp.invert(is_zero) & xp.invert(is_pi),
                     xp.atan2(rmats[..., 0, 2], rmats[..., 1, 2]), 0.0)
    alpha = xp.where(alpha < 0.0, alpha + 2 * xp.pi, alpha)
    gamma = xp.where(gamma < 0.0, gamma + 2 * xp.pi, gamma)
    return xp.stack((alpha, beta, gamma), axis=-1)

def euler_matrix(angles: RealArray, xp: AnyNamespace) -> RealArray:
    r"""Calculate rotation matrices from Euler angles with Bunge convention [EUL]_.

    Args:
        angles : Euler angles :math:`\phi_1, \Phi, \phi_2`.
        num_threads : Number of threads used in the calculations.

    Returns:
        A set of rotation matrices.
    """
    cos = xp.cos(angles)
    sin = xp.sin(angles)
    row0 = xp.stack([ cos[..., 0] * cos[..., 2] - sin[..., 0] * sin[..., 2] * cos[..., 1],
                      sin[..., 0] * cos[..., 2] + cos[..., 0] * sin[..., 2] * cos[..., 1],
                      sin[..., 2] * sin[..., 1]], axis=-1)
    row1 = xp.stack([-cos[..., 0] * sin[..., 2] - sin[..., 0] * cos[..., 2] * cos[..., 1],
                     -sin[..., 0] * sin[..., 2] + cos[..., 0] * cos[..., 2] * cos[..., 1],
                      cos[..., 2] * sin[..., 1]], axis=-1)
    row2 = xp.stack([ sin[..., 0] * sin[..., 1],
                     -cos[..., 0] * sin[..., 1],
                      cos[..., 1]], axis=-1)
    return xp.stack((row0, row1, row2), axis=-2)

def tilt_angles(rmats: RealArray, xp: AnyNamespace) -> RealArray:
    r"""Calculate an axis of rotation and a rotation angle for a rotation matrix.

    Args:
        rmats : A set of rotation matrices.

    Returns:
        A set of three angles :math:`\theta, \alpha, \beta`, a rotation angle :math:`\theta`, an
        angle between the axis of rotation and OZ axis :math:`\alpha`, and a polar angle of the
        axis of rotation :math:`\beta`.
    """
    # This transformation is accurate for proper rotations ONLY => det(rmats) == 1
    # from http://scipp.ucsc.edu/~haber/ph116A/rotation_11.pdf
    vec = xp.stack([rmats[..., 2, 1] - rmats[..., 1, 2],
                    rmats[..., 0, 2] - rmats[..., 2, 0],
                    rmats[..., 1, 0] - rmats[..., 0, 1]], axis=-1)
    rabs = safe_sqrt(xp.sum(vec**2, axis=-1), xp)
    return xp.stack([xp.atan2(rabs, xp.linalg.trace(rmats) - 1),
                     xp.acos(safe_divide(vec[..., 2], rabs, xp)),
                     xp.atan2(vec[..., 1], vec[..., 0])], axis=-1)

def tilt_matrix(angles: RealArray, xp: AnyNamespace) -> RealArray:
    r"""Calculate a rotation matrix for a set of three angles set of three angles :math:`\theta,
    \alpha, \beta`, a rotation angle :math:`\theta`, an angle between the axis of rotation and
    OZ axis :math:`\alpha`, and a polar angle of the axis of rotation :math:`\beta`.

    Args:
        angles : A set of angles :math:`\theta, \alpha, \beta`.

    Returns:
        A set of rotation matrices.
    """
    vec = xp.stack([ xp.cos(0.5 * angles[..., 0]),
                    -xp.sin(0.5 * angles[..., 0]) * xp.sin(angles[..., 1]) * xp.cos(angles[..., 2]),
                    -xp.sin(0.5 * angles[..., 0]) * xp.sin(angles[..., 1]) * xp.sin(angles[..., 2]),
                    -xp.sin(0.5 * angles[..., 0]) * xp.cos(angles[..., 1])], axis=-1)
    row0 = xp.stack([vec[..., 0]**2 + vec[..., 1]**2 - vec[..., 2]**2 - vec[..., 3]**2,
                     2 * (vec[..., 1] * vec[..., 2] + vec[..., 0] * vec[..., 3]),
                     2 * (vec[..., 1] * vec[..., 3] - vec[..., 0] * vec[..., 2])], axis=-1)
    row1 = xp.stack([2 * (vec[..., 1] * vec[..., 2] - vec[..., 0] * vec[..., 3]),
                     vec[..., 0]**2 + vec[..., 2]**2 - vec[..., 1]**2 - vec[..., 3]**2,
                     2 * (vec[..., 2] * vec[..., 3] + vec[..., 0] * vec[..., 1])], axis=-1)
    row2 = xp.stack([2 * (vec[..., 1] * vec[..., 3] + vec[..., 0] * vec[..., 2]),
                     2 * (vec[..., 2] * vec[..., 3] - vec[..., 0] * vec[..., 1]),
                     vec[..., 0]**2 + vec[..., 3]**2 - vec[..., 1]**2 - vec[..., 2]**2], axis=-1)
    return xp.stack((row0, row1, row2), axis=-2)

def det_to_k(pts: RealArray, src: RealArray, xp: AnyNamespace) -> RealArray:
    """Convert coordinates on the detector ``x`, ``y`` to wave-vectors originating from
    the source points ``src``.

    Args:
        x : x coordinates in pixels.
        y : y coordinates in pixels.
        src : Source points in meters (relative to the detector).

    Returns:
        A set of wave-vectors.
    """
    src = xp.expand_dims(src, axis=tuple(range(src.ndim - 1, pts.ndim - 1)))
    xy = pts - src[..., :2]
    norm = safe_sqrt(xp.sum(xy**2, axis=-1) + src[..., 2]**2, xp)
    vec = xp.concat((xy, xp.broadcast_to(-src[..., 2], pts.shape[:-1])[..., None]), axis=-1)
    return safe_divide(vec, norm[..., None], xp)

def k_to_det(k: RealArray, src: RealArray, xp: AnyNamespace) -> RealArray:
    """Convert wave-vectors originating from the source points ``src`` to coordinates on the
    detector.

    Args:
        k : An array of wave-vectors.
        src : Source points in meters (relative to the detector).
        idxs : Source point indices.

    Returns:
        A tuple of x and y coordinates in meters.
    """
    src = xp.expand_dims(src, axis=tuple(range(src.ndim - 1, k.ndim - 1)))
    slope = safe_divide(k[..., :2], k[..., 2, None], xp)
    pos = xp.where((k[..., 2] == 0)[..., None], 0.0, src[..., :2] - slope * src[..., 2, None])
    return pos

def k_to_smp(k: RealArray, defocus: RealArray, src: RealArray, xp: AnyNamespace) -> RealArray:
    """Convert wave-vectors originating from the source point ``src`` to sample
    planes at the ``defocus`` distance along z-axis.

    Args:
        k : An array of wave-vectors.
        src : Source point in meters (relative to the detector).
        defocus : Defocus distance along z-axis in meters (relative to the detector).
        idxs : Plane indices.

    Returns:
        An array of points belonging to the ``defocus`` planes.
    """
    src_xy = xp.expand_dims(src[..., :2], axis=tuple(range(src.ndim - 1, k.ndim - 1)))
    src_z = xp.expand_dims(src[..., 2], axis=tuple(range(src.ndim - 1, k.ndim - 1)))
    defocus = xp.expand_dims(defocus, axis=tuple(range(defocus.ndim, k.ndim - 1)))

    theta = safe_divide(k[..., :2], k[..., 2, None], xp)
    xy = src_xy + theta * defocus[..., None]
    z = xp.broadcast_to(src_z + defocus, xy.shape[:-1])
    return xp.stack((xy[..., 0], xy[..., 1], z), axis=-1)

def project_to_rect(point: RealArray, vmin: RealArray, vmax: RealArray,
                    xp: AnyNamespace) -> RealArray:
    vmin = xp.expand_dims(vmin, axis=tuple(range(vmin.ndim - 1, point.ndim - 1)))
    vmax = xp.expand_dims(vmax, axis=tuple(range(vmax.ndim - 1, point.ndim - 1)))
    return xp.clip(point, vmin, vmax)

class Projection(NamedTuple):
    center  : RealArray
    tau     : RealArray
    t       : RealArray

def project_to_streak(points: RealArray, pt0: RealArray, pt1: RealArray, xp: AnyNamespace) -> Projection:
    tau = pt1 - pt0
    center = 0.5 * (pt0 + pt1)
    r = points - center
    tau_mag = xp.sum(tau**2, axis=-1)
    r_tau = safe_divide(xp.sum(tau * r, axis=-1), tau_mag, xp)
    r_tau = xp.clip(r_tau, -0.5, 0.5)
    return Projection(center, tau, r_tau)

def circle(r: RealArray, center: RealArray, vec1: RealArray, vec2: RealArray, theta: RealArray,
           xp: AnyNamespace) -> RealArray:
    return (r * xp.cos(theta))[..., None] * vec1 + (r * xp.sin(theta))[..., None] * vec2 + center
