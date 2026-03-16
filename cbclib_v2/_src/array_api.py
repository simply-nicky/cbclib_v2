from typing import Any, Literal, Tuple, Set, overload, cast
import jax.numpy as jnp
from jax import devices, random
import numpy as np
from array_api_compat import array_namespace as get_array_namespace, device, numpy as np_array_api
from .annotations import (AnyFloat, Array, ArrayLike, AnyNamespace, ArrayNamespace, CPArray,
                          CPIntArray, CuPy, DTypeLike, Generator, IntArray, IntSequence, JaxArray,
                          JaxDevice, JaxNumPy, NDArray, NumPy, RealArray, RealSequence, Scalar,
                          Shape, ShapeLike, SupportsNamespace)

if CuPy is not None:
    import cupy as cp
    from cupy import fromDlpack as from_dlpack
    from jax import device_put, dlpack as jdl
    from array_api_compat import cupy as cp_array_api

    class CuPyGenerator(cp.random.RandomState):
        def integers(self, low: int | CPIntArray, high: int | None=None,
                    size: Shape | None=None, dtype: DTypeLike = np.int64) -> IntArray:
            return super().randint(low, high=high, size=size, dtype=cast(type[int], dtype))

        def random(self, size: Shape | None=None, dtype: DTypeLike = np.float64) -> RealArray:
            if size is None:
                return super().rand(1, dtype=dtype)
            return super().rand(*size, dtype=dtype)

    def ascupy(array: Array) -> CPArray:
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
    cp_array_api = None  # type: ignore

    def ascupy(array: Array) -> CPArray:
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
        return random.beta(self.key, jnp.asarray(a), jnp.asarray(b), shape=size)

    def binomial(self, n: IntSequence, p: AnyFloat, size: ShapeLike | None = None) -> JaxArray:
        if size is not None:
            size = to_shape(size)
        return random.binomial(self.key, jnp.asarray(n), jnp.asarray(p), shape=size)

    def chisquare(self, df: AnyFloat, size: ShapeLike | None = None) -> JaxArray:
        if size is not None:
            size = to_shape(size)
        return random.chisquare(self.key, jnp.asarray(df), shape=size)

    def choice(self, a: IntSequence | RealSequence | ArrayLike, size: ShapeLike | None = None,
               replace: bool = True, p: RealSequence | ArrayLike | None = None) -> JaxArray:
        if size is None:
            size = (1,)
        if p is not None:
            p = jnp.asarray(p)
        return random.choice(self.key, jnp.asarray(a), shape=to_shape(size), replace=replace, p=p)

    def dirichlet(self, alpha: RealSequence, size: ShapeLike | None = None) -> JaxArray:
        if size is not None:
            size = to_shape(size)
        return random.dirichlet(self.key, jnp.asarray(alpha), shape=size)

    def exponential(self, scale: AnyFloat = 1.0, size: ShapeLike | None = None) -> JaxArray:
        if size is None:
            size = (1,)
        return random.exponential(self.key, shape=to_shape(size)) * jnp.asarray(scale)

    def f(self, dfnum: AnyFloat, dfden: AnyFloat, size: ShapeLike | None = None) -> JaxArray:
        if size is not None:
            size = to_shape(size)
        return random.f(self.key, jnp.asarray(dfnum), jnp.asarray(dfden), shape=size)

    def gamma(self, shape: AnyFloat, scale: AnyFloat = 1.0, size: ShapeLike | None = None
              ) -> JaxArray:
        if size is not None:
            size = to_shape(size)
        return random.gamma(self.key, jnp.asarray(shape), shape=size) * jnp.asarray(scale)

    def geometric(self, p: AnyFloat, size: ShapeLike | None = None) -> JaxArray:
        if size is not None:
            size = to_shape(size)
        return random.geometric(self.key, jnp.asarray(p), shape=size)

    def integers(self, low: IntSequence, high: IntSequence | None = None,
                 size: ShapeLike | None = None, dtype: DTypeLike = np.int64) -> JaxArray:
        if size is None:
            size = (1,)
        if high is None:
            low, high = 0, low
        return random.randint(self.key, to_shape(size), jnp.asarray(low), jnp.asarray(high),
                              dtype=dtype)

    def laplace(self, loc: AnyFloat = 0.0, scale: AnyFloat = 1.0,
                size: ShapeLike | None = None) -> JaxArray:
        if size is None:
            size = (1,)
        laplace = random.laplace(self.key, shape=to_shape(size))
        return laplace * jnp.asarray(scale) + jnp.asarray(loc)

    def logistic(self, loc: AnyFloat = 0.0, scale: AnyFloat = 1.0,
                 size: ShapeLike | None = None) -> JaxArray:
        if size is None:
            size = (1,)
        logistic = random.logistic(self.key, shape=to_shape(size))
        return logistic * jnp.asarray(scale) + jnp.asarray(loc)

    def lognormal(self, mean: AnyFloat = 0.0, sigma: AnyFloat = 1.0, size: ShapeLike | None = None
                  ) -> JaxArray:
        if size is None:
            size = (1,)
        normal = random.normal(self.key, shape=to_shape(size))
        return jnp.exp(normal * jnp.asarray(sigma) + jnp.asarray(mean))

    def multivariate_normal(self, mean: RealSequence, cov: ArrayLike,
                            size: ShapeLike | None = None) -> JaxArray:
        if size is not None:
            size = to_shape(size)
        return random.multivariate_normal(self.key, jnp.asarray(mean), jnp.asarray(cov), shape=size)

    def normal(self, loc: AnyFloat = 0.0, scale: AnyFloat = 1.0,
               size: ShapeLike | None = None) -> JaxArray:
        if size is None:
            size = (1,)
        return random.normal(self.key, shape=to_shape(size)) * jnp.asarray(scale) + jnp.asarray(loc)

    def pareto(self, a: AnyFloat, size: ShapeLike | None = None) -> JaxArray:
        if size is not None:
            size = to_shape(size)
        return random.pareto(self.key, jnp.asarray(a), shape=size)

    def permutation(self, x: IntSequence | RealSequence | ArrayLike) -> JaxArray:
        return random.permutation(self.key, jnp.asarray(x))

    def poisson(self, lam: AnyFloat = 1.0, size: ShapeLike | None = None) -> JaxArray:
        if size is not None:
            size = to_shape(size)
        return random.poisson(self.key, jnp.asarray(lam), shape=size)

    def random(self, size: ShapeLike | None = None, dtype: DTypeLike = np.float64
               ) -> JaxArray:
        if size is None:
            size = (1,)
        return random.uniform(self.key, shape=to_shape(size), dtype=dtype)

    def rayleigh(self, scale: AnyFloat = 1.0, size: ShapeLike | None = None) -> JaxArray:
        if size is None:
            size = (1,)
        return random.rayleigh(self.key, jnp.asarray(scale), shape=to_shape(size))

    def standard_exponential(self, size: ShapeLike | None = None, dtype: DTypeLike = np.float64
                             ) -> JaxArray:
        if size is None:
            size = (1,)
        return random.exponential(self.key, shape=to_shape(size), dtype=dtype)

    def standard_gamma(self, shape: AnyFloat, size: ShapeLike | None = None,
                       dtype: DTypeLike = np.float64) -> JaxArray:
        if size is None:
            size = (1,)
        return random.gamma(self.key, jnp.asarray(shape), shape=to_shape(size), dtype=dtype)

    def standard_normal(self, size: ShapeLike | None = None, dtype: DTypeLike = np.float64
                        ) -> JaxArray:
        if size is None:
            size = (1,)
        return random.normal(self.key, shape=to_shape(size), dtype=dtype)

    def uniform(self, low: AnyFloat = 0.0, high: AnyFloat = 1.0, size: ShapeLike | None = None
                ) -> JaxArray:
        if size is None:
            size = (1,)
        return random.uniform(self.key, shape=to_shape(size), minval=jnp.asarray(low),
                              maxval=jnp.asarray(high))

@overload
def add_at(a: Array, indices: IntArray | Tuple[IntArray, ...], b: Array | Scalar) -> NDArray: ...

@overload
def add_at(a: Array, indices: IntArray | Tuple[IntArray, ...], b: Array | Scalar) -> JaxArray: ...

@overload
def add_at(a: Array, indices: IntArray | Tuple[IntArray, ...], b: Array | Scalar) -> CPArray: ...

def add_at(a: Array, indices: IntArray | Tuple[IntArray, ...], b: Array | Scalar) -> Array:
    xp = array_namespace(a)

    if xp is JaxNumPy:
        return jnp.asarray(a).at[indices].add(b)
    if xp is NumPy:
        np.add.at(np.asarray(a), indices, b)
        return a
    if cp is not None and xp is CuPy:
        a[indices] += b
        return a
    raise ValueError(f"Unsupported array namespace: {xp}")

@overload
def argmin_at(a: NDArray, indices: IntArray, xp: ArrayNamespace[NDArray] = ...) -> NDArray: ...

@overload
def argmin_at(a: JaxArray, indices: IntArray, xp: ArrayNamespace[JaxArray] = ...) -> JaxArray: ...

@overload
def argmin_at(a: CPArray, indices: IntArray, xp: ArrayNamespace[CPArray] = ...) -> CPArray: ...

def argmin_at(a: Array, indices: IntArray, xp: ArrayNamespace = JaxNumPy) -> Array:
    xp = array_namespace(a)

    sort_idxs = xp.argsort(a)
    idxs = set_at(xp.zeros(a.size, dtype=int), sort_idxs, xp.arange(a.size))
    result = xp.full(xp.unique_values(indices).size, a.size + 1, dtype=int)
    return sort_idxs[min_at(result, indices, idxs)]

@overload
def min_at(a: NDArray, indices: IntArray | Tuple[IntArray, ...], b: Array | Scalar) -> NDArray: ...

@overload
def min_at(a: JaxArray, indices: IntArray | Tuple[IntArray, ...], b: Array | Scalar) -> JaxArray: ...

@overload
def min_at(a: CPArray, indices: IntArray | Tuple[IntArray, ...], b: Array | Scalar) -> CPArray: ...

def min_at(a: Array, indices: IntArray | Tuple[IntArray, ...], b: Array | Scalar) -> Array:
    xp = array_namespace(a)

    if xp is JaxNumPy:
        return jnp.asarray(a).at[indices].min(b)
    if xp is NumPy:
        np.minimum.at(np.asarray(a), indices, b)
        return a
    if cp is not None and xp is CuPy:
        cp.minimum.at(cp.asarray(a), indices, b)
        return a
    raise ValueError(f"Unsupported array namespace: {xp}")

@overload
def set_at(a: Array, indices: IntArray | Tuple[IntArray, ...], b: Array | Scalar) -> NDArray: ...

@overload
def set_at(a: Array, indices: IntArray | Tuple[IntArray, ...], b: Array | Scalar) -> JaxArray: ...

@overload
def set_at(a: Array, indices: IntArray | Tuple[IntArray, ...], b: Array | Scalar) -> CPArray: ...

def set_at(a: Array, indices: IntArray | Tuple[IntArray, ...], b: Array | Scalar) -> Array:
    xp = array_namespace(a)

    if xp is JaxNumPy:
        return jnp.asarray(a).at[indices].set(b)
    if xp is NumPy:
        a[indices] = b
        return a
    if cp is not None and xp is CuPy:
        a[indices] = b
        return a
    raise ValueError(f"Unsupported array namespace: {xp}")

@overload
def default_rng(seed: int | None = None, xp: ArrayNamespace[CPArray] = ...
                ) -> Generator[CPArray]: ...

@overload
def default_rng(seed: int | None = None, xp: ArrayNamespace[JaxArray] = ...
                ) -> Generator[JaxArray]: ...

@overload
def default_rng(seed: int | None = None, xp: ArrayNamespace[NDArray] = ...
                ) -> Generator[NDArray]: ...

def default_rng(seed: int | None = None, xp: ArrayNamespace = NumPy) -> Generator:
    if xp is JaxNumPy:
        return cast(Generator, JaxGenerator(seed))
    if xp is NumPy:
        return cast(Generator, np.random.default_rng(seed))
    if cp is not None and xp is CuPy:
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
    def namespaces(*arrays: SupportsNamespace | Array | Any) -> Set:
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
        if cp in nspaces or cp_array_api in nspaces:
            return CuPy
    if jnp in nspaces:
        return JaxNumPy
    if np in nspaces or np_array_api in nspaces:
        return NumPy
    raise ValueError(f"The array namespace {nspaces.pop()} is not supported")

Platform = Literal['cpu', 'gpu']

def get_platform(array: Array) -> Platform:
    xp = array_namespace(array)
    if xp is JaxNumPy:
        dev : JaxDevice = device(array)
        return 'gpu' if dev.platform == 'gpu' else 'cpu'
    if xp is NumPy:
        return 'cpu'
    if cp is not None and xp is CuPy:
        return 'gpu'
    raise ValueError(f"Unsupported array namespace: {xp}")

def asnumpy(array: Array) -> NDArray:
    xp = array_namespace(array)
    if xp is JaxNumPy:
        return np.asarray(array)
    if xp is NumPy:
        return np.asarray(array)
    if cp is not None and xp is CuPy:
        return cp.asnumpy(array)
    raise ValueError(f"Unsupported array namespace: {xp}")

def asjax(array: Array) -> JaxArray:
    xp = array_namespace(array)

    if xp is JaxNumPy:
        return jnp.asarray(array)
    if xp is NumPy:
        return jnp.asarray(array)
    if cp is not None and xp is CuPy:
        return jdl.from_dlpack(cp.asarray(array).toDlpack())
    raise ValueError(f"Unsupported array namespace: {xp}")

def default_api(platform: Platform) -> AnyNamespace:
    if platform == 'cpu':
        return NumPy
    if platform == 'gpu':
        if CuPy is None:
            raise ValueError("CuPy is not available, cannot use GPU platform")
        return CuPy
    raise ValueError(f"Unsupported platform: {platform}")
