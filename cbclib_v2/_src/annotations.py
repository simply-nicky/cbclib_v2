from dataclasses import Field
from typing import (Any, Callable, ClassVar, Dict, Generic, List, Literal, NamedTuple, Protocol,
                    Sequence, Tuple, Type, TypeVar, Union, cast, overload, runtime_checkable)
from types import ModuleType
import numpy as np
import numpy.typing as npt
from jax import Array as JaxArray, Device
import jax.numpy as jnp

GenericAlies = type(list[int]) | type(List[int])
UnionType = type(int | list) | type(Union[int, list])

AnyType = Type | GenericAlies | UnionType | str | Any

T = TypeVar('T')
Self = TypeVar('Self')

class ReferenceType(Generic[T]):
    __callback__: Callable[['ReferenceType[T]'], Any]
    def __new__(cls: type[Self], o: T,
                callback: Callable[['ReferenceType[T]'], Any] | None=...) -> Self:
        ...
    def __call__(self) -> T:
        ...

@runtime_checkable
class Sized(Protocol):
    def __len__(self) -> int: ...

@runtime_checkable
class SupportsNamespace(Protocol):
    def __array_namespace__(self) -> 'ModuleType | ArrayNamespace': ...

@runtime_checkable
class DataclassInstance(Protocol):
    __dataclass_fields__: ClassVar[Dict[str, Field[Any]]]

DType = np.dtype

@runtime_checkable
class SupportsDType(Protocol):
    @property
    def dtype(self) -> DType: ...

DTypeLike = Union[
    str,            # like 'float32', 'int32'
    type[Any],      # like np.float32, np.int32, float, int
    DType,       # like np.dtype('float32'), np.dtype('int32')
    SupportsDType,  # like xp.float32, xp.int32
]

Axis = int | Sequence[int] | None
Scalar = int | float | np.bool_ | np.number | bool | complex
DimSize = int | Any
ShapeLike = Sequence[DimSize]
Shape = Tuple[int, ...]

IntTuple = Tuple[int, ...]
FloatTuple = Tuple[float, ...]

JaxBoolArray = JaxArray
JaxComplexArray = JaxArray
JaxIntArray = JaxArray
JaxRealArray = JaxArray

NDArray = npt.NDArray[Any]
NDArrayLike = npt.ArrayLike
NDBoolArray = npt.NDArray[np.bool_]
NDComplexArray = npt.NDArray[np.floating[Any] | np.complexfloating[Any, Any]]
NDIntArray = npt.NDArray[np.integer[Any]]
NDRealArray = npt.NDArray[np.floating[Any]]

ArrayLike = JaxArray | NDArray | Scalar | Sequence
Array = JaxArray | NDArray
BoolArray = JaxBoolArray | NDBoolArray
ComplexArray = JaxComplexArray | NDComplexArray
IntArray = JaxIntArray | NDIntArray
RealArray = JaxRealArray | NDRealArray

KeyArray = JaxArray

Indices = int | slice | IntArray | Sequence[int] | Tuple[IntArray, ...]

AnyFloat = float | np.floating[Any] | RealArray

IntSequence = int | np.integer[Any] | Sequence[int] | IntArray
RealSequence = float | np.floating[Any] | Sequence[float] | RealArray
ROI = List[int] | Tuple[int, int, int, int] | IntArray

PyTree = Any
ExpandedType = AnyType | Tuple[AnyType, List]

Norm = Literal['backward', 'forward', 'ortho']
Mode = Literal['constant', 'nearest', 'mirror', 'reflect', 'wrap']

ReadOut = Array | int | float | Sequence[int] | Sequence[float]

class EighResult(NamedTuple):
    eigenvalues : Array
    eigenvectors: Array

class UniqueInverseResult(NamedTuple):
    """Struct returned by :func:`unique_inverse`."""
    values: Array
    inverse_indices: Array

class UniqueCountsResult(NamedTuple):
    """Struct returned by :func:`unique_counts`."""
    values: Array
    counts: Array

class LinalgNamespace(Protocol):
    def det(self, a: ArrayLike) -> RealArray:
        """
        Compute the determinant of an array.

        Array API implementation of :func:`numpy.linalg.det`.

        Args:
            a: array of shape ``(..., M, M)`` for which to compute the determinant.

        Returns:
            An array of determinants of shape ``a.shape[:-2]``.

        See also:
            :func:`linalg.det`: Scipy-style API for determinant.

        Examples:
            >>> a = xp.array([[1, 2],
            ...               [3, 4]])
            >>> xp.linalg.det(a)
            Array(-2., dtype=float32)
        """
        ...

    def eigh(self, a: ArrayLike, UPLO: str | None = None, symmetrize_input: bool = True
             ) -> EighResult:
        """
        Compute the eigenvalues and eigenvectors of a Hermitian matrix.

        Array API implementation of :func:`numpy.linalg.eigh`.

        Args:
            a: array of shape ``(..., M, M)``, containing the Hermitian (if complex)
                or symmetric (if real) matrix.
            UPLO: specifies whether the calculation is done with the lower triangular
                part of ``a`` (``'L'``, default) or the upper triangular part (``'U'``).
            symmetrize_input: if True (default) then input is symmetrized, which leads
                to better behavior under automatic differentiation.

        Returns:
            A namedtuple ``(eigenvalues, eigenvectors)`` where

            - ``eigenvalues``: an array of shape ``(..., M)`` containing the eigenvalues,
                sorted in ascending order.
            - ``eigenvectors``: an array of shape ``(..., M, M)``, where column ``v[:, i]`` is the
                normalized eigenvector corresponding to the eigenvalue ``w[i]``.

            See also:
            - :func:`linalg.eig`: general eigenvalue decomposition.
            - :func:`linalg.eigvalsh`: compute eigenvalues only.

        Examples:
            >>> a = xp.array([[1, -2j],
            ...                [2j, 1]])
            >>> w, v = xp.linalg.eigh(a)
            >>> w
            Array([-1.,  3.], dtype=float32)
            >>> with xp.printoptions(precision=3):
            ...   v
            Array([[-0.707+0.j   , -0.707+0.j   ],
                   [ 0.   +0.707j,  0.   -0.707j]], dtype=complex64)
        """
        ...

    def inv(self, a: ArrayLike) -> RealArray:
        """Return the inverse of a square matrix

        Array API implementation of :func:`numpy.linalg.inv`.

        Args:
            a: array of shape ``(..., N, N)`` specifying square array(s) to be inverted.

        Returns:
            Array of shape ``(..., N, N)`` containing the inverse of the input.

        Notes:
            In most cases, explicitly computing the inverse of a matrix is ill-advised. For
            example, to compute ``x = inv(A) @ b``, it is more performant and numerically
            precise to use a direct solve.

        Examples:
            Compute the inverse of a 3x3 matrix

            >>> a = xp.array([[1., 2., 3.],
            ...               [2., 4., 2.],
            ...               [3., 2., 1.]])
            >>> a_inv = xp.linalg.inv(a)
            >>> a_inv  # doctest: +SKIP
            Array([[ 0.        , -0.25      ,  0.5       ],
                   [-0.25      ,  0.5       , -0.25000003],
                   [ 0.5       , -0.25      ,  0.        ]], dtype=float32)

            Check that multiplying with the inverse gives the identity:

            >>> xp.allclose(a @ a_inv, xp.eye(3), atol=1E-5)
            Array(True, dtype=bool)

            Multiply the inverse by a vector ``b``, to find a solution to ``a @ x = b``

            >>> b = xp.array([1., 4., 2.])
            >>> a_inv @ b
            Array([ 0.  ,  1.25, -0.5 ], dtype=float32)

            Note, however, that explicitly computing the inverse in such a case can lead
            to poor performance and loss of precision as the size of the problem grows.
            Instead, you should use a direct solver like :func:`jax.numpy.linalg.solve`:

            >>> xp.linalg.solve(a, b)
            Array([ 0.  ,  1.25, -0.5 ], dtype=float32)
        """
        ...

class ArrayNamespace(Protocol):
    uint4   = int
    uint8   = int
    uint16  = int
    uint32  = int
    uint64  = int
    int4    = int
    int8    = int
    int16   = int
    int32   = int
    int64   = int
    float8  = float
    float16 = float
    float32 = single = float
    float64 = double = float
    complex64 = csingle = float
    complex128 = cdouble = float

    inf     : float
    nan     : float
    pi      : float

    linalg  : LinalgNamespace

    def abs(self, x: ArrayLike, /) -> Array:
        """Alias of :func:`absolute`."""
        ...

    def all(self, a: ArrayLike, axis: Axis = None,
            keepdims: bool = False, *, where: ArrayLike | None = None) -> Array:
        r"""Test whether all array elements along a given axis evaluate to True.

        Array API implementation of :func:`numpy.all`.

        Args:
            a: Input array.
            axis: int or array, default=None. Axis along which to be tested. If None,
                tests along all the axes.
            keepdims: bool, default=False. If true, reduced axes are left in the result
                with size 1.
            where: int or array of boolean dtype, default=None. The elements to be used
                in the test. Array should be broadcast compatible to the input.

        Returns:
            An array of boolean values.

        Examples:
            By default, ``xp.all`` tests for True values along all the axes.

            >>> x = xp.array([[True, True, True, False],
            ...               [True, False, True, False],
            ...               [True, True, False, False]])
            >>> xp.all(x)
            Array(False, dtype=bool)

            If ``axis=0``, tests for True values along axis 0.

            >>> xp.all(x, axis=0)
            Array([ True, False, False, False], dtype=bool)

            If ``keepdims=True``, ``ndim`` of the output will be same of that of the input.

            >>> xp.all(x, axis=0, keepdims=True)
            Array([[ True, False, False, False]], dtype=bool)

            To include specific elements in testing for True values, you can use a``where``.

            >>> where=xp.array([[1, 0, 1, 0],
            ...                 [0, 0, 1, 1],
            ...                 [1, 1, 1, 0]], dtype=bool)
            >>> xp.all(x, axis=0, keepdims=True, where=where)
            Array([[ True,  True, False, False]], dtype=bool)
        """
        ...

    def allclose(self, a: ArrayLike, b: ArrayLike, rtol: ArrayLike = 1e-05,
                 atol: ArrayLike = 1e-08, equal_nan: bool = False) -> Array:
        r"""Check if two arrays are element-wise approximately equal within a tolerance.

        Array API implementation of :func:`numpy.allclose`.

        Essentially this function evaluates the following condition:

        .. math::

            |a - b| \le \mathtt{atol} + \mathtt{rtol} * |b|

        ``xp.inf`` in ``a`` will be considered equal to ``xp.inf`` in ``b``.

        Args:
            a: first input array to compare.
            b: second input array to compare.
            rtol: relative tolerance used for approximate equality. Default = 1e-05.
            atol: absolute tolerance used for approximate equality. Default = 1e-08.
            equal_nan: Boolean. If ``True``, NaNs in ``a`` will be considered
                equal to NaNs in ``b``. Default is ``False``.

        Returns:
            Boolean scalar array indicating whether the input arrays are element-wise
            approximately equal within the specified tolerances.

        See Also:
            - :func:`isclose`
            - :func:`equal`

        Examples:
            >>> xp.allclose(xp.array([1e6, 2e6, 3e6]), xp.array([1e6, 2e6, 3e7]))
            Array(False, dtype=bool)
            >>> xp.allclose(xp.array([1e6, 2e6, 3e6]),
            ...             xp.array([1.00008e6, 2.00008e7, 3.00008e8]), rtol=1e3)
            Array(True, dtype=bool)
            >>> xp.allclose(xp.array([1e6, 2e6, 3e6]),
            ...             xp.array([1.00001e6, 2.00002e6, 3.00009e6]), atol=1e3)
            Array(True, dtype=bool)
            >>> xp.allclose(xp.array([xp.nan, 1, 2]),
            ...             xp.array([xp.nan, 1, 2]), equal_nan=True)
            Array(True, dtype=bool)
        """
        ...

    def any(self, a: ArrayLike, axis: Axis = None,
            keepdims: bool = False, *, where: ArrayLike | None = None) -> Array:
        r"""Test whether any of the array elements along a given axis evaluate to True.

        Array API implementation of :func:`numpy.any`.

        Args:
            a: Input array.
            axis: int or array, default=None. Axis along which to be tested. If None,
                tests along all the axes.
            keepdims: bool, default=False. If true, reduced axes are left in the result
                with size 1.
            where: int or array of boolean dtype, default=None. The elements to be used
                in the test. Array should be broadcast compatible to the input.

        Returns:
            An array of boolean values.

        Examples:
            By default, ``xp.any`` tests along all the axes.

            >>> x = xp.array([[True, True, True, False],
            ...               [True, False, True, False],
            ...               [True, True, False, False]])
            >>> xp.any(x)
            Array(True, dtype=bool)

            If ``axis=0``, tests along axis 0.

            >>> xp.any(x, axis=0)
            Array([ True,  True,  True, False], dtype=bool)

            If ``keepdims=True``, ``ndim`` of the output will be same of that of the input.

            >>> xp.any(x, axis=0, keepdims=True)
            Array([[ True,  True,  True, False]], dtype=bool)

            To include specific elements in testing for True values, you can use a``where``.

            >>> where=xp.array([[1, 0, 1, 0],
            ...                 [0, 1, 0, 1],
            ...                 [1, 0, 1, 0]], dtype=bool)
            >>> xp.any(x, axis=0, keepdims=True, where=where)
            Array([[ True, False,  True, False]], dtype=bool)
        """
        ...

    def append(self, arr: ArrayLike, values: ArrayLike, axis: int | None = None) -> Array:
        """Return a new array with values appended to the end of the original array.

        Array API implementation of :func:`numpy.append`.

        Args:
            arr: original array.
            values: values to be appended to the array. The ``values`` must have
                the same number of dimensions as ``arr``, and all dimensions must
                match except in the specified axis.
            axis: axis along which to append values. If None (default), both ``arr``
                and ``values`` will be flattened before appending.

        Returns:
            A new array with values appended to ``arr``.

        See also:
            - :func:`insert`
            - :func:`delete`

        Examples:
            >>> a = xp.array([1, 2, 3])
            >>> b = xp.array([4, 5, 6])
            >>> xp.append(a, b)
            Array([1, 2, 3, 4, 5, 6], dtype=int32)

            Appending along a specific axis:

            >>> a = xp.array([[1, 2],
            ...                [3, 4]])
            >>> b = xp.array([[5, 6]])
            >>> xp.append(a, b, axis=0)
            Array([[1, 2],
                   [3, 4],
                   [5, 6]], dtype=int32)

            Appending along a trailing axis:

            >>> a = xp.array([[1, 2, 3],
            ...               [4, 5, 6]])
            >>> b = xp.array([[7], [8]])
            >>> xp.append(a, b, axis=1)
            Array([[1, 2, 3, 7],
                   [4, 5, 6, 8]], dtype=int32)
        """
        ...

    def array(self, object: Any, dtype: DTypeLike | None = None, copy: bool = True,
              order: str | None = "K", ndmin: int = 0, *, device: Device | None = None
              ) -> Array:
        """Convert an object to a Array API array.

        Array API implementation of :func:`numpy.array`.

        Args:
            object: an object that is convertible to an array. This includes Array API
                arrays, NumPy arrays, Python scalars, Python collections like lists
                and tuples, objects with an ``__array__`` method, and objects
                supporting the Python buffer protocol.
            dtype: optionally specify the dtype of the output array. If not
                specified it will be inferred from the input.
            copy: specify whether to force a copy of the input. Default: True.
            order: not implemented in Array API
            ndmin: integer specifying the minimum number of dimensions in the
                output array.
            device: optional :class:`~xp.Device` to which the created array will be
            committed.

        Returns:
            A Array API array constructed from the input.

        See also:
            - :func:`asarray`: like `array`, but by default only copies
                when necessary.
            - :func:`from_dlpack`: construct a Array API array from an object
                that implements the dlpack interface.
            - :func:`frombuffer`: construct a Array API array from an object
                that implements the buffer interface.

        Examples:
            Constructing Array API arrays from Python scalars:

            >>> xp.array(True)
            Array(True, dtype=bool)
            >>> xp.array(42)
            Array(42, dtype=int32, weak_type=True)
            >>> xp.array(3.5)
            Array(3.5, dtype=float32, weak_type=True)
            >>> xp.array(1 + 1j)
            Array(1.+1.j, dtype=complex64, weak_type=True)

            Constructing Array API arrays from Python collections:

            >>> xp.array([1, 2, 3])  # list of ints -> 1D array
            Array([1, 2, 3], dtype=int32)
            >>> xp.array([(1, 2, 3), (4, 5, 6)])  # list of tuples of ints -> 2D array
            Array([[1, 2, 3],
                   [4, 5, 6]], dtype=int32)
            >>> xp.array(range(5))
            Array([0, 1, 2, 3, 4], dtype=int32)

            Constructing Array API arrays from NumPy arrays:

            >>> xp.array(np.linspace(0, 2, 5))
            Array([0. , 0.5, 1. , 1.5, 2. ], dtype=float32)

            Constructing a Array API array via the Python buffer interface, using Python's
            built-in :mod:`array` module.

            >>> from array import array
            >>> pybuffer = array('i', [2, 3, 5, 7])
            >>> xp.array(pybuffer)
            Array([2, 3, 5, 7], dtype=int32)
        """
        ...

    def argsort(self, a: ArrayLike, axis: int | None = -1, *, kind: None = None,
                order: None = None, stable: bool = True, descending: bool = False
                ) -> Array:
        """Return indices that sort an array.

        Array API implementation of :func:`numpy.argsort`.

        Args:
            a: array to sort
            axis: integer axis along which to sort. Defaults to ``-1``, i.e. the last
                axis. If ``None``, then ``a`` is flattened before being sorted.
            stable: boolean specifying whether a stable sort should be used. Default=True.
            descending: boolean specifying whether to sort in descending order. Default=False.
            kind: deprecated; instead specify sort algorithm using stable=True or stable=False.
            order: not supported by Array API

        Returns:
            Array of indices that sort an array. Returned array will be of shape ``a.shape``
            (if ``axis`` is an integer) or of shape ``(a.size,)`` (if ``axis`` is None).

        Examples:
            Simple 1-dimensional sort

            >>> x = xp.array([1, 3, 5, 4, 2, 1])
            >>> indices = xp.argsort(x)
            >>> indices
            Array([0, 5, 4, 1, 3, 2], dtype=int32)
            >>> x[indices]
            Array([1, 1, 2, 3, 4, 5], dtype=int32)

            Sort along the last axis of an array:

            >>> x = xp.array([[2, 1, 3],
            ...                [6, 4, 3]])
            >>> indices = xp.argsort(x, axis=1)
            >>> indices
            Array([[1, 0, 2],
                   [2, 1, 0]], dtype=int32)
            >>> xp.take_along_axis(x, indices, axis=1)
            Array([[1, 2, 3],
                   [3, 4, 6]], dtype=int32)


        See also:
            - :func:`sort`: return sorted values directly.
            - :func:`lexsort`: lexicographical sort of multiple arrays.
        """
        ...

    def asarray(self, a: Any, dtype: DTypeLike | None = None, order: str | None = None,
                *, copy: bool | None = None, device: Device | None = None) -> Array:
        """Convert an object to a Array API array.

        Array API implementation of :func:`numpy.asarray`.

        Args:
            a: an object that is convertible to an array. This includes Array API
                arrays, NumPy arrays, Python scalars, Python collections like lists
                and tuples, objects with an ``__array__`` method, and objects
                supporting the Python buffer protocol.
            dtype: optionally specify the dtype of the output array. If not
                specified it will be inferred from the input.
            order: not implemented in Array API
            copy: optional boolean specifying the copy mode. If True, then always
                return a copy. If False, then error if a copy is necessary. Default is
                None, which will only copy when necessary.
            device: optional :class:`~xp.Device` to which the created array will be
                committed.

        Returns:
        A Array API array constructed from the input.

        See also:
            - :func:`array`: like `asarray`, but defaults to `copy=True`.
            - :func:`from_dlpack`: construct a Array API array from an object
                that implements the dlpack interface.
            - :func:`frombuffer`: construct a Array API array from an object
                that implements the buffer interface.

        Examples:
            Constructing Array API arrays from Python scalars:

            >>> xp.asarray(True)
            Array(True, dtype=bool)
            >>> xp.asarray(42)
            Array(42, dtype=int32, weak_type=True)
            >>> xp.asarray(3.5)
            Array(3.5, dtype=float32, weak_type=True)
            >>> xp.asarray(1 + 1j)
            Array(1.+1.j, dtype=complex64, weak_type=True)

            Constructing Array API arrays from Python collections:

            >>> xp.asarray([1, 2, 3])  # list of ints -> 1D array
            Array([1, 2, 3], dtype=int32)
            >>> xp.asarray([(1, 2, 3), (4, 5, 6)])  # list of tuples of ints -> 2D array
            Array([[1, 2, 3],
                   [4, 5, 6]], dtype=int32)
            >>> xp.asarray(range(5))
            Array([0, 1, 2, 3, 4], dtype=int32)

            Constructing Array API arrays from NumPy arrays:

            >>> xp.asarray(np.linspace(0, 2, 5))
            Array([0. , 0.5, 1. , 1.5, 2. ], dtype=float32)

            Constructing a Array API array via the Python buffer interface, using Python's
            built-in :mod:`array` module.

            >>> from array import array
            >>> pybuffer = array('i', [2, 3, 5, 7])
            >>> xp.asarray(pybuffer)
            Array([2, 3, 5, 7], dtype=int32)
        """
        ...

    def arange(self, start: ArrayLike | DimSize, stop: ArrayLike | DimSize | None = None,
               step: ArrayLike | None = None, dtype: DTypeLike | None = None) -> Array:
        """Create an array of evenly-spaced values.

        Array API implementation of :func:`numpy.arange`.

        Similar to Python's :func:`range` function, this can be called with a few
        different positional signatures:

        - ``xp.arange(stop)``: generate values from 0 to ``stop``, stepping by 1.
        - ``xp.arange(start, stop)``: generate values from ``start`` to ``stop``,
        stepping by 1.
        - ``xp.arange(start, stop, step)``: generate values from ``start`` to ``stop``,
        stepping by ``step``.

        Like with Python's :func:`range` function, the starting value is inclusive,
        and the stop value is exclusive.

        Args:
            start: start of the interval, inclusive.
            stop: optional end of the interval, exclusive. If not specified, then
                ``(start, stop) = (0, start)``
            step: optional step size for the interval. Default = 1.
            dtype: optional dtype for the returned array; if not specified it will
                be determined via type promotion of `start`, `stop`, and `step`.
            device: (optional) :class:`~xp.Device` to which the created array will be
                committed.

        Returns:
            Array of evenly-spaced values from ``start`` to ``stop``, separated by ``step``.

        Note:
            Using ``arange`` with a floating-point ``step`` argument can lead to unexpected
            results due to accumulation of floating-point errors, especially with
            lower-precision data types like ``float8_*`` and ``bfloat16``.
            To avoid precision errors, consider generating a range of integers, and scaling
            it to the desired range. For example, instead of this::

                xp.arange(-1, 1, 0.01, dtype='bfloat16')

            it can be more accurate to generate a sequence of integers, and scale them::

                (xp.arange(-100, 100) * 0.01).astype('bfloat16')

            Examples:
            Single-argument version specifies only the ``stop`` value:

            >>> xp.arange(4)
            Array([0, 1, 2, 3], dtype=int32)

            Passing a floating-point ``stop`` value leads to a floating-point result:

            >>> xp.arange(4.0)
            Array([0., 1., 2., 3.], dtype=float32)

            Two-argument version specifies ``start`` and ``stop``, with ``step=1``:

            >>> xp.arange(1, 6)
            Array([1, 2, 3, 4, 5], dtype=int32)

            Three-argument version specifies ``start``, ``stop``, and ``step``:

            >>> xp.arange(0, 2, 0.5)
            Array([0. , 0.5, 1. , 1.5], dtype=float32)

        See Also:
            - :func:`linspace`: generate a fixed number of evenly-spaced values.
        """
        ...

    def arccos(self, x: ArrayLike, /) -> RealArray:
        """Compute element-wise inverse of trigonometric cosine of input.

        Array API implementation of :obj:`numpy.arccos`.

        Args:
            x: input array or scalar.

        Returns:
            An array containing the inverse trigonometric cosine of each element of ``x``
            in radians in the range ``[0, pi]``, promoting to inexact dtype.

        Note:
            - ``xp.arccos`` returns ``nan`` when ``x`` is real-valued and not in the closed
                interval ``[-1, 1]``.
            - ``xp.arccos`` follows the branch cut convention of :obj:`numpy.arccos` for
                complex inputs.

        See also:
            - :func:`cos`: Computes a trigonometric cosine of each element of
                input.
            - :func:`arcsin` and :func:`asin`: Computes the inverse of
                trigonometric sine of each element of input.
            - :func:`arctan` and :func:`atan`: Computes the inverse of
                trigonometric tangent of each element of input.

        Examples:
            >>> x = xp.array([-2, -1, -0.5, 0, 0.5, 1, 2])
            >>> with xp.printoptions(precision=3, suppress=True):
            ...   xp.arccos(x)
            Array([  nan, 3.142, 2.094, 1.571, 1.047, 0.   ,   nan], dtype=float32)

            For complex inputs:

            >>> with xp.printoptions(precision=3, suppress=True):
            ...   xp.arccos(4-1j)
            Array(0.252+2.097j, dtype=complex64, weak_type=True)
        """
        ...

    def arcsin(self, x: ArrayLike, /) -> RealArray:
        r"""Compute element-wise inverse of trigonometric sine of input.

        Array API implementation of :obj:`numpy.arcsin`.

        Args:
            x: input array or scalar.

        Returns:
            An array containing the inverse trigonometric sine of each element of ``x``
            in radians in the range ``[-pi/2, pi/2]``, promoting to inexact dtype.

        Note:
            - ``xp.arcsin`` returns ``nan`` when ``x`` is real-valued and not in the closed
                interval ``[-1, 1]``.
            - ``xp.arcsin`` follows the branch cut convention of :obj:`numpy.arcsin` for
                complex inputs.

        See also:
            - :func:`sin`: Computes a trigonometric sine of each element of input.
            - :func:`arccos` and :func:`acos`: Computes the inverse of
                trigonometric cosine of each element of input.
            - :func:`arctan` and :func:`atan`: Computes the inverse of
                trigonometric tangent of each element of input.

        Examples:
            >>> x = xp.array([-2, -1, -0.5, 0, 0.5, 1, 2])
            >>> with xp.printoptions(precision=3, suppress=True):
            ...   xp.arcsin(x)
            Array([   nan, -1.571, -0.524,  0.   ,  0.524,  1.571,    nan], dtype=float32)

            For complex-valued inputs:

            >>> with xp.printoptions(precision=3, suppress=True):
            ...   xp.arcsin(3+4j)
            Array(0.634+2.306j, dtype=complex64, weak_type=True)
        """
        ...

    def arctan(self, x: ArrayLike, /) -> RealArray:
        """Compute element-wise inverse of trigonometric tangent of input.

        Array API implement of :obj:`numpy.arctan`.

        Args:
            x: input array or scalar.

        Returns:
            An array containing the inverse trigonometric tangent of each element ``x``
            in radians in the range ``[-pi/2, pi/2]``, promoting to inexact dtype.

        Note:
            ``xp.arctan`` follows the branch cut convention of :obj:`numpy.arctan` for
            complex inputs.

        See also:
            - :func:`tan`: Computes a trigonometric tangent of each element of
                input.
            - :func:`arcsin` and :func:`asin`: Computes the inverse of
                trigonometric sine of each element of input.
            - :func:`arccos` and :func:`atan`: Computes the inverse of
                trigonometric cosine of each element of input.

        Examples:
            >>> x = xp.array([-xp.inf, -20, -1, 0, 1, 20, xp.inf])
            >>> with xp.printoptions(precision=3, suppress=True):
            ...   xp.arctan(x)
            Array([-1.571, -1.521, -0.785,  0.   ,  0.785,  1.521,  1.571], dtype=float32)

            For complex-valued inputs:

            >>> with xp.printoptions(precision=3, suppress=True):
            ...   xp.arctan(2+7j)
            Array(1.532+0.133j, dtype=complex64, weak_type=True)
        """
        ...

    def arctan2(self, x1: ArrayLike, x2: ArrayLike, /) -> RealArray:
        ...

    def argmax(self, a: ArrayLike, axis: int | None = None,
               keepdims: bool | None = None) -> Array:
        """Return the index of the maximum value of an array.

        Array API implementation of :func:`numpy.argmax`.

        Args:
            a: input array
            axis: optional integer specifying the axis along which to find the maximum
                value. If ``axis`` is not specified, ``a`` will be flattened.
            keepdims: if True, then return an array with the same number of dimensions
                as ``a``.

        Returns:
            an array containing the index of the maximum value along the specified axis.

        See also:
            - :func:`argmin`: return the index of the minimum value.
            - :func:`nanargmax`: compute ``argmax`` while ignoring NaN values.

        Examples:
            >>> x = xp.array([1, 3, 5, 4, 2])
            >>> xp.argmax(x)
            Array(2, dtype=int32)

            >>> x = xp.array([[1, 3, 2],
            ...                [5, 4, 1]])
            >>> xp.argmax(x, axis=1)
            Array([1, 0], dtype=int32)

            >>> xp.argmax(x, axis=1, keepdims=True)
            Array([[1],
                   [0]], dtype=int32)
        """
        ...

    def argmin(self, a: ArrayLike, axis: int | None = None,
               keepdims: bool | None = None) -> Array:
        """Return the index of the minimum value of an array.

        Array API implementation of :func:`numpy.argmax`.

        Args:
            a: input array
            axis: optional integer specifying the axis along which to find the maximum
                value. If ``axis`` is not specified, ``a`` will be flattened.
            keepdims: if True, then return an array with the same number of dimensions
                as ``a``.

        Returns:
            an array containing the index of the maximum value along the specified axis.

        See also:
            - :func:`argmax`: return the index of the maximum value.
            - :func:`nanargmin`: compute ``argmin`` while ignoring NaN values.

        Examples:
            >>> x = xp.array([1, 3, 5, 4, 2])
            >>> xp.argmin(x)
            Array(0, dtype=int32)

            >>> x = xp.array([[1, 3, 2],
            ...               [5, 4, 1]])
            >>> xp.argmin(x, axis=1)
            Array([0, 2], dtype=int32)

            >>> xp.argmin(x, axis=1, keepdims=True)
            Array([[0],
                   [2]], dtype=int32)
        """
        ...

    @overload
    def atleast_1d(self, arys: ArrayLike, /) -> Array: ...

    @overload
    def atleast_1d(self, *arys: ArrayLike) -> List[Array]: ...

    def atleast_1d(self, *arys: ArrayLike) -> Array | List[Array]:
        """Convert inputs to arrays with at least 1 dimension.

        Array API implementation of :func:`numpy.atleast_1d`.

        Args:
            zero or more arraylike arguments.

        Returns:
            an array or list of arrays corresponding to the input values. Arrays
            of shape ``()`` are converted to shape ``(1,)``, and arrays with other
            shapes are returned unchanged.

        See also:
            - :func:`asarray`
            - :func:`atleast_2d`
            - :func:`atleast_3d`

        Examples:
            Scalar arguments are converted to 1D, length-1 arrays:

            >>> x = xp.float32(1.0)
            >>> xp.atleast_1d(x)
            Array([1.], dtype=float32)

            Higher dimensional inputs are returned unchanged:

            >>> y = xp.arange(4)
            >>> xp.atleast_1d(y)
            Array([0, 1, 2, 3], dtype=int32)

            Multiple arguments can be passed to the function at once, in which
            case a list of results is returned:

            >>> xp.atleast_1d(x, y)
            [Array([1.], dtype=float32), Array([0, 1, 2, 3], dtype=int32)]
        """
        ...

    def broadcast_to(self, array: ArrayLike, shape: DimSize | ShapeLike) -> Array:
        """Broadcast an array to a specified shape.

        Array API implementation of :func:`numpy.broadcast_to`. Array API uses NumPy-style
        broadcasting rules, which you can read more about at `NumPy broadcasting`_.

        Args:
            array: array to be broadcast.
            shape: shape to which the array will be broadcast.

        Returns:
            a copy of array broadcast to the specified shape.

        See also:
            - :func:`broadcast_arrays`: broadcast arrays to a common shape.
            - :func:`broadcast_shapes`: broadcast input shapes to a common shape.

        Examples:
            >>> x = xp.int32(1)
            >>> xp.broadcast_to(x, (1, 4))
            Array([[1, 1, 1, 1]], dtype=int32)

            >>> x = xp.array([1, 2, 3])
            >>> xp.broadcast_to(x, (2, 3))
            Array([[1, 2, 3],
                   [1, 2, 3]], dtype=int32)

            >>> x = xp.array([[2], [4]])
            >>> xp.broadcast_to(x, (2, 4))
            Array([[2, 2, 2, 2],
                   [4, 4, 4, 4]], dtype=int32)

            .. _NumPy broadcasting: https://numpy.org/doc/stable/user/basics.broadcasting.html
        """
        ...

    def ceil(self, x: ArrayLike, /) -> Array:
        """Round input to the nearest integer upwards.

        Array API implementation of :obj:`numpy.ceil`.

        Args:
            x: input array or scalar. Must not have complex dtype.

        Returns:
            An array with same shape and dtype as ``x`` containing the values rounded to
            the nearest integer that is greater than or equal to the value itself.

        See also:
            - :func:`trunc`: Rounds the input to the nearest interger towards
                zero.
            - :func:`floor`: Rounds the input down to the nearest integer.

        Examples:
            >>> key = jax.random.key(1)
            >>> x = jax.random.uniform(key, (3, 3), minval=-5, maxval=5)
            >>> with xp.printoptions(precision=2, suppress=True):
            ...     print(x)
            [[ 2.55 -1.87 -3.76]
             [ 0.48  3.85 -1.94]
             [ 3.2   4.56 -1.43]]
            >>> xp.ceil(x)
            Array([[ 3., -1., -3.],
                   [ 1.,  4., -1.],
                   [ 4.,  5., -1.]], dtype=float32)
        """
        ...

    def clip(self, arr: ArrayLike | None = None, /, min: ArrayLike | None = None,
             max: ArrayLike | None = None) -> Array:
        """Clip array values to a specified range.

        Array API implementation of :func:`numpy.clip`.

        Args:
            arr: N-dimensional array to be clipped.
            min: optional minimum value of the clipped range; if ``None`` (default) then
                result will not be clipped to any minimum value. If specified, it should be
                broadcast-compatible with ``arr`` and ``max``.
            max: optional maximum value of the clipped range; if ``None`` (default) then
                result will not be clipped to any maximum value. If specified, it should be
                broadcast-compatible with ``arr`` and ``min``.

        Returns:
            An array containing values from ``arr``, with values smaller than ``min`` set
            to ``min``, and values larger than ``max`` set to ``max``.

        See also:
            - :func:`minimum`: Compute the element-wise minimum value of two arrays.
            - :func:`maximum`: Compute the element-wise maximum value of two arrays.

        Examples:
            >>> arr = xp.array([0, 1, 2, 3, 4, 5, 6, 7])
            >>> xp.clip(arr, 2, 5)
            Array([2, 2, 2, 3, 4, 5, 5, 5], dtype=int32)
        """
        ...

    def compress(self, condition: ArrayLike, a: ArrayLike, axis: int | None = None, *,
                 fill_value: ArrayLike = 0) -> Array:
        """Compress an array along a given axis using a boolean condition.

        Array API implementation of :func:`numpy.compress`.

        Args:
            condition: 1-dimensional array of conditions. Will be converted to boolean.
            a: N-dimensional array of values.
            axis: axis along which to compress. If None (default) then ``a`` will be
                flattened, and axis will be set to 0.
            fill_value: if ``size`` is specified, fill padded entries with this value (default: 0).

        Returns:
            An array of dimension ``a.ndim``, compressed along the specified axis.

        See also:
            - :func:`extract`: 1D version of ``compress``.
            - :meth:`compress`: equivalent functionality as an array method.

        Notes:
            This function does not require strict shape agreement between ``condition`` and ``a``.
            If ``condition.size > a.shape[axis]``, then ``condition`` will be truncated, and if
            ``a.shape[axis] > condition.size``, then ``a`` will be truncated.

        Examples:
            Compressing along the rows of a 2D array:

            >>> a = xp.array([[1,  2,  3,  4],
            ...               [5,  6,  7,  8],
            ...               [9,  10, 11, 12]])
            >>> condition = xp.array([True, False, True])
            >>> xp.compress(condition, a, axis=0)
            Array([[ 1,  2,  3,  4],
                   [ 9, 10, 11, 12]], dtype=int32)

            For convenience, you can equivalently use the :meth:`~compress`
            method of Array API arrays:

            >>> a.compress(condition, axis=0)
            Array([[ 1,  2,  3,  4],
                   [ 9, 10, 11, 12]], dtype=int32)

            Note that the condition need not match the shape of the specified axis;
            here we compress the columns with the length-3 condition. Values beyond
            the size of the condition are ignored:

            >>> xp.compress(condition, a, axis=1)
            Array([[ 1,  3],
                   [ 5,  7],
                   [ 9, 11]], dtype=int32)
        """
        ...

    def concatenate(self, arrays: np.ndarray | Array | Sequence[ArrayLike],
                    axis: int | None = 0, dtype: DTypeLike | None = None) -> Array:
        """Join arrays along an existing axis.

        Array API implementation of :func:`numpy.concatenate`.

        Args:
            arrays: a sequence of arrays to concatenate; each must have the same shape
                except along the specified axis. If a single array is given it will be
                treated equivalently to `arrays = unstack(arrays)`, but the implementation
                will avoid explicit unstacking.
            axis: specify the axis along which to concatenate.
            dtype: optional dtype of the resulting array. If not specified, the dtype
                will be determined via type promotion rules described in :ref:`type-promotion`.

        Returns:
            the concatenated result.

        See also:
            - :func:`concat`: Array API version of this function.
            - :func:`stack`: concatenate arrays along a new axis.

        Examples:
            One-dimensional concatenation:

            >>> x = xp.arange(3)
            >>> y = xp.zeros(3, dtype=int)
            >>> xp.concatenate([x, y])
            Array([0, 1, 2, 0, 0, 0], dtype=int32)

            Two-dimensional concatenation:

            >>> x = xp.ones((2, 3))
            >>> y = xp.zeros((2, 1))
            >>> xp.concatenate([x, y], axis=1)
            Array([[1., 1., 1., 0.],
                   [1., 1., 1., 0.]], dtype=float32)
        """
        ...

    def copy(self, a: ArrayLike, order: str | None = None) -> Array:
        """Return a copy of the array.

        Array API implementation of :func:`numpy.copy`.

        Args:
            a: arraylike object to copy
            order: Controls the memory layout of the copy. 'C' means C-order,
                'F' means F-order, 'A' means 'F' if `a` is Fortran contiguous,
                'C' otherwise. 'K' means match the layout of `a` as closely
                as possible. (Note that this function and :meth:`ndarray.copy`
                are very similar, but have different default values for their order
                arguments.)

        Returns:
            a copy of the input array ``a``.

        See Also:
            - :func:`array`: create an array with or without a copy.

        >>> f = lambda x: 2 * x
        >>> x = xp.arange(4)
        >>> y = f(x)
        >>> print(y)
        [0 2 4 6]

        In situations like this, an explicit copy will let you keep access to the
        original buffer:

        >>> x = xp.arange(4)
        >>> y = f(x.copy())
        >>> print(y)
        [0 2 4 6]
        >>> print(x)
        [0 1 2 3]
        """
        ...

    def cos(self, x: ArrayLike, /) -> RealArray:
        """Compute a trigonometric cosine of each element of input.

        Array API implementation of :obj:`numpy.cos`.

        Args:
            x: scalar or array. Angle in radians.

        Returns:
            An array containing the cosine of each element in ``x``, promotes to inexact
            dtype.

        See also:
            - :func:`sin`: Computes a trigonometric sine of each element of input.
            - :func:`tan`: Computes a trigonometric tangent of each element of
                input.
            - :func:`arccos` and :func:`acos`: Computes the inverse of
                trigonometric cosine of each element of input.

        Examples:
            >>> pi = xp.pi
            >>> x = xp.array([pi/4, pi/2, 3*pi/4, 5*pi/6])
            >>> with xp.printoptions(precision=3, suppress=True):
            ...   print(xp.cos(x))
            [ 0.707 -0.    -0.707 -0.866]
        """
        ...

    def cosh(self, x: ArrayLike, /) -> RealArray:
        r"""Calculate element-wise hyperbolic cosine of input.

        Array API implementation of :obj:`numpy.cosh`.

        The hyperbolic cosine is defined by:

        .. math::

            cosh(x) = \frac{e^x + e^{-x}}{2}

        Args:
            x: input array or scalar.

        Returns:
            An array containing the hyperbolic cosine of each element of ``x``, promoting
            to inexact dtype.

        Note:
            ``xp.cosh`` is equivalent to computing ``xp.cos(1j * x)``.

        See also:
            - :func:`sinh`: Computes the element-wise hyperbolic sine of the input.
            - :func:`tanh`: Computes the element-wise hyperbolic tangent of the
                input.
            - :func:`arccosh`:  Computes the element-wise inverse of hyperbolic
                cosine of the input.

        Examples:
            >>> x = xp.array([[3, -1, 0],
            ...                [4, 7, -5]])
            >>> with xp.printoptions(precision=3, suppress=True):
            ...   xp.cosh(x)
            Array([[ 10.068,   1.543,   1.   ],
                   [ 27.308, 548.317,  74.21 ]], dtype=float32)
            >>> with xp.printoptions(precision=3, suppress=True):
            ...   xp.cos(1j * x)
            Array([[ 10.068+0.j,   1.543+0.j,   1.   +0.j],
                   [ 27.308+0.j, 548.317+0.j,  74.21 +0.j]], dtype=complex64, weak_type=True)

            For complex-valued input:

            >>> with xp.printoptions(precision=3, suppress=True):
            ...   xp.cosh(5+1j)
            Array(40.096+62.44j, dtype=complex64, weak_type=True)
            >>> with xp.printoptions(precision=3, suppress=True):
            ...   xp.cos(1j * (5+1j))
            Array(40.096+62.44j, dtype=complex64, weak_type=True)
        """
        ...

    def cross(self, a, b, axisa: int = -1, axisb: int = -1, axisc: int = -1,
              axis: int | None = None) -> Array:
        r"""Compute the (batched) cross product of two arrays.

        Array API implementation of :func:`numpy.cross`.

        This computes the 2-dimensional or 3-dimensional cross product,

        .. math::

            c = a \times b

        In 3 dimensions, ``c`` is a length-3 array. In 2 dimensions, ``c`` is
        a scalar.

        Args:
            a: N-dimensional array. ``a.shape[axisa]`` indicates the dimension of
                the cross product, and must be 2 or 3.
            b: N-dimensional array. Must have ``b.shape[axisb] == a.shape[axisb]``,
                and other dimensions of ``a`` and ``b`` must be broadcast compatible.
            axisa: specicy the axis of ``a`` along which to compute the cross product.
            axisb: specicy the axis of ``b`` along which to compute the cross product.
            axisc: specicy the axis of ``c`` along which the cross product result
                will be stored.
            axis: if specified, this overrides ``axisa``, ``axisb``, and ``axisc``
                with a single value.

        Returns:
            The array ``c`` containing the (batched) cross product of ``a`` and ``b``
            along the specified axes.

        See also:
            - :func:`linalg.cross`: an array API compatible function for
                computing cross products over 3-vectors.

        Examples:
            A 2-dimensional cross product returns a scalar:

            >>> a = xp.array([1, 2])
            >>> b = xp.array([3, 4])
            >>> xp.cross(a, b)
            Array(-2, dtype=int32)

            A 3-dimensional cross product returns a length-3 vector:

            >>> a = xp.array([1, 2, 3])
            >>> b = xp.array([4, 5, 6])
            >>> xp.cross(a, b)
            Array([-3,  6, -3], dtype=int32)

            With multi-dimensional inputs, the cross-product is computed along
            the last axis by default. Here's a batched 3-dimensional cross
            product, operating on the rows of the inputs:

            >>> a = xp.array([[1, 2, 3],
            ...               [3, 4, 3]])
            >>> b = xp.array([[2, 3, 2],
            ...               [4, 5, 6]])
            >>> xp.cross(a, b)
            Array([[-5,  4, -1],
                   [ 9, -6, -1]], dtype=int32)

            Specifying axis=0 makes this a batched 2-dimensional cross product,
            operating on the columns of the inputs:

            >>> xp.cross(a, b, axis=0)
            Array([-2, -2, 12], dtype=int32)

            Equivalently, we can independently specify the axis of the inputs ``a``
            and ``b`` and the output ``c``:

            >>> xp.cross(a, b, axisa=0, axisb=0, axisc=0)
            Array([-2, -2, 12], dtype=int32)
        """
        ...

    def delete(self, arr: ArrayLike, obj: ArrayLike | slice, axis: int | None = None) -> Array:
        """Delete entry or entries from an array.

        Array API implementation of :func:`numpy.delete`.

        Args:
            arr: array from which entries will be deleted.
            obj: index, indices, or slice to be deleted.
            axis: axis along which entries will be deleted.

        Returns:
            Copy of ``arr`` with specified indices deleted.

        Note:
            ``delete()`` usually requires the index specification to be static. If the
            index is an integer array that is guaranteed to contain unique entries, you
            may specify ``assume_unique_indices=True`` to perform the operation in a
            manner that does not require static indices.

        See also:
            - :func:`insert`: insert entries into an array.

        Examples:
            Delete entries from a 1D array:

            >>> a = xp.array([4, 5, 6, 7, 8, 9])
            >>> xp.delete(a, 2)
            Array([4, 5, 7, 8, 9], dtype=int32)
            >>> xp.delete(a, slice(1, 4))  # delete a[1:4]
            Array([4, 8, 9], dtype=int32)
            >>> xp.delete(a, slice(None, None, 2))  # delete a[::2]
            Array([5, 7, 9], dtype=int32)

            Delete entries from a 2D array along a specified axis:

            >>> a2 = xp.array([[4, 5, 6],
            ...                [7, 8, 9]])
            >>> xp.delete(a2, 1, axis=1)
            Array([[4, 6],
                   [7, 9]], dtype=int32)

            Delete multiple entries via a sequence of indices:

            >>> indices = xp.array([0, 1, 3])
            >>> xp.delete(a, indices)
            Array([6, 8, 9], dtype=int32)
        """
        ...

    def dot(self, a: ArrayLike, b: ArrayLike) -> Array:
        """Compute the dot product of two arrays.

        Array API implementation of :func:`numpy.dot`.

        This differs from :func:`matmul` in two respects:

        - if either ``a`` or ``b`` is a scalar, the result of ``dot`` is equivalent to
        :func:`multiply`, while the result of ``matmul`` is an error.
        - if ``a`` and ``b`` have more than 2 dimensions, the batch indices are
        stacked rather than broadcast.

        Args:
            a: first input array, of shape ``(..., N)``.
            b: second input array. Must have shape ``(N,)`` or ``(..., N, M)``.
                In the multi-dimensional case, leading dimensions must be broadcast-compatible
                with the leading dimensions of ``a``.

        Returns:
            array containing the dot product of the inputs, with batch dimensions of
            ``a`` and ``b`` stacked rather than broadcast.

        See also:
            - :func:`matmul`: broadcasted batched matmul.

        Examples:
            For scalar inputs, ``dot`` computes the element-wise product:

            >>> x = xp.array([1, 2, 3])
            >>> xp.dot(x, 2)
            Array([2, 4, 6], dtype=int32)

            For vector or matrix inputs, ``dot`` computes the vector or matrix product:

            >>> M = xp.array([[2, 3, 4],
            ...               [5, 6, 7],
            ...               [8, 9, 0]])
            >>> xp.dot(M, x)
            Array([20, 38, 26], dtype=int32)
            >>> xp.dot(M, M)
            Array([[ 51,  60,  29],
                   [ 96, 114,  62],
                   [ 61,  78,  95]], dtype=int32)

            For higher-dimensional matrix products, batch dimensions are stacked, whereas
            in :func:`~matmul` they are broadcast. For example:

            >>> a = xp.zeros((3, 2, 4))
            >>> b = xp.zeros((3, 4, 1))
            >>> xp.dot(a, b).shape
            (3, 2, 3, 1)
            >>> xp.matmul(a, b).shape
            (3, 2, 1)
        """
        ...

    def eye(self, N: DimSize, M: DimSize | None = None, k: int | ArrayLike = 0,
            dtype: DTypeLike | None = None, *, device: Device | None = None) -> Array:
        """Create a square or rectangular identity matrix

        Array API implementation of :func:`numpy.eye`.

        Args:
            N: integer specifying the first dimension of the array.
            M: optional integer specifying the second dimension of the array;
                defaults to the same value as ``N``.
            k: optional integer specifying the offset of the diagonal. Use positive
                values for upper diagonals, and negative values for lower diagonals.
                Default is zero.
            dtype: optional dtype; defaults to floating point.
            device: optional :class:`~xp.Device` to which the created array will be
                committed.

        Returns:
            Identity array of shape ``(N, M)``, or ``(N, N)`` if ``M`` is not specified.

        See also:
            :func:`identity`: Simpler API for generating square identity matrices.

        Examples:
            A simple 3x3 identity matrix:

            >>> xp.eye(3)
            Array([[1., 0., 0.],
                   [0., 1., 0.],
                   [0., 0., 1.]], dtype=float32)

            Integer identity matrices with offset diagonals:

            >>> xp.eye(3, k=1, dtype=int)
            Array([[0, 1, 0],
                   [0, 0, 1],
                   [0, 0, 0]], dtype=int32)
            >>> xp.eye(3, k=-1, dtype=int)
            Array([[0, 0, 0],
                   [1, 0, 0],
                   [0, 1, 0]], dtype=int32)

            Non-square identity matrix:

            >>> xp.eye(3, 5, k=1)
            Array([[0., 1., 0., 0., 0.],
                   [0., 0., 1., 0., 0.],
                   [0., 0., 0., 1., 0.]], dtype=float32)
        """
        ...

    def exp(self, x: ArrayLike, /) -> RealArray:
        """Calculate element-wise exponential of the input.

        Array API implementation of :obj:`numpy.exp`.

        Args:
            x: input array or scalar

        Returns:
            An array containing the exponential of each element in ``x``, promotes to
            inexact dtype.

        See also:
            - :func:`log`: Calculates element-wise logarithm of the input.
            - :func:`expm1`: Calculates :math:`e^x-1` of each element of the
                input.
            - :func:`exp2`: Calculates base-2 exponential of each element of
                the input.

        Examples:
            ``xp.exp`` follows the properties of exponential such as :math:`e^{(a+b)}
            = e^a * e^b`.

            >>> x1 = xp.array([2, 4, 3, 1])
            >>> x2 = xp.array([1, 3, 2, 3])
            >>> with xp.printoptions(precision=2, suppress=True):
            ...   print(xp.exp(x1+x2))
            [  20.09 1096.63  148.41   54.6 ]
            >>> with xp.printoptions(precision=2, suppress=True):
            ...   print(xp.exp(x1)*xp.exp(x2))
            [  20.09 1096.63  148.41   54.6 ]

            This property holds for complex input also:

            >>> xp.allclose(xp.exp(3-4j), xp.exp(3)*xp.exp(-4j))
            Array(True, dtype=bool)
        """
        ...

    def expand_dims(self, a: ArrayLike, axis: int | Sequence[int]) -> Array:
        """Insert dimensions of length 1 into array

        Array API implementation of :func:`numpy.expand_dims`.

        Args:
            a: input array
            axis: integer or sequence of integers specifying positions of axes to add.

        Returns:
            Copy of ``a`` with added dimensions.

        See Also:
            - :func:`squeeze`: inverse of this operation, i.e. remove length-1 dimensions.

        Examples:
            >>> x = xp.array([1, 2, 3])
            >>> x.shape
            (3,)

            Expand the leading dimension:

            >>> xp.expand_dims(x, 0)
            Array([[1, 2, 3]], dtype=int32)
            >>> _.shape
            (1, 3)

            Expand the trailing dimension:

            >>> xp.expand_dims(x, 1)
            Array([[1],
                   [2],
                   [3]], dtype=int32)
            >>> _.shape
            (3, 1)

            Expand multiple dimensions:

            >>> xp.expand_dims(x, (0, 1, 3))
            Array([[[[1],
                     [2],
                     [3]]]], dtype=int32)
            >>> _.shape
            (1, 1, 3, 1)

            Dimensions can also be expanded more succinctly by indexing with ``None``:

            >>> x[None]  # equivalent to xp.expand_dims(x, 0)
            Array([[1, 2, 3]], dtype=int32)
            >>> x[:, None]  # equivalent to xp.expand_dims(x, 1)
            Array([[1],
                   [2],
                   [3]], dtype=int32)
            >>> x[None, None, :, None]  # equivalent to xp.expand_dims(x, (0, 1, 3))
            Array([[[[1],
                     [2],
                     [3]]]], dtype=int32)
        """
        ...

    def floor(self, x: ArrayLike, /) -> Array:
        """Round input to the nearest integer downwards.

        Array API implementation of :obj:`numpy.floor`.

        Args:
            x: input array or scalar. Must not have complex dtype.

        Returns:
            An array with same shape and dtype as ``x`` containing the values rounded to
            the nearest integer that is less than or equal to the value itself.

        See also:
            - :func:`fix`: Rounds the input to the nearest interger towards zero.
            - :func:`trunc`: Rounds the input to the nearest interger towards
                zero.
            - :func:`ceil`: Rounds the input up to the nearest integer.

        Examples:
            >>> key = jax.random.key(42)
            >>> x = jax.random.uniform(key, (3, 3), minval=-5, maxval=5)
            >>> with xp.printoptions(precision=2, suppress=True):
            ...     print(x)
            [[ 1.44 -1.77 -3.07]
             [ 3.86  2.25 -3.08]
             [-1.55 -2.48  1.32]]
            >>> xp.floor(x)
            Array([[ 1., -2., -4.],
                   [ 3.,  2., -4.],
                   [-2., -3.,  1.]], dtype=float32)
        """
        ...

    def fromstring(self, string: str, dtype: DTypeLike = float, count: int = -1, *,
                   sep: str) -> Array:
        """Convert a string of text into 1-D JAX array.

        Array API implementation of :func:`numpy.fromstring`.

        Args:
            string: input string containing the data.
            dtype: optional. Desired data type for the array. Default is ``float``.
            count: optional integer specifying the number of items to read from the string.
                If -1 (default), all items are read.
            sep: the string used to separate values in the input string.

        Returns:
            A 1-D array containing the parsed data from the input string.

        Examples:
            >>> xp.fromstring("1 2 3", dtype=int, sep=" ")
            Array([1, 2, 3], dtype=int32)
            >>> xp.fromstring("0.1, 0.2, 0.3", dtype=float, count=2, sep=",")
            Array([0.1, 0.2], dtype=float32)
        """
        ...

    def full(self, shape: Any, fill_value: ArrayLike, dtype: DTypeLike | None = None, *,
             device: Device | None = None) -> Array:
        """Create an array full of a specified value.

        Array API implementation of :func:`numpy.full`.

        Args:
            shape: int or sequence of ints specifying the shape of the created array.
            fill_value: scalar or array with which to fill the created array.
            dtype: optional dtype for the created array; defaults to the dtype of the
                fill value.
            device: (optional) :class:`~xp.Device` to which the created array will be
                committed.

        Returns:
            Array of the specified shape and dtype, on the specified device if specified.

        See also:
            - :func:`full_like`
            - :func:`empty`
            - :func:`zeros`
            - :func:`ones`

        Examples:
            >>> xp.full(4, 2, dtype=float)
            Array([2., 2., 2., 2.], dtype=float32)
            >>> xp.full((2, 3), 0, dtype=bool)
            Array([[False, False, False],
                   [False, False, False]], dtype=bool)

            `fill_value` may also be an array that is broadcast to the specified shape:

            >>> xp.full((2, 3), fill_value=xp.arange(3))
            Array([[0, 1, 2],
                   [0, 1, 2]], dtype=int32)
        """
        ...

    def identity(self, n: DimSize, dtype: DTypeLike | None = None) -> Array:
        """Create a square identity matrix

        Array API implementation of :func:`numpy.identity`.

        Args:
            n: integer specifying the size of each array dimension.
            dtype: optional dtype; defaults to floating point.

        Returns:
            Identity array of shape ``(n, n)``.

        See also:
            :func:`eye`: non-square and/or offset identity matrices.

        Examples:
            A simple 3x3 identity matrix:

            >>> xp.identity(3)
            Array([[1., 0., 0.],
                   [0., 1., 0.],
                   [0., 0., 1.]], dtype=float32)

            A 2x2 integer identity matrix:

            >>> xp.identity(2, dtype=int)
            Array([[1, 0],
                   [0, 1]], dtype=int32)
            """
        ...

    def invert(self, x: ArrayLike, /) -> Array:
        """Compute the bitwise inversion of an input.

        Array API implementation of :func:`numpy.invert`. This function provides the
        implementation of the ``~`` operator for Array API arrays.

        Args:
            x: input array, must be boolean or integer typed.

        Returns:
            An array of the same shape and dtype as ```x``, with the bits inverted.

        See also:
            - :func:`bitwise_invert`: Array API alias of this function.
            - :func:`logical_not`: Invert after casting input to boolean.

        Examples:
            >>> x = xp.arange(5, dtype='uint8')
            >>> print(x)
            [0 1 2 3 4]
            >>> print(xp.invert(x))
            [255 254 253 252 251]

            This function implements the unary ``~`` operator for Array API arrays:

            >>> print(~x)
            [255 254 253 252 251]

            :func:`invert` operates bitwise on the input, and so the meaning of its
            output may be more clear by showing the bitwise representation:

            >>> with xp.printoptions(formatter={'int': lambda x: format(x, '#010b')}):
            ...   print(f"{x  = }")
            ...   print(f"{~x = }")
            x  = Array([0b00000000, 0b00000001, 0b00000010, 0b00000011, 0b00000100], dtype=uint8)
            ~x = Array([0b11111111, 0b11111110, 0b11111101, 0b11111100, 0b11111011], dtype=uint8)

            For boolean inputs, :func:`invert` is equivalent to :func:`logical_not`:

            >>> x = xp.array([True, False, True, True, False])
            >>> xp.invert(x)
            Array([False,  True, False, False,  True], dtype=bool)
        """
        ...

    def isclose(self, a: ArrayLike, b: ArrayLike, rtol: ArrayLike = 1e-05,
                atol: ArrayLike = 1e-08, equal_nan: bool = False) -> Array:
        r"""Check if the elements of two arrays are approximately equal within a tolerance.

        Array API implementation of :func:`numpy.allclose`.

        Essentially this function evaluates the following condition:

        .. math::

            |a - b| \le \mathtt{atol} + \mathtt{rtol} * |b|

        ``xp.inf`` in ``a`` will be considered equal to ``xp.inf`` in ``b``.

        Args:
            a: first input array to compare.
            b: second input array to compare.
            rtol: relative tolerance used for approximate equality. Default = 1e-05.
            atol: absolute tolerance used for approximate equality. Default = 1e-08.
            equal_nan: Boolean. If ``True``, NaNs in ``a`` will be considered
                equal to NaNs in ``b``. Default is ``False``.

        Returns:
            A new array containing boolean values indicating whether the input arrays
            are element-wise approximately equal within the specified tolerances.

        See Also:
            - :func:`allclose`
            - :func:`equal`

        Examples:
            >>> xp.isclose(xp.array([1e6, 2e6, xp.inf]), xp.array([1e6, 2e7, xp.inf]))
            Array([ True, False,  True], dtype=bool)
            >>> xp.isclose(xp.array([1e6, 2e6, 3e6]),
            ...            xp.array([1.00008e6, 2.00008e7, 3.00008e8]), rtol=1e3)
            Array([ True,  True,  True], dtype=bool)
            >>> xp.isclose(xp.array([1e6, 2e6, 3e6]),
            ...            xp.array([1.00001e6, 2.00002e6, 3.00009e6]), atol=1e3)
            Array([ True,  True,  True], dtype=bool)
            >>> xp.isclose(xp.array([xp.nan, 1, 2]),
            ...            xp.array([xp.nan, 1, 2]), equal_nan=True)
            Array([ True,  True,  True], dtype=bool)
        """
        ...

    def lexsort(self, keys: Array | Sequence[ArrayLike], axis: int = -1) -> Array:
        """Sort a sequence of keys in lexicographic order.

        Array API implementation of :func:`numpy.lexsort`.

        Args:
            keys: a sequence of arrays to sort; all arrays must have the same shape.
                The last key in the sequence is used as the primary key.
            axis: the axis along which to sort (default: -1).

        Returns:
            An array of integers of shape ``keys[0].shape`` giving the indices of the
            entries in lexicographically-sorted order.

        See also:
            - :func:`argsort`: sort a single entry by index.

        Examples:
        :func:`lexsort` with a single key is equivalent to :func:`argsort`:

        >>> key1 = xp.array([4, 2, 3, 2, 5])
        >>> xp.lexsort([key1])
        Array([1, 3, 2, 0, 4], dtype=int32)
        >>> xp.argsort(key1)
        Array([1, 3, 2, 0, 4], dtype=int32)

        With multiple keys, :func:`lexsort` uses the last key as the primary key:

        >>> key2 = xp.array([2, 1, 1, 2, 2])
        >>> xp.lexsort([key1, key2])
        Array([1, 2, 3, 0, 4], dtype=int32)

        The meaning of the indices become more clear when printing the sorted keys:

        >>> indices = xp.lexsort([key1, key2])
        >>> print(f"{key1[indices]}\\n{key2[indices]}")
        [2 3 2 4 5]
        [1 1 2 2 2]

        Notice that the elements of ``key2`` appear in order, and within the sequences
        of duplicated values the corresponding elements of ```key1`` appear in order.

        For multi-dimensional inputs, :func:`lexsort` defaults to sorting along the
        last axis:

        >>> key1 = xp.array([[2, 4, 2, 3],
        ...                  [3, 1, 2, 2]])
        >>> key2 = xp.array([[1, 2, 1, 3],
        ...                  [2, 1, 2, 1]])
        >>> xp.lexsort([key1, key2])
        Array([[0, 2, 1, 3],
               [1, 3, 2, 0]], dtype=int32)

        A different sort axis can be chosen using the ``axis`` keyword; here we sort
        along the leading axis:

        >>> xp.lexsort([key1, key2], axis=0)
        Array([[0, 1, 0, 1],
               [1, 0, 1, 0]], dtype=int32)
        """
        ...

    @overload
    def linspace(self, start: ArrayLike, stop: ArrayLike, num: int = 50,
                 endpoint: bool = True, retstep: Literal[False] = False,
                 dtype: DTypeLike | None = None, axis: int = 0, *,
                 device: Device | None = None) -> Array:
        ...

    @overload
    def linspace(self, start: ArrayLike, stop: ArrayLike, num: int,
                 endpoint: bool, retstep: Literal[True],
                 dtype: DTypeLike | None = None, axis: int = 0, *,
                 device: Device | None = None) -> tuple[Array, Array]:
        ...

    @overload
    def linspace(self, start: ArrayLike, stop: ArrayLike, num: int = 50,
                 endpoint: bool = True, *, retstep: Literal[True],
                 dtype: DTypeLike | None = None, axis: int = 0,
                 device: Device | None = None) -> tuple[Array, Array]:
        ...

    @overload
    def linspace(self, start: ArrayLike, stop: ArrayLike, num: int = 50,
                 endpoint: bool = True, retstep: bool = False,
                 dtype: DTypeLike | None = None, axis: int = 0, *,
                 device: Device | None = None) -> Array | tuple[Array, Array]:
        ...

    def linspace(self, start: ArrayLike, stop: ArrayLike, num: int = 50,
                 endpoint: bool = True, retstep: bool = False,
                 dtype: DTypeLike | None = None, axis: int = 0, *,
                 device: Device | None = None) -> Array | tuple[Array, Array]:
        """Return evenly-spaced numbers within an interval.

        Array API implementation of :func:`numpy.linspace`.

        Args:
            start: scalar or array of starting values.
            stop: scalar or array of stop values.
            num: number of values to generate. Default: 50.
            endpoint: if True (default) then include the ``stop`` value in the result.
                If False, then exclude the ``stop`` value.
            retstep: If True, then return a ``(result, step)`` tuple, where ``step`` is the
                interval between adjacent values in ``result``.
            axis: integer axis along which to generate the linspace. Defaults to zero.
            device: optional :class:`~xp.Device` to which the created array will be
                committed.

        Returns:
            An array ``values``, or a tuple ``(values, step)`` if ``retstep`` is True, where:

            - ``values`` is an array of evenly-spaced values from ``start`` to ``stop``
            - ``step`` is the interval between adjacent values.

        See also:
            - :func:`arange`: Generate ``N`` evenly-spaced values given a starting
                point and a step
            - :func:`logspace`: Generate logarithmically-spaced values.
            - :func:`geomspace`: Generate geometrically-spaced values.

        Examples:
            List of 5 values between 0 and 10:

            >>> xp.linspace(0, 10, 5)
            Array([ 0. ,  2.5,  5. ,  7.5, 10. ], dtype=float32)

            List of 8 values between 0 and 10, excluding the endpoint:

            >>> xp.linspace(0, 10, 8, endpoint=False)
            Array([0.  , 1.25, 2.5 , 3.75, 5.  , 6.25, 7.5 , 8.75], dtype=float32)

            List of values and the step size between them

            >>> vals, step = xp.linspace(0, 10, 9, retstep=True)
            >>> vals
            Array([ 0.  ,  1.25,  2.5 ,  3.75,  5.  ,  6.25,  7.5 ,  8.75, 10.  ], dtype=float32)
            >>> step
            Array(1.25, dtype=float32)

            Multi-dimensional linspace:

            >>> start = xp.array([0, 5])
            >>> stop = xp.array([5, 10])
            >>> xp.linspace(start, stop, 5)
            Array([[ 0.  ,  5.  ],
                   [ 1.25,  6.25],
                   [ 2.5 ,  7.5 ],
                   [ 3.75,  8.75],
                   [ 5.  , 10.  ]], dtype=float32)
        """
        ...

    def log(self, x: ArrayLike, /) -> RealArray:
        """Calculate element-wise natural logarithm of the input.

        Array API implementation of :obj:`numpy.log`.

        Args:
            x: input array or scalar.

        Returns:
            An array containing the logarithm of each element in ``x``, promotes to inexact
            dtype.

        See also:
            - :func:`exp`: Calculates element-wise exponential of the input.
            - :func:`log2`: Calculates base-2 logarithm of each element of input.
            - :func:`log1p`: Calculates element-wise logarithm of one plus input.

        Examples:
            ``xp.log`` and ``xp.exp`` are inverse functions of each other. Applying
            ``xp.log`` on the result of ``xp.exp(x)`` yields the original input ``x``.

            >>> x = xp.array([2, 3, 4, 5])
            >>> xp.log(xp.exp(x))
            Array([2., 3., 4., 5.], dtype=float32)

            Using ``xp.log`` we can demonstrate well-known properties of logarithms, such
            as :math:`log(a*b) = log(a)+log(b)`.

            >>> x1 = xp.array([2, 1, 3, 1])
            >>> x2 = xp.array([1, 3, 2, 4])
            >>> xp.allclose(xp.log(x1*x2), xp.log(x1)+xp.log(x2))
            Array(True, dtype=bool)
        """
        ...

    def max(self, a: ArrayLike, axis: Axis = None, keepdims: bool = False,
            initial: ArrayLike | None = None, where: ArrayLike | None = None) -> Array:
        r"""Return the maximum of the array elements along a given axis.

        Array API implementation of :func:`numpy.max`.

        Args:
            a: Input array.
            axis: int or array, default=None. Axis along which the maximum to be computed.
                If None, the maximum is computed along all the axes.
            keepdims: bool, default=False. If true, reduced axes are left in the result
                with size 1.
            initial: int or array, default=None. Initial value for the maximum.
                where: int or array of boolean dtype, default=None. The elements to be used
                in the maximum. Array should be broadcast compatible to the input.
                ``initial`` must be specified when ``where`` is used.

        Returns:
            An array of maximum values along the given axis.

        See also:
            - :func:`min`: Compute the minimum of array elements along a given
                axis.
            - :func:`sum`: Compute the sum of array elements along a given axis.
            - :func:`prod`: Compute the product of array elements along a given
                axis.

        Examples:

            By default, ``xp.max`` computes the maximum of elements along all the axes.

            >>> x = xp.array([[9, 3, 4, 5],
            ...               [5, 2, 7, 4],
            ...               [8, 1, 3, 6]])
            >>> xp.max(x)
            Array(9, dtype=int32)

            If ``axis=1``, the maximum will be computed along axis 1.

            >>> xp.max(x, axis=1)
            Array([9, 7, 8], dtype=int32)

            If ``keepdims=True``, ``ndim`` of the output will be same of that of the input.

            >>> xp.max(x, axis=1, keepdims=True)
            Array([[9],
                   [7],
                   [8]], dtype=int32)

            To include only specific elements in computing the maximum, you can use
            ``where``. It can either have same dimension as input

            >>> where=xp.array([[0, 0, 1, 0],
            ...                 [0, 0, 1, 1],
            ...                 [1, 1, 1, 0]], dtype=bool)
            >>> xp.max(x, axis=1, keepdims=True, initial=0, where=where)
            Array([[4],
                   [7],
                   [8]], dtype=int32)

            or must be broadcast compatible with input.

            >>> where = xp.array([[False],
            ...                    [False],
            ...                    [False]])
            >>> xp.max(x, axis=0, keepdims=True, initial=0, where=where)
            Array([[0, 0, 0, 0]], dtype=int32)
        """
        ...

    def mean(self, a: ArrayLike, axis: Axis = None, dtype: DTypeLike | None = None,
             keepdims: bool = False, *, where: ArrayLike | None = None
             ) -> Array:
        r"""Return the mean of array elements along a given axis.

        Array API implementation of :func:`numpy.mean`.

        Args:
            a: input array.
            axis: optional, int or sequence of ints, default=None. Axis along which the
            mean to be computed. If None, mean is computed along all the axes.
            dtype: The type of the output array. Default=None.
            keepdims: bool, default=False. If true, reduced axes are left in the result
                with size 1.
            where: optional, boolean array, default=None. The elements to be used in the
                mean. Array should be broadcast compatible to the input.

        Returns:
            An array of the mean along the given axis.

        See also:
            - :func:`average`: Compute the weighted average of array elements
            - :func:`sum`: Compute the sum of array elements.

        Examples:
            By default, the mean is computed along all the axes.

            >>> x = xp.array([[1, 3, 4, 2],
            ...               [5, 2, 6, 3],
            ...               [8, 1, 2, 9]])
            >>> xp.mean(x)
            Array(3.8333335, dtype=float32)

            If ``axis=1``, the mean is computed along axis 1.

            >>> xp.mean(x, axis=1)
            Array([2.5, 4. , 5. ], dtype=float32)

            If ``keepdims=True``, ``ndim`` of the output is equal to that of the input.

            >>> xp.mean(x, axis=1, keepdims=True)
            Array([[2.5],
                   [4. ],
                   [5. ]], dtype=float32)

            To use only specific elements of ``x`` to compute the mean, you can use
            ``where``.

            >>> where = xp.array([[1, 0, 1, 0],
            ...                   [0, 1, 0, 1],
            ...                   [1, 1, 0, 1]], dtype=bool)
            >>> xp.mean(x, axis=1, keepdims=True, where=where)
            Array([[2.5],
                   [2.5],
                   [6. ]], dtype=float32)
        """
        ...

    def median(self, a: ArrayLike, axis: int | tuple[int, ...] | None = None,
               keepdims: bool = False) -> Array:
        r"""Return the median of array elements along a given axis.

        Array API implementation of :func:`numpy.median`.

        Args:
            a: input array.
            axis: optional, int or sequence of ints, default=None. Axis along which the
                median to be computed. If None, median is computed for the flattened array.
            keepdims: bool, default=False. If true, reduced axes are left in the result
                with size 1.

        Returns:
            An array of the median along the given axis.

        See also:
            - :func:`mean`: Compute the mean of array elements over a given axis.
            - :func:`max`: Compute the maximum of array elements over given axis.
            - :func:`min`: Compute the minimum of array elements over given axis.

        Examples:
            By default, the median is computed for the flattened array.

            >>> x = xp.array([[2, 4, 7, 1],
            ...               [3, 5, 9, 2],
            ...               [6, 1, 8, 3]])
            >>> xp.median(x)
            Array(3.5, dtype=float32)

            If ``axis=1``, the median is computed along axis 1.

            >>> xp.median(x, axis=1)
            Array([3. , 4. , 4.5], dtype=float32)

            If ``keepdims=True``, ``ndim`` of the output is equal to that of the input.

            >>> xp.median(x, axis=1, keepdims=True)
            Array([[3. ],
                   [4. ],
                   [4.5]], dtype=float32)
        """
        ...

    def meshgrid(self, *xi: ArrayLike, copy: bool = True, sparse: bool = False,
                 indexing: str = 'xy') -> list[Array]:
        """Construct N-dimensional grid arrays from N 1-dimensional vectors.

        Array API implementation of :func:`numpy.meshgrid`.

        Args:
            xi: N arrays to convert to a grid.
            copy: whether to copy the input arrays. Array API supports only ``copy=True``,
                though under JIT compilation the compiler may opt to avoid copies.
            sparse: if False (default), then each returned arrays will be of shape
                ``[len(x1), len(x2), ..., len(xN)]``. If False, then returned arrays
                will be of shape ``[1, 1, ..., len(xi), ..., 1, 1]``.
                indexing: options are ``'xy'`` for cartesian indexing (default) or ``'ij'``
                for matrix indexing.

        Returns:
            A length-N list of grid arrays.

        See also:
            - :obj:`mgrid`: create a meshgrid using indexing syntax.
            - :obj:`ogrid`: create an open meshgrid using indexing syntax.

        Examples:
            For the following examples, we'll use these 1D arrays as inputs:

            >>> x = xp.array([1, 2])
            >>> y = xp.array([10, 20, 30])

            2D cartesian mesh grid:

            >>> x_grid, y_grid = xp.meshgrid(x, y)
            >>> print(x_grid)
            [[1 2]
             [1 2]
             [1 2]]
            >>> print(y_grid)
            [[10 10]
             [20 20]
             [30 30]]

            2D sparse cartesian mesh grid:

            >>> x_grid, y_grid = xp.meshgrid(x, y, sparse=True)
            >>> print(x_grid)
            [[1 2]]
            >>> print(y_grid)
            [[10]
             [20]
             [30]]

            2D matrix-index mesh grid:

            >>> x_grid, y_grid = xp.meshgrid(x, y, indexing='ij')
            >>> print(x_grid)
            [[1 1 1]
             [2 2 2]]
            >>> print(y_grid)
            [[10 20 30]
             [10 20 30]]
        """
        ...

    def min(self, a: ArrayLike, axis: Axis = None, keepdims: bool = False,
            initial: ArrayLike | None = None, where: ArrayLike | None = None) -> Array:
        r"""Return the minimum of array elements along a given axis.

        Array API implementation of :func:`numpy.min`.

        Args:
            a: Input array.
            axis: int or array, default=None. Axis along which the minimum to be computed.
                If None, the minimum is computed along all the axes.
            keepdims: bool, default=False. If true, reduced axes are left in the result
                with size 1.
            initial: int or array, Default=None. Initial value for the minimum.
            where: int or array, default=None. The elements to be used in the minimum.
                Array should be broadcast compatible to the input. ``initial`` must be
                specified when ``where`` is used.

        Returns:
            An array of minimum values along the given axis.

        See also:
            - :func:`max`: Compute the maximum of array elements along a given
                axis.
            - :func:`sum`: Compute the sum of array elements along a given axis.
            - :func:`prod`: Compute the product of array elements along a given
                axis.

        Examples:
            By default, the minimum is computed along all the axes.

            >>> x = xp.array([[2, 5, 1, 6],
            ...               [3, -7, -2, 4],
            ...               [8, -4, 1, -3]])
            >>> xp.min(x)
            Array(-7, dtype=int32)

            If ``axis=1``, the minimum is computed along axis 1.

            >>> xp.min(x, axis=1)
            Array([ 1, -7, -4], dtype=int32)

            If ``keepdims=True``, ``ndim`` of the output will be same of that of the input.

            >>> xp.min(x, axis=1, keepdims=True)
            Array([[ 1],
                   [-7],
                   [-4]], dtype=int32)

            To include only specific elements in computing the minimum, you can use
            ``where``. ``where`` can either have same dimension as input.

            >>> where=xp.array([[1, 0, 1, 0],
            ...                 [0, 0, 1, 1],
            ...                 [1, 1, 1, 0]], dtype=bool)
            >>> xp.min(x, axis=1, keepdims=True, initial=0, where=where)
            Array([[ 0],
                   [-2],
                   [-4]], dtype=int32)

            or must be broadcast compatible with input.

            >>> where = xp.array([[False],
            ...                   [False],
            ...                   [False]])
            >>> xp.min(x, axis=0, keepdims=True, initial=0, where=where)
            Array([[0, 0, 0, 0]], dtype=int32)
        """
        ...

    def nan_to_num(self, x: ArrayLike, copy: bool = True, nan: ArrayLike = 0.0,
                   posinf: ArrayLike | None = None, neginf: ArrayLike | None = None
                   ) -> Array:
        """Replace NaN and infinite entries in an array.

        Array API implementation of :func:`numpy.nan_to_num`.

        Args:
            x: array of values to be replaced. If it does not have an inexact
                dtype it will be returned unmodified.
            copy: Whether to create a copy of x (True) or to replace values
                in-place (False). The in-place operation only occurs if casting
                to an array does not require a copy. Default is True.
            nan: value to substitute for NaN entries. Defaults to 0.0.
            posinf: value to substitute for positive infinite entries.
                Defaults to the maximum representable value.
            neginf: value to substitute for positive infinite entries.
                Defaults to the minimum representable value.

        Returns:
            A copy of ``x`` with the requested substitutions.

        See also:
            - :func:`isnan`: return True where the array contains NaN
            - :func:`isposinf`: return True where the array contains +inf
            - :func:`isneginf`: return True where the array contains -inf

        Examples:
            >>> x = xp.array([0, xp.nan, 1, xp.inf, 2, -xp.inf])

            Default substitution values:

            >>> xp.nan_to_num(x)
            Array([ 0.0000000e+00,  0.0000000e+00,  1.0000000e+00,  3.4028235e+38,
                    2.0000000e+00, -3.4028235e+38], dtype=float32)

            Overriding substitutions for ``-inf`` and ``+inf``:

            >>> xp.nan_to_num(x, posinf=999, neginf=-999)
            Array([   0.,    0.,    1.,  999.,    2., -999.], dtype=float32)

            If you only wish to substitute for NaN values while leaving ``inf`` values
            untouched, using :func:`~where` with :func:`isnan` is a better option:

            >>> xp.where(xp.isnan(x), 0, x)
            Array([  0.,   0.,   1.,  inf,   2., -inf], dtype=float32)
        """
        ...

    def ones(self, shape: Any, dtype: DTypeLike | None = None, *, device: Device | None = None
             ) -> Array:
        """Create an array full of ones.

        Array API implementation of :func:`numpy.ones`.

        Args:
            shape: int or sequence of ints specifying the shape of the created array.
            dtype: optional dtype for the created array; defaults to floating point.
            device: (optional) :class:`~xp.Device` to which the created array will be
                committed.

        Returns:
            Array of the specified shape and dtype, on the specified device if specified.

        See also:
            - :func:`ones_like`
            - :func:`empty`
            - :func:`zeros`
            - :func:`full`

        Examples:
            >>> xp.ones(4)
            Array([1., 1., 1., 1.], dtype=float32)
            >>> xp.ones((2, 3), dtype=bool)
            Array([[ True,  True,  True],
                   [ True,  True,  True]], dtype=bool)
        """
        ...

    def prod(self, a: ArrayLike, axis: Axis = None, dtype: DTypeLike | None = None,
             keepdims: bool = False, initial: ArrayLike | None = None,
             where: ArrayLike | None = None, promote_integers: bool = True) -> Array:
        r"""Return product of the array elements over a given axis.

        Array API implementation of :func:`numpy.prod`.

        Args:
            a: Input array.
            axis: int or array, default=None. Axis along which the product to be computed.
                If None, the product is computed along all the axes.
            dtype: The type of the output array. Default=None.
            keepdims: bool, default=False. If true, reduced axes are left in the result
                with size 1.
            initial: int or array, Default=None. Initial value for the product.
            where: int or array, default=None. The elements to be used in the product.
                Array should be broadcast compatible to the input.
            promote_integers : bool, default=True. If True, then integer inputs will be
                promoted to the widest available integer dtype, following numpy's behavior.
                If False, the result will have the same dtype as the input.
                ``promote_integers`` is ignored if ``dtype`` is specified.

        Returns:
            An array of the product along the given axis.

        See also:
            - :func:`sum`: Compute the sum of array elements over a given axis.
            - :func:`max`: Compute the maximum of array elements over given axis.
            - :func:`min`: Compute the minimum of array elements over given axis.

        Examples:
            By default, ``xp.prod`` computes along all the axes.

            >>> x = xp.array([[1, 3, 4, 2],
            ...               [5, 2, 1, 3],
            ...               [2, 1, 3, 1]])
            >>> xp.prod(x)
            Array(4320, dtype=int32)

            If ``axis=1``, product is computed along axis 1.

            >>> xp.prod(x, axis=1)
            Array([24, 30,  6], dtype=int32)

            If ``keepdims=True``, ``ndim`` of the output is equal to that of the input.

            >>> xp.prod(x, axis=1, keepdims=True)
            Array([[24],
                   [30],
                   [ 6]], dtype=int32)

            To include only specific elements in the sum, you can use a``where``.

            >>> where=xp.array([[1, 0, 1, 0],
            ...                 [0, 0, 1, 1],
            ...                 [1, 1, 1, 0]], dtype=bool)
            >>> xp.prod(x, axis=1, keepdims=True, where=where)
            Array([[4],
                   [3],
                   [6]], dtype=int32)
            >>> where = xp.array([[False],
            ...                   [False],
            ...                   [False]])
            >>> xp.prod(x, axis=1, keepdims=True, where=where)
            Array([[1],
                   [1],
                   [1]], dtype=int32)
        """
        ...

    def ravel(self, a: ArrayLike, order: str = "C") -> Array:
        """Flatten array into a 1-dimensional shape.

        Array API implementation of :func:`numpy.ravel`.

        ``ravel(arr, order=order)`` is equivalent to ``reshape(arr, -1, order=order)``.

        Args:
            a: array to be flattened.
            order: ``'F'`` or ``'C'``, specifies whether the reshape should apply column-major
                (fortran-style, ``"F"``) or row-major (C-style, ``"C"``) order; default is ``"C"``.
                Array API does not support `order="A"` or `order="K"`.

        Returns:
            flattened copy of input array.

        See Also:
            - :meth:`ravel`: equivalent functionality via an array method.
            - :func:`reshape`: general array reshape.

        Examples:
            >>> x = xp.array([[1, 2, 3],
            ...               [4, 5, 6]])

            By default, ravel in C-style, row-major order

            >>> xp.ravel(x)
            Array([1, 2, 3, 4, 5, 6], dtype=int32)

            Optionally ravel in Fortran-style, column-major:

            >>> xp.ravel(x, order='F')
            Array([1, 4, 2, 5, 3, 6], dtype=int32)

            For convenience, the same functionality is available via the :meth:`ravel`
            method:

            >>> x.ravel()
            Array([1, 2, 3, 4, 5, 6], dtype=int32)
        """
        ...

    def reshape(self, a: ArrayLike, shape: DimSize | ShapeLike | None = None, order: str = "C"
                ) -> Array:
        """Return a reshaped copy of an array.

        Array API implementation of :func:`numpy.reshape`.

        Args:
            a: input array to reshape
            shape: integer or sequence of integers giving the new shape, which must match the
                size of the input array. If any single dimension is given size ``-1``, it will be
                replaced with a value such that the output has the correct size.
            order: ``'F'`` or ``'C'``, specifies whether the reshape should apply column-major
                (fortran-style, ``"F"``) or row-major (C-style, ``"C"``) order; default is ``"C"``.
                Array API does not support ``order="A"``.
                newshape: deprecated alias of the ``shape`` argument. Will result in a
                :class:`DeprecationWarning` if used.

        Returns:
            reshaped copy of input array with the specified shape.

        See Also:
            - :meth:`reshape`: equivalent functionality via an array method.
            - :func:`ravel`: flatten an array into a 1D shape.
            - :func:`squeeze`: remove one or more length-1 axes from an array's shape.

        Examples:
            >>> x = xp.array([[1, 2, 3],
            ...               [4, 5, 6]])
            >>> xp.reshape(x, 6)
            Array([1, 2, 3, 4, 5, 6], dtype=int32)
            >>> xp.reshape(x, (3, 2))
            Array([[1, 2],
                   [3, 4],
                   [5, 6]], dtype=int32)

            You can use ``-1`` to automatically compute a shape that is consistent with
            the input size:

            >>> xp.reshape(x, -1)  # -1 is inferred to be 6
            Array([1, 2, 3, 4, 5, 6], dtype=int32)
            >>> xp.reshape(x, (-1, 2))  # -1 is inferred to be 3
            Array([[1, 2],
                   [3, 4],
                   [5, 6]], dtype=int32)

            The default ordering of axes in the reshape is C-style row-major ordering.
            To use Fortran-style column-major ordering, specify ``order='F'``:

            >>> xp.reshape(x, 6, order='F')
            Array([1, 4, 2, 5, 3, 6], dtype=int32)
            >>> xp.reshape(x, (3, 2), order='F')
            Array([[1, 5],
                   [4, 3],
                   [2, 6]], dtype=int32)

            For convenience, this functionality is also available via the
            :meth:`reshape` method:

            >>> x.reshape(3, 2)
            Array([[1, 2],
                   [3, 4],
                   [5, 6]], dtype=int32)
        """
        ...

    def rint(self, x: ArrayLike, /) -> Array:
        """Rounds the elements of x to the nearest integer

        Array API implementation of :obj:`numpy.rint`.

        Args:
            x: Input array

        Returns:
            An array-like object containing the rounded elements of ``x``. Always promotes
            to inexact.

        Note:
            If an element of x is exactly half way, e.g. ``0.5`` or ``1.5``, rint will round
            to the nearest even integer.

        Examples:
            >>> x1 = xp.array([5, 4, 7])
            >>> xp.rint(x1)
            Array([5., 4., 7.], dtype=float32)

            >>> x2 = xp.array([-2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5, 4.5])
            >>> xp.rint(x2)
            Array([-2., -2., -0.,  0.,  2.,  2.,  4.,  4.], dtype=float32)

            >>> x3 = xp.array([-2.5+3.5j, 4.5-0.5j])
            >>> xp.rint(x3)
            Array([-2.+4.j,  4.-0.j], dtype=complex64)
        """
        ...

    def round(self, a: ArrayLike, decimals: int = 0) -> Array:
        """Round input evenly to the given number of decimals.

        Array API implementation of :func:`numpy.round`.

        Args:
            a: input array or scalar.
            decimals: int, default=0. Number of decimal points to which the input needs
                to be rounded. It must be specified statically. Not implemented for
                ``decimals < 0``.

        Returns:
            An array containing the rounded values to the specified ``decimals`` with
            same shape and dtype as ``a``.

        Note:
            ``xp.round`` rounds to the nearest even integer for the values exactly halfway
            between rounded decimal values.

        See also:
            - :func:`floor`: Rounds the input to the nearest integer downwards.
            - :func:`ceil`: Rounds the input to the nearest integer upwards.
            - :func:`fix` and :func:numpy.trunc`: Rounds the input to the
                nearest integer towards zero.

        Examples:
            >>> x = xp.array([1.532, 3.267, 6.149])
            >>> xp.round(x)
            Array([2., 3., 6.], dtype=float32)
            >>> xp.round(x, decimals=2)
            Array([1.53, 3.27, 6.15], dtype=float32)

            For values exactly halfway between rounded values:

            >>> x1 = xp.array([10.5, 21.5, 12.5, 31.5])
            >>> xp.round(x1)
            Array([10., 22., 12., 32.], dtype=float32)
        """
        ...

    def sin(self, x: ArrayLike, /) -> RealArray:
        """Compute a trigonometric sine of each element of input.

        Array API implementation of :obj:`numpy.sin`.

        Args:
            x: array or scalar. Angle in radians.

        Returns:
            An array containing the sine of each element in ``x``, promotes to inexact
            dtype.

        See also:
            - :func:`cos`: Computes a trigonometric cosine of each element of
                input.
            - :func:`tan`: Computes a trigonometric tangent of each element of
                input.
            - :func:`arcsin` and :func:`asin`: Computes the inverse of
                trigonometric sine of each element of input.

        Examples:
            >>> pi = xp.pi
            >>> x = xp.array([pi/4, pi/2, 3*pi/4, pi])
            >>> with xp.printoptions(precision=3, suppress=True):
            ...   print(xp.sin(x))
            [ 0.707  1.     0.707 -0.   ]
        """
        ...

    def size(self, a: ArrayLike, axis: int | None=None) -> int:
        """Return the number of elements along a given axis.

        Args:
            a: Input data.
            axis : Axis along which the elements are counted. By default, give
                the total number of elements.

        Returns:
            Number of elements along the specified axis.

        Examples:
            >>> a = xp.array([[1, 2, 3],
                              [4, 5, 6]])
            >>> xp.size(a)
            6
            >>> xp.size(a, 1)
            3
            >>> xp.size(a, 0)
            2

        """
        ...

    def sort(self, a: ArrayLike, axis: int | None = -1, *, kind: None = None,
             stable: bool = True, descending: bool = False) -> Array:
        """Return a sorted copy of an array.

        Array API implementation of :func:`numpy.sort`.

        Args:
            a: array to sort
            axis: integer axis along which to sort. Defaults to ``-1``, i.e. the last
                axis. If ``None``, then ``a`` is flattened before being sorted.
            stable: boolean specifying whether a stable sort should be used. Default=True.
            descending: boolean specifying whether to sort in descending order. Default=False.
            kind: deprecated; instead specify sort algorithm using stable=True or stable=False.

        Returns:
            Sorted array of shape ``a.shape`` (if ``axis`` is an integer) or of shape
            ``(a.size,)`` (if ``axis`` is None).

        Examples:
            Simple 1-dimensional sort

            >>> x = xp.array([1, 3, 5, 4, 2, 1])
            >>> xp.sort(x)
            Array([1, 1, 2, 3, 4, 5], dtype=int32)

            Sort along the last axis of an array:

            >>> x = xp.array([[2, 1, 3],
            ...               [4, 3, 6]])
            >>> xp.sort(x, axis=1)
            Array([[1, 2, 3],
                   [3, 4, 6]], dtype=int32)

            See also:
            - :func:`argsort`: return indices of sorted values.
            - :func:`lexsort`: lexicographical sort of multiple arrays.
        """
        ...

    def sqrt(self, x: ArrayLike, /) -> Array:
        """Calculates element-wise non-negative square root of the input array.

        Array API implementation of :obj:`numpy.sqrt`.

        Args:
            x: input array or scalar.

        Returns:
            An array containing the non-negative square root of the elements of ``x``.

        Note:
            - For real-valued negative inputs, ``xp.sqrt`` produces a ``nan`` output.
            - For complex-valued negative inputs, ``xp.sqrt`` produces a ``complex`` output.

        See also:
            - :func:`square`: Calculates the element-wise square of the input.
            - :func:`power`: Calculates the element-wise base ``x1`` exponential
                of ``x2``.

        Examples:
            >>> x = xp.array([-8-6j, 1j, 4])
            >>> with xp.printoptions(precision=3, suppress=True):
            ...   xp.sqrt(x)
            Array([1.   -3.j   , 0.707+0.707j, 2.   +0.j   ], dtype=complex64)
            >>> xp.sqrt(-1)
            Array(nan, dtype=float32, weak_type=True)
        """
        ...

    def squeeze(self, a: ArrayLike, axis: int | Sequence[int] | None = None) -> Array:
        """Remove one or more length-1 axes from array

        Array API implementation of :func:`numpy.sqeeze`.

        Args:
            a: input array
            axis: integer or sequence of integers specifying axes to remove. If any specified
                axis does not have a length of 1, an error is raised. If not specified, squeeze
                all length-1 axes in ``a``.

        Returns:
            copy of ``a`` with length-1 axes removed.

        See Also:
            - :func:`expand_dims`: the inverse of ``squeeze``: add dimensions of length 1.
            - :func:`ravel`: flatten an array into a 1D shape.
            - :func:`reshape`: general array reshape.

        Examples:
            >>> x = xp.array([[[0]], [[1]], [[2]]])
            >>> x.shape
            (3, 1, 1)

            Squeeze all length-1 dimensions:

            >>> xp.squeeze(x)
            Array([0, 1, 2], dtype=int32)
            >>> _.shape
            (3,)

            Equivalent while specifying the axes explicitly:

            >>> xp.squeeze(x, axis=(1, 2))
            Array([0, 1, 2], dtype=int32)

            Attempting to squeeze a non-unit axis results in an error:

            >>> xp.squeeze(x, axis=0)  # doctest: +IGNORE_EXCEPTION_DETAIL
            Traceback (most recent call last):
                ...
            ValueError: cannot select an axis to squeeze out which has size not equal to one,
                got shape=(3, 1, 1) and dimensions=(0,)
        """
        ...

    def stack(self, array: Array | Sequence[ArrayLike], axis: int = 0,
              dtype: DTypeLike | None = None) -> Array:
        """Join arrays along a new axis.

        Array API implementation of :func:`numpy.stack`.

        Args:
            arrays: a sequence of arrays to stack; each must have the same shape. If a
                single array is given it will be treated equivalently to
                `arrays = unstack(arrays)`, but the implementation will avoid explicit
                unstacking.
            axis: specify the axis along which to stack.
            dtype: optional dtype of the resulting array. If not specified, the dtype
                will be determined via type promotion rules described in :ref:`type-promotion`.

        Returns:
            the stacked result.

        See also:
            - :func:`unstack`: inverse of ``stack``.
            - :func:`concatenate`: concatenation along existing axes.
            - :func:`vstack`: stack vertically, i.e. along axis 0.
            - :func:`hstack`: stack horizontally, i.e. along axis 1.
            - :func:`dstack`: stack depth-wise, i.e. along axis 2.
            - :func:`column_stack`: stack columns.

        Examples:
            >>> x = xp.array([1, 2, 3])
            >>> y = xp.array([4, 5, 6])
            >>> xp.stack([x, y])
            Array([[1, 2, 3],
                   [4, 5, 6]], dtype=int32)
            >>> xp.stack([x, y], axis=1)
            Array([[1, 4],
                   [2, 5],
                   [3, 6]], dtype=int32)

            :func:`~unstack` performs the inverse operation:

            >>> arr = xp.stack([x, y], axis=1)
            >>> x, y = xp.unstack(arr, axis=1)
            >>> x
            Array([1, 2, 3], dtype=int32)
            >>> y
            Array([4, 5, 6], dtype=int32)
        """
        ...

    def std(self, a: ArrayLike, axis: Axis = None, dtype: DTypeLike | None = None,
            ddof: int = 0, keepdims: bool = False, *, where: ArrayLike | None = None,
            correction: int | float | None = None) -> Array:
        r"""Compute the standard deviation along a given axis.

        Array API implementation of :func:`numpy.std`.

        Args:
            a: input array.
            axis: optional, int or sequence of ints, default=None. Axis along which the
                standard deviation is computed. If None, standard deviaiton is computed
                along all the axes.
            dtype: The type of the output array. Default=None.
            ddof: int, default=0. Degrees of freedom. The divisor in the standard deviation
                computation is ``N-ddof``, ``N`` is number of elements along given axis.
            keepdims: bool, default=False. If true, reduced axes are left in the result
                with size 1.
            where: optional, boolean array, default=None. The elements to be used in the
                standard deviation. Array should be broadcast compatible to the input.
            correction: int or float, default=None. Alternative name for ``ddof``.
                Both ddof and correction can't be provided simultaneously.

        Returns:
            An array of the standard deviation along the given axis.

        See also:
            - :func:`var`: Compute the variance of array elements over given
                axis.
            - :func:`mean`: Compute the mean of array elements over a given axis.
            - :func:`nanvar`: Compute the variance along a given axis, ignoring
                NaNs values.
            - :func:`nanstd`: Computed the standard deviation of a given axis,
                ignoring NaN values.

        Examples:
            By default, ``xp.std`` computes the standard deviation along all axes.

            >>> x = xp.array([[1, 3, 4, 2],
            ...               [4, 2, 5, 3],
            ...               [5, 4, 2, 3]])
            >>> with xp.printoptions(precision=2, suppress=True):
            ...   xp.std(x)
            Array(1.21, dtype=float32)

            If ``axis=0``, computes along axis 0.

            >>> with xp.printoptions(precision=2, suppress=True):
            ...   print(xp.std(x, axis=0))
            [1.7  0.82 1.25 0.47]

            To preserve the dimensions of input, you can set ``keepdims=True``.

            >>> with xp.printoptions(precision=2, suppress=True):
            ...   print(xp.std(x, axis=0, keepdims=True))
            [[1.7  0.82 1.25 0.47]]

            If ``ddof=1``:

            >>> with xp.printoptions(precision=2, suppress=True):
            ...   print(xp.std(x, axis=0, keepdims=True, ddof=1))
            [[2.08 1.   1.53 0.58]]

            To include specific elements of the array to compute standard deviation, you
            can use ``where``.

            >>> where = xp.array([[1, 0, 1, 0],
            ...                   [0, 1, 0, 1],
            ...                   [1, 1, 1, 0]], dtype=bool)
            >>> xp.std(x, axis=0, keepdims=True, where=where)
            Array([[2., 1., 1., 0.]], dtype=float32)
        """
        ...

    def sum(self, a: ArrayLike, axis: Axis = None, dtype: DTypeLike | None = None,
            keepdims: bool = False, initial: ArrayLike | None = None,
            where: ArrayLike | None = None) -> Array:
        r"""Sum of the elements of the array over a given axis.

        Array API implementation of :func:`numpy.sum`.

        Args:
            a: Input array.
            axis: int or array, default=None. Axis along which the sum to be computed.
                If None, the sum is computed along all the axes.
            dtype: The type of the output array. Default=None.
            keepdims: bool, default=False. If true, reduced axes are left in the result
                with size 1.
            initial: int or array, Default=None. Initial value for the sum.
            where: int or array, default=None. The elements to be used in the sum. Array
                should be broadcast compatible to the input.
            promote_integers : bool, default=True. If True, then integer inputs will be
                promoted to the widest available integer dtype, following numpy's behavior.
                If False, the result will have the same dtype as the input.
                ``promote_integers`` is ignored if ``dtype`` is specified.

        Returns:
            An array of the sum along the given axis.

        See also:
            - :func:`prod`: Compute the product of array elements over a given
                axis.
            - :func:`max`: Compute the maximum of array elements over given axis.
            - :func:`min`: Compute the minimum of array elements over given axis.

        Examples:
            By default, the sum is computed along all the axes.

            >>> x = xp.array([[1, 3, 4, 2],
            ...               [5, 2, 6, 3],
            ...               [8, 1, 3, 9]])
            >>> xp.sum(x)
            Array(47, dtype=int32)

            If ``axis=1``, the sum is computed along axis 1.

            >>> xp.sum(x, axis=1)
            Array([10, 16, 21], dtype=int32)

            If ``keepdims=True``, ``ndim`` of the output is equal to that of the input.

            >>> xp.sum(x, axis=1, keepdims=True)
            Array([[10],
                   [16],
                   [21]], dtype=int32)

            To include only specific elements in the sum, you can use ``where``.

            >>> where=xp.array([[0, 0, 1, 0],
            ...                 [0, 0, 1, 1],
            ...                 [1, 1, 1, 0]], dtype=bool)
            >>> xp.sum(x, axis=1, keepdims=True, where=where)
            Array([[ 4],
                   [ 9],
                   [12]], dtype=int32)
            >>> where=xp.array([[False],
            ...                 [False],
            ...                 [False]])
            >>> xp.sum(x, axis=0, keepdims=True, where=where)
            Array([[0, 0, 0, 0]], dtype=int32)
        """
        ...

    def swapaxes(self, a: ArrayLike, axis1: int, axis2: int) -> Array:
        """Swap two axes of an array.

        Array API implementation of :func:`numpy.swapaxes`.

        Args:
            a: input array
            axis1: index of first axis
            axis2: index of second axis

        Returns:
            Copy of ``a`` with specified axes swapped.

        See Also:
            - :func:`moveaxis`: move a single axis of an array.
            - :func:`rollaxis`: older API for ``moveaxis``.
            - :meth:`swapaxes`: same functionality via an array method.

        Examples:
            >>> a = xp.ones((2, 3, 4, 5))
            >>> xp.swapaxes(a, 1, 3).shape
            (2, 5, 4, 3)

            Equivalent output via the ``swapaxes`` array method:

            >>> a.swapaxes(1, 3).shape
            (2, 5, 4, 3)

            Equivalent output via :func:`~transpose`:

            >>> a.transpose(0, 3, 2, 1).shape
            (2, 5, 4, 3)
        """
        ...

    def take_along_axis(self, arr: ArrayLike, indices: ArrayLike, axis: int | None,
                        mode: str  | None = None, fill_value: Scalar | None = None
                        ) -> Array:
        """Take elements from an array.

        Array API implementation of :func:`numpy.take_along_axis`. Array API's behavior differs
        from NumPy in the case of out-of-bound indices; see the ``mode`` parameter below.

        Args:
            a: array from which to take values.
            indices: array of integer indices. If ``axis`` is ``None``, must be one-dimensional.
                If ``axis`` is not None, must have ``a.ndim == indices.ndim``, and ``a`` must be
                broadcast-compatible with ``indices`` along dimensions other than ``axis``.
            axis: the axis along which to take values. If not specified, the array will
                be flattened before indexing is applied.
            mode: Out-of-bounds indexing mode, either ``"fill"`` or ``"clip"``. The default
                ``mode="fill"`` returns invalid values (e.g. NaN) for out-of bounds indices.
                For more discussion of ``mode`` options, see :attr:`ndarray.at`.

        Returns:
            Array of values extracted from ``a``.

        See also:
            - :attr:`ndarray.at`: take values via indexing syntax.
            - :func:`take`: take the same indices along every axis slice.

        Examples:
            >>> x = xp.array([[1., 2., 3.],
            ...               [4., 5., 6.]])
            >>> indices = xp.array([[0, 2],
            ...                     [1, 0]])
            >>> xp.take_along_axis(x, indices, axis=1)
            Array([[1., 3.],
                   [5., 4.]], dtype=float32)
            >>> x[xp.arange(2)[:, None], indices]  # equivalent via indexing syntax
            Array([[1., 3.],
                   [5., 4.]], dtype=float32)

            Out-of-bound indices fill with invalid values. For float inputs, this is `NaN`:

            >>> indices = xp.array([[1, 0, 2]])
            >>> xp.take_along_axis(x, indices, axis=0)
            Array([[ 4.,  2., nan]], dtype=float32)
            >>> x.at[indices, xp.arange(3)].get(
            ...     mode='fill', fill_value=xp.nan)  # equivalent via indexing syntax
            Array([[ 4.,  2., nan]], dtype=float32)

            ``take_along_axis`` is helpful for extracting values from multi-dimensional
            argsorts and arg reductions. For, here we compute :func:`~argsort`
            indices along an axis, and use ``take_along_axis`` to construct the sorted
            array:

            >>> x = xp.array([[5, 3, 4],
            ...               [2, 7, 6]])
            >>> indices = xp.argsort(x, axis=1)
            >>> indices
            Array([[1, 2, 0],
                   [0, 2, 1]], dtype=int32)
            >>> xp.take_along_axis(x, indices, axis=1)
            Array([[3, 4, 5],
                   [2, 6, 7]], dtype=int32)

            Similarly, we can use :func:`~argmin` with ``keepdims=True`` and
            use ``take_along_axis`` to extract the minimum value:

            >>> idx = xp.argmin(x, axis=1, keepdims=True)
            >>> idx
            Array([[1],
                   [0]], dtype=int32)
            >>> xp.take_along_axis(x, idx, axis=1)
            Array([[3],
                   [2]], dtype=int32)
        """
        ...

    def tan(self, x: ArrayLike, /) -> Array:
        """Compute a trigonometric tangent of each element of input.

        Array API implementation of :obj:`numpy.tan`.

        Args:
            x: scalar or array. Angle in radians.

        Returns:
            An array containing the tangent of each element in ``x``, promotes to inexact
            dtype.

        See also:
            - :func:`sin`: Computes a trigonometric sine of each element of input.
            - :func:`cos`: Computes a trigonometric cosine of each element of
                input.
            - :func:`arctan` and :func:`atan`: Computes the inverse of
                trigonometric tangent of each element of input.

        Examples:
        >>> pi = xp.pi
        >>> x = xp.array([0, pi/6, pi/4, 3*pi/4, 5*pi/6])
        >>> with xp.printoptions(precision=3, suppress=True):
        ...   print(xp.tan(x))
        [ 0.     0.577  1.    -1.    -0.577]
        """
        ...

    def tanh(self, x: ArrayLike, /) -> Array:
        r"""Calculate element-wise hyperbolic tangent of input.

        Array API implementation of :obj:`numpy.tanh`.

        The hyperbolic tangent is defined by:

        .. math::

            tanh(x) = \frac{sinh(x)}{cosh(x)} = \frac{e^x - e^{-x}}{e^x + e^{-x}}

        Args:
            x: input array or scalar.

        Returns:
            An array containing the hyperbolic tangent of each element of ``x``, promoting
            to inexact dtype.

        Note:
        ``xp.tanh`` is equivalent to computing ``-1j * xp.tan(1j * x)``.

        See also:
            - :func:`sinh`: Computes the element-wise hyperbolic sine of the input.
            - :func:`cosh`: Computes the element-wise hyperbolic cosine of the input.
            - :func:`arctanh`:  Computes the element-wise inverse of hyperbolic
                tangent of the input.

        Examples:
        >>> x = xp.array([[-1, 0, 1],
        ...               [3, -2, 5]])
        >>> with xp.printoptions(precision=3, suppress=True):
        ...   xp.tanh(x)
        Array([[-0.762,  0.   ,  0.762],
               [ 0.995, -0.964,  1.   ]], dtype=float32)
        >>> with xp.printoptions(precision=3, suppress=True):
        ...   -1j * xp.tan(1j * x)
        Array([[-0.762+0.j,  0.   -0.j,  0.762-0.j],
               [ 0.995-0.j, -0.964+0.j,  1.   -0.j]], dtype=complex64, weak_type=True)

        For complex-valued input:

        >>> with xp.printoptions(precision=3, suppress=True):
        ...   xp.tanh(2-5j)
        Array(1.031+0.021j, dtype=complex64, weak_type=True)
        >>> with xp.printoptions(precision=3, suppress=True):
        ...   -1j * xp.tan(1j * (2-5j))
        Array(1.031+0.021j, dtype=complex64, weak_type=True)
        """
        ...

    def tile(self, a: ArrayLike, reps: DimSize | Sequence[DimSize]) -> Array:
        """Construct an array by repeating ``A`` along specified dimensions.

        Array API implementation of :func:`numpy.tile`.

        If ``A`` is an array of shape ``(d1, d2, ..., dn)`` and ``reps`` is a sequence of integers,
        the resulting array will have a shape of ``(reps[0] * d1, reps[1] * d2, ..., reps[n] * dn)``,
        with ``A`` tiled along each dimension.

        Args:
            A: input array to be repeated. Can be of any shape or dimension.
            reps: specifies the number of repetitions along each axis.

        Returns:
            a new array where the input array has been repeated according to ``reps``.

        See also:
            - :func:`repeat`: Construct an array from repeated elements.
            - :func:`broadcast_to`: Broadcast an array to a specified shape.

        Examples:
            >>> arr = xp.array([1, 2])
            >>> xp.tile(arr, 2)
            Array([1, 2, 1, 2], dtype=int32)
            >>> arr = xp.array([[1, 2],
            ...                 [3, 4,]])
            >>> xp.tile(arr, (2, 1))
            Array([[1, 2],
                   [3, 4],
                   [1, 2],
                   [3, 4]], dtype=int32)
        """
        ...

    def trace(self, a: ArrayLike, offset: int | ArrayLike = 0, axis1: int = 0, axis2: int = 1,
              dtype: DTypeLike | None = None) -> Array:
        """Calculate sum of the diagonal of input along the given axes.

        Array API implementation of :func:`numpy.trace`.

        Args:
            a: input array. Must have ``a.ndim >= 2``.
            offset: optional, int, default=0. Diagonal offset from the main diagonal.
                Can be positive or negative.
            axis1: optional, default=0. The first axis along which to take the sum of
                diagonal. Must be a static integer value.
            axis2: optional, default=1. The second axis along which to take the sum of
                diagonal. Must be a static integer value.
            dtype: optional. The dtype of the output array. Should be provided as static
                argument in JIT compilation.

        Returns:
            An array of dimension x.ndim-2 containing the sum of the diagonal elements
            along axes (axis1, axis2)

        See also:
            - :func:`diag`: Returns the specified diagonal or constructs a diagonal
                array
            - :func:`diagonal`: Returns the specified diagonal of an array.
            - :func:`diagflat`: Returns a 2-D array with the flattened input array
                laid out on the diagonal.

        Examples:
            >>> x = xp.arange(1, 9).reshape(2, 2, 2)
            >>> x
            Array([[[1, 2],
                    [3, 4]],
            <BLANKLINE>
                    [[5, 6],
                    [7, 8]]], dtype=int32)
            >>> xp.trace(x)
            Array([ 8, 10], dtype=int32)
            >>> xp.trace(x, offset=1)
            Array([3, 4], dtype=int32)
            >>> xp.trace(x, axis1=1, axis2=2)
            Array([ 5, 13], dtype=int32)
            >>> xp.trace(x, offset=1, axis1=1, axis2=2)
            Array([2, 6], dtype=int32)
        """
        ...

    @overload
    def unique(self, ar: ArrayLike, *, return_index: Literal[False] = False,
               return_inverse: Literal[False] = False, return_counts: Literal[False] = False,
               axis: int | None = None, equal_nan: bool = True) -> Array: ...

    @overload
    def unique(self, ar: ArrayLike, *, return_index: Literal[True],
               return_inverse: Literal[False] = False, return_counts: Literal[False] = False,
               axis: int | None = None, equal_nan: bool = True) -> Tuple[Array, Array]: ...

    @overload
    def unique(self, ar: ArrayLike, *, return_index: Literal[False] = False,
               return_inverse: Literal[True], return_counts: Literal[False] = False,
               axis: int | None = None, equal_nan: bool = True) -> Tuple[Array, Array]: ...

    @overload
    def unique(self, ar: ArrayLike, *, return_index: Literal[True], return_inverse: Literal[True],
               return_counts: Literal[False] = False, axis: int | None = None,
               equal_nan: bool = True) -> Tuple[Array, Array, Array]: ...

    @overload
    def unique(self, ar: ArrayLike, *, return_index: Literal[False] = False,
               return_inverse: Literal[False] = False, return_counts: Literal[True],
               axis: int | None = None, equal_nan: bool = True) -> Tuple[Array, Array]: ...

    @overload
    def unique(self, ar: ArrayLike, *, return_index: Literal[True],
               return_inverse: Literal[False] = False, return_counts: Literal[True],
               axis: int | None = None, equal_nan: bool = True) -> Tuple[Array, Array, Array]: ...

    @overload
    def unique(self, ar: ArrayLike, *, return_index: Literal[False] = False,
               return_inverse: Literal[True], return_counts: Literal[True], axis: int | None = None,
               equal_nan: bool = True) -> Tuple[Array, Array, Array]: ...

    @overload
    def unique(self, ar: ArrayLike, *, return_index: Literal[True], return_inverse: Literal[True],
               return_counts: Literal[True], axis: int | None = None, equal_nan: bool = True
               ) -> Tuple[Array, Array, Array, Array]: ...

    def unique(self, ar: ArrayLike, *, return_index: bool = False, return_inverse: bool = False,
               return_counts: bool = False, axis: int | None = None, equal_nan: bool = True
               ) -> Array | Tuple[Array, ...]:
        """Return the unique values from an array.

        Array API implementation of :func:`numpy.unique`.

        Args:
            ar: N-dimensional array from which unique values will be extracted.
            return_index: if True, also return the indices in ``ar`` where each value occurs
            return_inverse: if True, also return the indices that can be used to reconstruct
                ``ar`` from the unique values.
            return_counts: if True, also return the number of occurrences of each unique value.
            axis: if specified, compute unique values along the specified axis. If None (default),
                then flatten ``ar`` before computing the unique values.
            equal_nan: if True, consider NaN values equivalent when determining uniqueness.
            fill_value: when ``size`` is specified and there are fewer than the indicated number of
                elements, fill the remaining entries ``fill_value``. Defaults to the minimum unique
                value.

        Returns:
            An array or tuple of arrays, depending on the values of ``return_index``,
            ``return_inverse``, and ``return_counts``. Returned values are

            - ``unique_values``:
                if ``axis`` is None, a 1D array of length ``n_unique``, If ``axis`` is
                specified, shape is ``(*ar.shape[:axis], n_unique, *ar.shape[axis + 1:])``.
            - ``unique_index``:
                *(returned only if return_index is True)* An array of shape ``(n_unique,)``.
                Contains the indices of the first occurrence of each unique value in ``ar``. For 1D
                inputs, ``ar[unique_index]`` is equivalent to ``unique_values``.
            - ``unique_inverse``:
                *(returned only if return_inverse is True)* An array of shape ``(ar.size,)`` if
                ``axis`` is None, or of shape ``(ar.shape[axis],)`` if ``axis`` is specified.
                Contains the indices within ``unique_values`` of each value in ``ar``. For 1D
                inputs, ``unique_values[unique_inverse]`` is equivalent to ``ar``.
            - ``unique_counts``:
                *(returned only if return_counts is True)* An array of shape ``(n_unique,)``.
                Contains the number of occurrences of each unique value in ``ar``.

        See also:
            - :func:`unique_counts`: shortcut to ``unique(arr, return_counts=True)``.
            - :func:`unique_inverse`: shortcut to ``unique(arr, return_inverse=True)``.
            - :func:`unique_all`: shortcut to ``unique`` with all return values.
            - :func:`unique_values`: like ``unique``, but no optional return values.

        Examples:
            >>> x = xp.array([3, 4, 1, 3, 1])
            >>> xp.unique(x)
            Array([1, 3, 4], dtype=int32)

            **Multi-dimensional unique values**

            If you pass a multi-dimensional array to ``unique``, it will be flattened by default:

            >>> M = xp.array([[1, 2],
            ...               [2, 3],
            ...               [1, 2]])
            >>> xp.unique(M)
            Array([1, 2, 3], dtype=int32)

            If you pass an ``axis`` keyword, you can find unique *slices* of the array along
            that axis:

            >>> xp.unique(M, axis=0)
            Array([[1, 2],
                   [2, 3]], dtype=int32)

            **Returning indices**

            If you set ``return_index=True``, then ``unique`` returns the indices of the
            first occurrence of each unique value:

            >>> x = xp.array([3, 4, 1, 3, 1])
            >>> values, indices = xp.unique(x, return_index=True)
            >>> print(values)
            [1 3 4]
            >>> print(indices)
            [2 0 1]
            >>> xp.all(values == x[indices])
            Array(True, dtype=bool)

            In multiple dimensions, the unique values can be extracted with :func:`take`
            evaluated along the specified axis:

            >>> values, indices = xp.unique(M, axis=0, return_index=True)
            >>> xp.all(values == xp.take(M, indices, axis=0))
            Array(True, dtype=bool)

            **Returning inverse**

            If you set ``return_inverse=True``, then ``unique`` returns the indices within the
            unique values for every entry in the input array:

            >>> x = xp.array([3, 4, 1, 3, 1])
            >>> values, inverse = xp.unique(x, return_inverse=True)
            >>> print(values)
            [1 3 4]
            >>> print(inverse)
            [1 2 0 1 0]
            >>> xp.all(values[inverse] == x)
            Array(True, dtype=bool)

            In multiple dimensions, the input can be reconstructed using
            :func:`take`:

            >>> values, inverse = xp.unique(M, axis=0, return_inverse=True)
            >>> xp.all(xp.take(values, inverse, axis=0) == M)
            Array(True, dtype=bool)

            **Returning counts**

            If you set ``return_counts=True``, then ``unique`` returns the number of occurrences
            within the input for every unique value:

            >>> x = xp.array([3, 4, 1, 3, 1])
            >>> values, counts = xp.unique(x, return_counts=True)
            >>> print(values)
            [1 3 4]
            >>> print(counts)
            [2 2 1]

            For multi-dimensional arrays, this also returns a 1D array of counts
            indicating number of occurrences along the specified axis:

            >>> values, counts = xp.unique(M, axis=0, return_counts=True)
            >>> print(values)
            [[1 2]
             [2 3]]
            >>> print(counts)
            [2 1]
        """
        ...

    def unique_counts(self, x: ArrayLike, /) -> UniqueCountsResult:
        """Return unique values from x, along with counts.

        Array API implementation of :func:`numpy.unique_counts`; this is equivalent to calling
        :func:`unique` with `return_counts` and `equal_nan` set to True.

        Args:
            x: N-dimensional array from which unique values will be extracted.

        Returns:
            A tuple ``(values, counts)``, with the following properties:

            - ``values``:
                an array of shape ``(n_unique,)`` containing the unique values from ``x``.
            - ``counts``:
                An array of shape ``(n_unique,)``. Contains the number of occurrences of each unique
                value in ``x``.

        See also:
            - :func:`unique`: general function for computing unique values.
            - :func:`unique_values`: compute only ``values``.
            - :func:`unique_inverse`: compute only ``values`` and ``inverse``.
            - :func:`unique_all`: compute ``values``, ``indices``, ``inverse_indices``,
                and ``counts``.

        Examples:
            Here we compute the unique values in a 1D array:

            >>> x = xp.array([3, 4, 1, 3, 1])
            >>> result = xp.unique_counts(x)

            The result is a :class:`~typing.NamedTuple` with two named attributes.
            The ``values`` attribute contains the unique values from the array:

            >>> result.values
            Array([1, 3, 4], dtype=int32)

            The ``counts`` attribute contains the counts of each unique value in the input:

            >>> result.counts
            Array([2, 2, 1], dtype=int32)
        """
        ...

    def unique_inverse(self, x: ArrayLike, /) -> UniqueInverseResult:
        """Return unique values from x, along with indices, inverse indices, and counts.

        Array API implementation of :func:`numpy.unique_inverse`; this is equivalent to calling
        :func:`unique` with `return_inverse` and `equal_nan` set to True.

        Args:
        x: N-dimensional array from which unique values will be extracted.

        Returns:
        A tuple ``(values, indices, inverse_indices, counts)``, with the following properties:

        - ``values``:
            an array of shape ``(n_unique,)`` containing the unique values from ``x``.
        - ``inverse_indices``:
            An array of shape ``x.shape``. Contains the indices within ``values`` of each value
            in ``x``. For 1D inputs, ``values[inverse_indices]`` is equivalent to ``x``.

        See also:
        - :func:`unique`: general function for computing unique values.
        - :func:`unique_values`: compute only ``values``.
        - :func:`unique_counts`: compute only ``values`` and ``counts``.
        - :func:`unique_all`: compute ``values``, ``indices``, ``inverse_indices``, and ``counts``.

        Examples:
        Here we compute the unique values in a 1D array:

        >>> x = xp.array([3, 4, 1, 3, 1])
        >>> result = xp.unique_inverse(x)

        The result is a :class:`~typing.NamedTuple` with two named attributes.
        The ``values`` attribute contains the unique values from the array:

        >>> result.values
        Array([1, 3, 4], dtype=int32)

        The ``indices`` attribute contains the indices of the unique ``values`` within
        the input array:

        The ``inverse_indices`` attribute contains the indices of the input within ``values``:

        >>> result.inverse_indices
        Array([1, 2, 0, 1, 0], dtype=int32)
        >>> xp.all(x == result.values[result.inverse_indices])
        Array(True, dtype=bool)
        """
        ...

    def unravel_index(self, indices: ArrayLike, shape: Shape) -> tuple[Array, ...]:
        """Convert flat indices into multi-dimensional indices.

        Array API implementation of :func:`numpy.unravel_index`. The Array API version differs in
        its treatment of out-of-bound indices: unlike NumPy, negative indices are
        supported, and out-of-bound indices are clipped to the nearest valid value.

        Args:
            indices: integer array of flat indices
            shape: shape of multidimensional array to index into

        Returns:
            Tuple of unraveled indices

        See also:
            :func:`ravel_multi_index`: Inverse of this function.

        Examples:
            Start with a 1D array values and indices:

            >>> x = xp.array([2., 3., 4., 5., 6., 7.])
            >>> indices = xp.array([1, 3, 5])
            >>> print(x[indices])
            [3. 5. 7.]

            Now if ``x`` is reshaped, ``unravel_indices`` can be used to convert
            the flat indices into a tuple of indices that access the same entries:

            >>> shape = (2, 3)
            >>> x_2D = x.reshape(shape)
            >>> indices_2D = xp.unravel_index(indices, shape)
            >>> indices_2D
            (Array([0, 1, 1], dtype=int32), Array([1, 0, 2], dtype=int32))
            >>> print(x_2D[indices_2D])
            [3. 5. 7.]

            The inverse function, ``ravel_multi_index``, can be used to obtain the
            original indices:

            >>> xp.ravel_multi_index(indices_2D, shape)
            Array([1, 3, 5], dtype=int32)
        """
        ...

    def where(self, condition: BoolArray | ArrayLike, x: ArrayLike | None = None,
              y: ArrayLike | None = None) -> Array:
        """Select elements from two arrays based on a condition.

        Array API implementation of :func:`numpy.where`.

        .. note::
            when only ``condition`` is provided, ``xp.where(condition)`` is equivalent
            to ``xp.nonzero(condition)``. For that case, refer to the documentation of
            :func:`nonzero`. The docstring below focuses on the case where
            ``x`` and ``y`` are specified.

        Args:
            condition: boolean array. Must be broadcast-compatible with ``x`` and ``y`` when
                they are specified.
            x: arraylike. Should be broadcast-compatible with ``condition`` and ``y``, and
                typecast-compatible with ``y``.
            y: arraylike. Should be broadcast-compatible with ``condition`` and ``x``, and
                typecast-compatible with ``x``.
            size: integer, only referenced when ``x`` and ``y`` are ``None``. For details,
                see :func:`nonzero`.
            fill_value: only referenced when ``x`` and ``y`` are ``None``. For details,
                see :func:`nonzero`.

        Returns:
            An array of dtype ``xp.result_type(x, y)`` with values drawn from ``x`` where
            ``condition`` is True, and from ``y`` where condition is ``False``. If ``x`` and
            ``y`` are ``None``, the function behaves differently; see :func:`nonzero`
            for a description of the return type.

        See Also:
            - :func:`nonzero`
            - :func:`argwhere`

        Examples:
            When ``x`` and ``y`` are not provided, ``where`` behaves equivalently to
            :func:`nonzero`:

            >>> x = xp.arange(10)
            >>> xp.where(x > 4)
            (Array([5, 6, 7, 8, 9], dtype=int32),)
            >>> xp.nonzero(x > 4)
            (Array([5, 6, 7, 8, 9], dtype=int32),)

            When ``x`` and ``y`` are provided, ``where`` selects between them based on
            the specified condition:

            >>> xp.where(x > 4, x, 0)
            Array([0, 0, 0, 0, 0, 5, 6, 7, 8, 9], dtype=int32)
        """
        ...

    def zeros(self, shape: Any, dtype: DTypeLike | None = None, *, device: Device | None = None
              ) -> Array:
        """Create an array full of zeros.

        Array API implementation of :func:`numpy.zeros`.

        Args:
            shape: int or sequence of ints specifying the shape of the created array.
            dtype: optional dtype for the created array; defaults to floating point.
            device: (optional) :class:`~xp.Device` to which the created array will be
                committed.

        Returns:
            Array of the specified shape and dtype, on the specified device if specified.

        See also:
            - :func:`zeros_like`
            - :func:`empty`
            - :func:`ones`
            - :func:`full`

        Examples:
            >>> xp.zeros(4)
            Array([0., 0., 0., 0.], dtype=float32)
            >>> xp.zeros((2, 3), dtype=bool)
            Array([[False, False, False],
                   [False, False, False]], dtype=bool)
        """
        ...

JaxNumPy = cast(ArrayNamespace, jnp)
NumPy = cast(ArrayNamespace, np)
