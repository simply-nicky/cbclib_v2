from typing import Callable, Dict, Literal, Sequence, Tuple, Union, cast
import numpy as np
import jax.numpy as jnp
from jax import core, dtypes
from jax.interpreters import ad, mlir
from ..src import (QuadStackDouble, QuadStackFloat, build_quad_stack,
                   OctStackDouble, OctStackFloat, build_oct_stack)
from ..annotations import RealArray, IntArray

Kind = Literal['2d', '3d']
TreeStack = Union[QuadStackDouble, QuadStackFloat, OctStackDouble, OctStackFloat]
Query = Callable[[RealArray, IntArray], IntArray]
BuildAndQuery = Callable[[RealArray, IntArray, RealArray, IntArray], IntArray]
BuildStack = Callable[[RealArray, IntArray, Sequence[float]], TreeStack]

def knn_query(stack: TreeStack, k: int, num_threads: int=1) -> Query:
    knn_query_p = core.Primitive("knn_query")

    # The abstract evaluation rule tells us the expected shape & dtype of the output.
    def _knn_query_abstract_eval(query: core.ShapedArray, idxs: core.ShapedArray):
        assert query.shape[-1] == 4 or query.shape[-1] == 6
        assert query.shape[:-1] == idxs.shape
        itype = dtypes.canonicalize_dtype(idxs.dtype)
        return core.ShapedArray(idxs.shape + (k,), itype)

    # The impl rule defines how the primitive is evaluated *outside* jit.
    def _knn_query_impl(query: RealArray, idxs: IntArray) -> IntArray:
        _, idxs = stack.find_k_nearest(np.asarray(query), np.asarray(idxs), k, num_threads)
        return idxs

    # The lowering rule defines how the primitive is evaluated *within* jit
    def _knn_query_lowering(ctx: mlir.LoweringRuleContext, query: mlir.Value, idxs: mlir.Value
                              ) -> Sequence[Union[mlir.Value, Tuple[mlir.Value, ...]]]:
        # Note: callback function must return a tuple of arrays with the expected shape & dtype
        def _knn_callback(query, idxs):
            _, idxs = stack.find_k_nearest(query, idxs, k, num_threads)
            return (idxs.astype(ctx.avals_out[0].dtype),)
        token = None
        avals_in = cast(Sequence[core.ShapedArray], ctx.avals_in)
        result, token, keepalive = mlir.emit_python_callback(
            ctx, _knn_callback, token, [query, idxs], avals_in, ctx.avals_out,
            has_side_effect=False
        )
        ctx.module_context.add_keepalive(keepalive)
        return result

    knn_query_p.def_abstract_eval(_knn_query_abstract_eval)
    knn_query_p.def_impl(_knn_query_impl)
    mlir.register_lowering(knn_query_p, cast(mlir.LoweringRule, _knn_query_lowering))

    ad.defjvp_zero(knn_query_p)

    def wrapper(query: RealArray, idxs: IntArray) -> IntArray:
        return cast(IntArray, knn_query_p.bind(query, idxs))

    return wrapper

def build_and_knn_query(box: Sequence[float], k: int, num_threads: int=1, kind: Kind='2d'
                        ) -> BuildAndQuery:
    knn_query_p = core.Primitive("knn_query")
    build_stack: Dict[Kind, BuildStack] = {'2d': build_quad_stack, '3d': build_oct_stack}
    element_size: Dict[Kind, int] = {'2d': 4, '3d': 6}

    # The abstract evaluation rule tells us the expected shape & dtype of the output.
    def _knn_query_abstract_eval(database: core.ShapedArray, didxs: core.ShapedArray,
                                 query: core.ShapedArray, idxs: core.ShapedArray
                                 ) -> core.ShapedArray:
        assert query.shape[-1] == element_size[kind] and database.shape[-1] == element_size[kind]
        assert query.shape[:-1] == idxs.shape and database.shape[:-1] == didxs.shape

        assert jnp.issubdtype(dtypes.canonicalize_dtype(database.dtype), jnp.floating)
        assert jnp.issubdtype(dtypes.canonicalize_dtype(query.dtype), jnp.floating)

        assert jnp.issubdtype(dtypes.canonicalize_dtype(didxs.dtype), jnp.integer)
        assert jnp.issubdtype(dtypes.canonicalize_dtype(idxs.dtype), jnp.integer)

        itype = dtypes.result_type(dtypes.canonicalize_dtype(idxs.dtype),
                                   dtypes.canonicalize_dtype(didxs.dtype),
                                   return_weak_type_flag=False)
        return core.ShapedArray(idxs.shape + (k,), itype)

    # The impl rule defines how the primitive is evaluated *outside* jit.
    def _knn_query_impl(database: RealArray, didxs: IntArray, query: RealArray,
                        idxs: IntArray) -> IntArray:
        stack = build_stack[kind](np.asarray(database), np.asarray(didxs), box)
        _, idxs = stack.find_k_nearest(np.asarray(query), np.asarray(idxs), k, num_threads)
        return idxs

    # The lowering rule defines how the primitive is evaluated *within* jit
    def _knn_query_lowering(ctx: mlir.LoweringRuleContext, database: mlir.Value,
                            didxs: mlir.Value, query: mlir.Value, idxs: mlir.Value
                            ) -> Sequence[Union[mlir.Value, Tuple[mlir.Value, ...]]]:
        # Note: callback function must return a tuple of arrays with the expected shape & dtype
        def _knn_callback(database, didxs, query, idxs):
            stack = build_stack[kind](database, didxs, box)
            _, idxs = stack.find_k_nearest(query, idxs, k, num_threads)
            return (idxs.astype(ctx.avals_out[0].dtype),)
        token = None
        avals_in = cast(Sequence[core.ShapedArray], ctx.avals_in)
        result, token, keepalive = mlir.emit_python_callback(
            ctx, _knn_callback, token, [database, didxs, query, idxs], avals_in, ctx.avals_out,
            has_side_effect=False
        )
        ctx.module_context.add_keepalive(keepalive)
        return result

    knn_query_p.def_abstract_eval(_knn_query_abstract_eval)
    knn_query_p.def_impl(_knn_query_impl)
    mlir.register_lowering(knn_query_p, cast(mlir.LoweringRule, _knn_query_lowering))

    ad.defjvp_zero(knn_query_p)

    def wrapper(database: RealArray, didxs: IntArray, query: RealArray, idxs: IntArray) -> IntArray:
        return cast(IntArray, knn_query_p.bind(database, didxs, query, idxs))

    return wrapper
