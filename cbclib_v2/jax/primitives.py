from functools import partial
from typing import Dict, Protocol, Sequence, Tuple, TypeAlias, Union, cast
import numpy as np
import jax.numpy as jnp
from jax import core, dtypes, custom_vjp
from jax.lib import xla_client
from jax.interpreters import mlir, xla
from jaxlib.hlo_helpers import custom_call
from ..src import cpu_ops
from ..annotations import RealArray, IntArray

# Helper functions

def default_layouts(*shapes):
    return [range(len(shape) - 1, -1, -1) for shape in shapes]

MLIRValue : TypeAlias = mlir.ir.Value
MLIRType : TypeAlias = mlir.ir.Type

class RankedTensorTypeProtocol(Protocol):
    shape : Sequence[Union[int, MLIRValue]]
    element_type : MLIRType

    @classmethod
    def get(cls, shape: Sequence[Union[int, MLIRValue]], element_type: MLIRType
            ) -> 'RankedTensorTypeProtocol':
        ...

RankedTensorType : RankedTensorTypeProtocol = mlir.ir.RankedTensorType

class MLIRArray(Protocol):
    type : RankedTensorTypeProtocol

# Register the CPU XLA custom calls
for _name, _value in cpu_ops.registrations().items():
    xla_client.register_custom_call_target(_name, _value, platform="cpu")

line_distances_fwd_keys : Dict[Tuple[str, str], str] = {
    ('float32', 'int32') : 'line_distances_fwd_f32_i32',
    ('float64', 'int32') : 'line_distances_fwd_f64_i32',
    ('float32', 'int64') : 'line_distances_fwd_f32_i64',
    ('float64', 'int64') : 'line_distances_fwd_f64_i64'
}

line_distances_bwd_keys : Dict[Tuple[str, str], str] = {
    ('float32', 'int32') : 'line_distances_bwd_f32_i32',
    ('float64', 'int32') : 'line_distances_bwd_f64_i32',
    ('float32', 'int64') : 'line_distances_bwd_f32_i64',
    ('float64', 'int64') : 'line_distances_bwd_f64_i64'
}

# Create _line_distances_fwd_p for forward operation
_line_distances_fwd_p = core.Primitive("line_distances_fwd")
_line_distances_fwd_p.multiple_results = True
_line_distances_fwd_p.def_impl(partial(xla.apply_primitive, _line_distances_fwd_p))

# Create abstract evaluation, used to figure out the shapes and datatypes of outputs
def _line_distances_fwd_abstract(image: core.ShapedArray, lines: core.ShapedArray, idxs: core.ShapedArray
                                 ) -> Tuple[core.ShapedArray, core.ShapedArray]:
    shape = image.shape
    float_dtype = dtypes.canonicalize_dtype(image.dtype)
    int_dtype = dtypes.canonicalize_dtype(idxs.dtype)

    assert lines.shape[:-1] == idxs.shape
    assert float_dtype == dtypes.canonicalize_dtype(lines.dtype)

    return (core.ShapedArray(shape, float_dtype),
            core.ShapedArray(shape, int_dtype))

_line_distances_fwd_p.def_abstract_eval(_line_distances_fwd_abstract)

# This provides a mechanism for exposing our custom C++ interfaces to the JAX XLA backend.
def _line_distances_fwd_cpu_lowering(ctx: mlir.LoweringRuleContext, out: MLIRArray, lines: MLIRArray, idxs: MLIRArray):
    out_type = out.type
    lines_type = lines.type
    idxs_type = idxs.type

    out_shape = out_type.shape
    lines_shape = lines_type.shape
    idxs_shape = idxs_type.shape

    _, lines_aval, idxs_aval = cast(Sequence[core.ShapedArray], ctx.avals_in)
    float_dtype = np.dtype(lines_aval.dtype)
    int_dtype = np.dtype(idxs_aval)

    op_name = line_distances_fwd_keys[(float_dtype.name, int_dtype.name)]

    return custom_call(
        op_name,
        result_types=[
            RankedTensorType.get(out_shape, out_type.element_type),
            RankedTensorType.get(out_shape, idxs_type.element_type)
        ],
        operands=[
            mlir.ir_constant(np.prod(out_shape[:-2], dtype=int)),
            mlir.ir_constant(out_shape[-2]),
            mlir.ir_constant(out_shape[-1]),
            mlir.ir_constant(np.prod(lines_shape[:-1], dtype=int)),
            mlir.ir_constant(ctx.module_context.backend.device_count()),
            lines, idxs,
        ],
        operand_layouts=5 * [(),] + default_layouts(lines_shape, idxs_shape),
        result_layouts=default_layouts(out_shape, out_shape)
    ).results

mlir.register_lowering(_line_distances_fwd_p, cast(mlir.LoweringRule, _line_distances_fwd_cpu_lowering),
                       platform="cpu")

# Create _line_distances_bwd_p for backward operation
_line_distances_bwd_p = core.Primitive("line_distances_bwd")
_line_distances_bwd_p.def_impl(partial(xla.apply_primitive, _line_distances_bwd_p))

def _line_distances_bwd_abstract(cotangents: core.ShapedArray, line_idxs: core.ShapedArray,
                                 lines: core.ShapedArray, idxs: core.ShapedArray):
    out_shape = lines.shape
    float_dtype = dtypes.canonicalize_dtype(cotangents.dtype)
    int_dtype = dtypes.canonicalize_dtype(line_idxs.dtype)

    assert cotangents.shape == line_idxs.shape
    assert lines.shape[:-1] == idxs.shape
    assert float_dtype == dtypes.canonicalize_dtype(lines.dtype)
    assert int_dtype == dtypes.canonicalize_dtype(idxs.dtype)

    return core.ShapedArray(out_shape, float_dtype)

_line_distances_bwd_p.def_abstract_eval(_line_distances_bwd_abstract)

def _line_distances_bwd_cpu_lowering(ctx: mlir.LoweringRuleContext, cotangents: MLIRArray,
                                     line_idxs: MLIRArray, lines: MLIRArray, idxs: MLIRArray):
    ct_type = cotangents.type
    lidxs_type = line_idxs.type
    lines_type = lines.type
    idxs_type = idxs.type

    ct_shape = ct_type.shape
    lidxs_shape = lidxs_type.shape
    lines_shape = lines_type.shape
    idxs_shape = idxs_type.shape

    ct_aval, lidxs_aval, _, _ = cast(Sequence[core.ShapedArray], ctx.avals_in)
    float_dtype = np.dtype(ct_aval.dtype)
    int_dtype = np.dtype(lidxs_aval.dtype)

    op_name = line_distances_bwd_keys[(float_dtype.name, int_dtype.name)]

    return custom_call(
        op_name,
        result_types=[RankedTensorType.get(lines_shape, ct_type.element_type),],
        operands=[
            mlir.ir_constant(np.prod(ct_shape[:-2], dtype=int)),
            mlir.ir_constant(ct_shape[-2]),
            mlir.ir_constant(ct_shape[-1]),
            mlir.ir_constant(np.prod(lines_shape[:-1], dtype=int)),
            mlir.ir_constant(ctx.module_context.backend.device_count()),
            cotangents, line_idxs, lines, idxs
        ],
        operand_layouts=5 * [(),] + default_layouts(ct_shape, lidxs_shape,
                                                    lines_shape, idxs_shape),
        result_layouts=default_layouts(lines_shape),
    ).results

mlir.register_lowering(_line_distances_bwd_p, cast(mlir.LoweringRule, _line_distances_bwd_cpu_lowering),
                       platform="cpu")

# Define JAX functions for custom differentiation rule
def line_distances_fwd(image: RealArray, lines: RealArray, idxs: IntArray
                       ) -> Tuple[RealArray, Tuple[IntArray, RealArray, IntArray]]:
    out, line_idxs = _line_distances_fwd_p.bind(image, lines, idxs)
    return out + image, (line_idxs, lines, idxs)

def line_distances_bwd(res: Tuple[IntArray, RealArray, IntArray], g: RealArray
                       ) -> Tuple[RealArray, RealArray, IntArray]:
    line_idxs, lines, idxs = res
    grad_lines = cast(RealArray, _line_distances_bwd_p.bind(g, line_idxs, lines, idxs))
    return g, grad_lines, jnp.zeros(idxs.shape, dtype=idxs.dtype)

# Define the main function
@partial(custom_vjp)
def line_distances(image: RealArray, lines: RealArray, idxs: IntArray) -> RealArray:
    out, _ = line_distances_fwd(image, lines, idxs)
    return out

line_distances.defvjp(line_distances_fwd, line_distances_bwd)
