import pytest

from cudagen.types import (
    Assignment,
    BinaryOpcode,
    BinaryOperator,
    BitCast,
    SignExt,
    ZeroExt,
    Trunc,
    CudaType,
    CudaTypeId,
    Var,
)
from emission.expr import emit_expr, emit_assignment_stmt


t_i32 = CudaType(32, CudaTypeId.Unsigned)
t_i32s = CudaType(32, CudaTypeId.Signed)
t_f32 = CudaType(32, CudaTypeId.Float)
t_i64 = CudaType(64, CudaTypeId.Signed)
t_f64 = CudaType(64, CudaTypeId.Float)


def test_emit_expr_binary_add():
    expr = BinaryOperator(
        opcode=BinaryOpcode.Add,
        operand_a=Var("r2", t_i32),
        operand_b=Var("r3", t_i32),
    )
    assert emit_expr(expr) == "(r2 + r3)"


def test_emit_assignment_stmt():
    assign = Assignment(
        lhs=Var("r1", t_i32),
        rhs=BinaryOperator(
            opcode=BinaryOpcode.Add,
            operand_a=Var("r2", t_i32),
            operand_b=Var("r3", t_i32),
        ),
    )
    assert emit_assignment_stmt(assign) == "r1 = (r2 + r3);"


def test_emit_expr_bitcast_int_to_float():
    expr = BitCast(new_type=t_f32, operand=Var("r1", t_i32s))
    assert emit_expr(expr) == "__int_as_float(r1)"


def test_emit_expr_bitcast_float_to_int():
    expr = BitCast(new_type=t_i32, operand=Var("f1", t_f32))
    assert emit_expr(expr) == "__float_as_uint(f1)"


def test_emit_expr_bitcast_double_to_longlong():
    expr = BitCast(new_type=t_i64, operand=Var("d1", t_f64))
    assert emit_expr(expr) == "__double_as_longlong(d1)"


def test_emit_expr_bitcast_unsupported():
    with pytest.raises(ValueError):
        emit_expr(
            BitCast(
                new_type=t_i32, operand=Var("h1", CudaType(16, CudaTypeId.Unsigned))
            )
        )


def test_emit_expr_zeroext():
    src_ty = CudaType(16, CudaTypeId.Unsigned)
    dst_ty = CudaType(32, CudaTypeId.Unsigned)
    expr = ZeroExt(operand=Var("x", src_ty), new_type=dst_ty)
    assert emit_expr(expr) == "(unsigned int)(x)"


def test_emit_expr_signext():
    src_ty = CudaType(16, CudaTypeId.Signed)
    dst_ty = CudaType(32, CudaTypeId.Signed)
    expr = SignExt(operand=Var("x", src_ty), new_type=dst_ty)
    assert emit_expr(expr) == "(int)(x)"


def test_emit_trunc_signed():
    expr = Trunc(
        operand=Var("r1", CudaType(64, CudaTypeId.Signed)),
        new_type=CudaType(32, CudaTypeId.Signed),
    )
    assert emit_expr(expr) == "(int)(r1)"


def test_emit_trunc_unsigned():
    expr = Trunc(
        operand=Var("r1", CudaType(64, CudaTypeId.Unsigned)),
        new_type=CudaType(16, CudaTypeId.Unsigned),
    )
    assert emit_expr(expr) == "(unsigned short)(r1)"
