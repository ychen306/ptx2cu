from __future__ import annotations

from typing import Callable

from cudagen.types import (
    Assignment,
    BitCast,
    BinaryOpcode,
    BinaryOperator,
    CudaType,
    CudaTypeId,
    CudaPointerType,
    Expr,
    ConstantInt,
    Var,
)


def _binary_op_symbol(op: BinaryOpcode) -> str:
    match op:
        case BinaryOpcode.Add | BinaryOpcode.FAdd:
            return "+"
        case BinaryOpcode.Sub:
            return "-"
        case BinaryOpcode.Mul | BinaryOpcode.FMul:
            return "*"
        case BinaryOpcode.SDiv | BinaryOpcode.UDiv | BinaryOpcode.FDiv:
            return "/"
        case BinaryOpcode.And:
            return "&"
        case BinaryOpcode.Or:
            return "|"
        case BinaryOpcode.Xor:
            return "^"
        case BinaryOpcode.Shl:
            return "<<"
        case BinaryOpcode.AShr | BinaryOpcode.LShr:
            return ">>"
    raise ValueError(f"Unsupported binary opcode: {op}")


def _bitcast_intrinsic(src: CudaType, dst: CudaType) -> str:
    # Only support 32/64-bit float <-> int conversions for now.
    if src.bitwidth == 32 and dst.bitwidth == 32:
        if src.is_float and not dst.is_float:
            return "__float_as_int" if dst.is_signed else "__float_as_uint"
        if dst.is_float and not src.is_float:
            return "__int_as_float" if src.is_signed else "__uint_as_float"
    if src.bitwidth == 64 and dst.bitwidth == 64:
        if src.is_float and not dst.is_float:
            return "__double_as_longlong"
        if dst.is_float and not src.is_float:
            return "__longlong_as_double"
    raise ValueError(f"Unsupported bitcast from {src} to {dst}")


def _ctype_for_type(ty: CudaType | CudaPointerType) -> str:
    if ty.is_float:
        if ty.bitwidth == 64:
            return "double"
        if ty.bitwidth == 16:
            return "__half"
        if ty.bitwidth == 32:
            return "float"
        raise ValueError(f"Unsupported float bitwidth: {ty.bitwidth}")
    if isinstance(ty, CudaPointerType):
        return _ctype_for_type(ty.elem) + "*"
    if ty.bitwidth == 64:
        return "long long" if ty.is_signed else "unsigned long long"
    if ty.bitwidth == 32:
        return "int" if ty.is_signed else "unsigned int"
    if ty.bitwidth == 16:
        return "short" if ty.is_signed else "unsigned short"
    raise ValueError(f"Unsupported integer bitwidth: {ty.bitwidth}")


def emit_expr(expr: Expr) -> str:
    if isinstance(expr, Var):
        return expr.name
    if isinstance(expr, BitCast):
        src_ty = expr.operand.get_type()
        dst_ty = expr.new_type
        if src_ty is None or dst_ty is None:
            raise ValueError("BitCast requires source and destination types")
        if isinstance(dst_ty, CudaPointerType):
            return f"reinterpret_cast<{_ctype_for_type(dst_ty)}>({emit_expr(expr.operand)})"
        # If both integer and same width, use a C cast to switch signedness.
        if (
            not src_ty.is_float
            and not dst_ty.is_float
            and src_ty.bitwidth == dst_ty.bitwidth
        ):
            return f"({ _ctype_for_type(dst_ty) })({emit_expr(expr.operand)})"
        if dst_ty.is_float and dst_ty.bitwidth == 16:
            raise ValueError("16-bit float bitcasts are not supported in emission")
        intrinsic = (
            _bitcast_intrinsic(src_ty, dst_ty)
            if isinstance(src_ty, CudaType) and isinstance(dst_ty, CudaType)
            else None
        )
        if intrinsic is None and isinstance(dst_ty, CudaPointerType):
            return f"reinterpret_cast<{_ctype_for_type(dst_ty)}>({emit_expr(expr.operand)})"
        if intrinsic is None:
            raise ValueError(f"Unsupported bitcast from {src_ty} to {dst_ty}")
        return f"{intrinsic}({emit_expr(expr.operand)})"
    if isinstance(expr, ConstantInt):
        return str(expr.value)
    if isinstance(expr, BinaryOperator):
        op_symbol = _binary_op_symbol(expr.opcode)
        lhs = emit_expr(expr.operand_a)
        rhs = emit_expr(expr.operand_b)
        return f"({lhs} {op_symbol} {rhs})"
    raise ValueError(f"Unsupported expression type: {type(expr).__name__}")


def emit_assignment_stmt(assign: Assignment) -> str:
    return f"{emit_expr(assign.lhs)} = {emit_expr(assign.rhs)};"
