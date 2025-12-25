from __future__ import annotations

from typing import Callable

from cudagen.types import (
    Assignment,
    BitCast,
    SignExt,
    ZeroExt,
    Trunc,
    Compare,
    CompareOpcode,
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
    if isinstance(expr, (SignExt, ZeroExt)):
        src_ty = expr.operand.get_type()
        dst_ty = expr.new_type
        if src_ty is None:
            raise ValueError("Extension requires a source type")
        # Enforce integer-only, widening, and matching signedness per ext kind. If the
        # source type does not have the correct signedness (or is float), bitcast it first.
        assert not dst_ty.is_float, "Sign/ZeroExt expects integer destination types"
        assert dst_ty.bitwidth > src_ty.bitwidth, "Extension must widen the bitwidth"
        required_sign = (
            CudaTypeId.Signed if isinstance(expr, SignExt) else CudaTypeId.Unsigned
        )
        src_expr_str = emit_expr(expr.operand)
        if src_ty.is_float or src_ty.type_id != required_sign:
            cast_src_ty = CudaType(bitwidth=src_ty.bitwidth, type_id=required_sign)
            src_expr_str = f"({ _ctype_for_type(cast_src_ty) })({src_expr_str})"
        else:
            assert not src_ty.is_float, "Sign/ZeroExt expects integer source types"
        ctype = _ctype_for_type(dst_ty)
        return f"({ctype})({src_expr_str})"
    if isinstance(expr, Trunc):
        src_ty = expr.operand.get_type()
        dst_ty = expr.new_type
        if src_ty is None:
            raise ValueError("Trunc requires a source type")
        assert not dst_ty.is_float, "Trunc expects integer destination types"
        assert not src_ty.is_float, "Trunc expects integer source types"
        assert dst_ty.bitwidth < src_ty.bitwidth, "Trunc must narrow the bitwidth"
        assert dst_ty.is_signed == src_ty.is_signed, "Trunc preserves signedness"
        ctype = _ctype_for_type(dst_ty)
        return f"({ctype})({emit_expr(expr.operand)})"
    if isinstance(expr, Compare):
        lhs_ty = expr.operand_a.get_type()
        rhs_ty = expr.operand_b.get_type()
        assert (
            lhs_ty == rhs_ty and lhs_ty is not None
        ), "Compare operands must have matching types"
        assert not lhs_ty.is_float, "Compare only supports integer operands"

        match expr.opcode:
            case CompareOpcode.ICmpEQ:
                op_symbol = "=="
            case CompareOpcode.ICmpNE:
                op_symbol = "!="
            case CompareOpcode.ICmpSLT | CompareOpcode.ICmpULT:
                op_symbol = "<"
            case CompareOpcode.ICmpSLE | CompareOpcode.ICmpULE:
                op_symbol = "<="
            case CompareOpcode.ICmpSGT | CompareOpcode.ICmpUGT:
                op_symbol = ">"
            case CompareOpcode.ICmpSGE | CompareOpcode.ICmpUGE:
                op_symbol = ">="
            case _:
                raise ValueError(f"Unsupported compare opcode: {expr.opcode}")

        if expr.opcode in {
            CompareOpcode.ICmpSLT,
            CompareOpcode.ICmpSLE,
            CompareOpcode.ICmpSGT,
            CompareOpcode.ICmpSGE,
        }:
            assert lhs_ty.is_signed, "Signed compare requires signed operands"
        if expr.opcode in {
            CompareOpcode.ICmpULT,
            CompareOpcode.ICmpULE,
            CompareOpcode.ICmpUGT,
            CompareOpcode.ICmpUGE,
        }:
            assert not lhs_ty.is_signed, "Unsigned compare requires unsigned operands"

        return f"({emit_expr(expr.operand_a)} {op_symbol} {emit_expr(expr.operand_b)})"
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
