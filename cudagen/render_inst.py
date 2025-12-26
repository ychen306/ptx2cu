from __future__ import annotations

import ptx

from typing import Mapping, Optional

import re

from .types import (
    InlineAsm,
    Var,
    Expr,
    AddressOf,
    BinaryOpcode,
    BinaryOperator,
    Assignment,
    BitCast,
    CudaType,
    CudaTypeId,
    ConstantInt,
    CudaPointerType,
    Load,
    Store,
    SignExt,
    ZeroExt,
    Trunc,
    Compare,
    CompareOpcode,
)
from .datatype import type_info_for_datatype
from .utils import collect_registers, render_operand_with_index


def get_output_registers(instr: ptx.Instruction) -> list[ptx.Register]:
    """
    Return the first register operand(s), if any, as the output registers.
    If the first operand is a vector, return all its registers.
    Store instructions (st.*) have no outputs.
    """
    if instr.opcode.startswith("st."):
        return []
    first = instr.operands[0] if instr.operands else None
    if isinstance(first, ptx.Register):
        return [first]
    if isinstance(first, ptx.Vector):
        return list(first.values)
    return []


def _opcode_bitwidth(opcode: str) -> Optional[int]:
    m = re.findall(r"\.(?:b|s|u|f)(\d+)", opcode)
    if not m:
        return None
    return int(m[-1])


def _get_mnemonic(opcode: str) -> str:
    """
    Strip type suffixes and keep relevant modifier (e.g., mul.lo).
    """
    parts = opcode.split(".")
    if not parts:
        return opcode
    if parts[0] == "mad" and len(parts) > 1 and parts[1] == "lo":
        return "mad.lo"
    if parts[0] == "mul" and len(parts) > 1 and parts[1] in {"lo", "hi", "wide"}:
        return f"{parts[0]}.{parts[1]}"
    return parts[0]


def _parse_int_suffix(opcode: str, prefix: str) -> Optional[tuple[bool, int]]:
    """
    Parse signedness and bitwidth for opcodes like mul.wide.s32 or mad.lo.u32.
    Returns (signed, bitwidth) or None if not matched.
    """
    m = re.search(rf"{re.escape(prefix)}\.(s|u)(\d+)", opcode)
    if not m:
        return None
    signed = m.group(1) == "s"
    bitwidth = int(m.group(2))
    return signed, bitwidth


def _ensure_expr_type(expr: Expr, target: CudaType) -> Expr:
    """
    Ensure the expression has the requested type, inserting a BitCast if needed.
    """
    if expr.get_type() == target:
        return expr
    return BitCast(new_type=target, operand=expr)


def _widen_operand(
    expr: Expr, input_ty: CudaType, wide_ty: CudaType, signed: bool
) -> Expr:
    """
    Bitcast to input_ty if needed, then sign/zero extend to wide_ty.
    """
    narrowed = _ensure_expr_type(expr, input_ty)
    ext_cls = SignExt if signed else ZeroExt
    return ext_cls(operand=narrowed, new_type=wide_ty)


def emit_mov(
    instr: ptx.Instruction, regmap: Mapping[ptx.Register, Var]
) -> Optional[Assignment]:
    """
    Lower a simple mov.* into an Assignment, bitcasting the RHS to the LHS type when needed.
    Supports register destination and register/immediate source.
    """
    if instr.predicate is not None:
        return None
    if not instr.opcode.startswith("mov"):
        return None
    if len(instr.operands) < 2:
        return None

    dest_op, src_op = instr.operands[0], instr.operands[1]
    if not isinstance(dest_op, ptx.Register):
        return None
    dest_var = regmap.get(dest_op)
    if dest_var is None:
        return None
    dest_ty = dest_var.get_type()
    if dest_ty is None:
        return None

    rhs_expr: Expr
    if isinstance(src_op, ptx.Register):
        src_var = regmap.get(src_op)
        if src_var is None:
            return None
        if src_var.get_type() != dest_ty:
            rhs_expr = BitCast(new_type=dest_ty, operand=src_var)
        else:
            rhs_expr = src_var
    elif isinstance(src_op, ptx.Immediate):
        try:
            imm_val = int(src_op.value, 0)
        except ValueError:
            return None
        rhs_expr = ConstantInt(value=imm_val, ty=dest_ty)
    else:
        return None

    return Assignment(lhs=dest_var, rhs=rhs_expr)


def emit_mad_lo(
    instr: ptx.Instruction, regmap: Mapping[ptx.Register, Var]
) -> Optional[Assignment]:
    """
    Lower mad.lo.{s,u}<bits> into widening multiply + trunc + add.
    """
    if instr.predicate is not None:
        return None
    mnemonic = _get_mnemonic(instr.opcode)
    if mnemonic != "mad.lo":
        return None
    if len(instr.operands) < 4:
        return None

    parsed = _parse_int_suffix(instr.opcode, "mad.lo")
    if parsed is None:
        return None
    signed, input_bw = parsed
    input_ty = CudaType(
        bitwidth=input_bw,
        type_id=CudaTypeId.Signed if signed else CudaTypeId.Unsigned,
        represents_predicate=False,
    )
    wide_ty = CudaType(
        bitwidth=input_bw * 2,
        type_id=CudaTypeId.Signed if signed else CudaTypeId.Unsigned,
        represents_predicate=False,
    )

    dest_op, mul_a_op, mul_b_op, add_op = instr.operands[0:4]
    if not isinstance(dest_op, ptx.Register):
        return None
    if not isinstance(mul_a_op, ptx.Register):
        return None
    if not isinstance(mul_b_op, ptx.Register):
        return None
    if not isinstance(add_op, (ptx.Register, ptx.Immediate)):
        return None

    dest_var = regmap.get(dest_op)
    mul_a_var = regmap.get(mul_a_op)
    mul_b_var = regmap.get(mul_b_op)
    add_var = regmap.get(add_op) if isinstance(add_op, ptx.Register) else None
    if dest_var is None or mul_a_var is None or mul_b_var is None:
        return None
    if isinstance(add_op, ptx.Register) and add_var is None:
        return None

    mul_a_expr = _widen_operand(mul_a_var, input_ty, wide_ty, signed)
    mul_b_expr = _widen_operand(mul_b_var, input_ty, wide_ty, signed)
    prod = BinaryOperator(
        opcode=BinaryOpcode.Mul, operand_a=mul_a_expr, operand_b=mul_b_expr
    )
    low = Trunc(operand=prod, new_type=input_ty)

    if isinstance(add_op, ptx.Register):
        assert add_var is not None
        add_expr = _ensure_expr_type(add_var, input_ty)
    else:
        try:
            imm_val = int(add_op.value, 0)
        except ValueError:
            return None
        add_expr = ConstantInt(value=imm_val, ty=input_ty)

    rhs: Expr = BinaryOperator(
        opcode=BinaryOpcode.Add,
        operand_a=low,
        operand_b=add_expr,
    )

    dest_ty = dest_var.get_type()
    if dest_ty is not None and dest_ty != input_ty:
        rhs = BitCast(new_type=dest_ty, operand=rhs)

    return Assignment(lhs=dest_var, rhs=rhs)


def emit_predicate(
    instr: ptx.Instruction, regmap: Mapping[ptx.Register, Var]
) -> Optional[Assignment]:
    """
    Lower setp.* predicate instructions into a Compare assignment.
    Supports integer signed/unsigned comparisons; floats are not lowered.
    """
    if instr.predicate is not None:
        return None
    if not instr.opcode.startswith("setp."):
        return None
    if len(instr.operands) < 3:
        return None

    dest_op, lhs_op, rhs_op = instr.operands[0:3]
    if not isinstance(dest_op, ptx.Register):
        return None
    if not isinstance(lhs_op, ptx.Register):
        return None
    if not isinstance(rhs_op, (ptx.Register, ptx.Immediate)):
        return None

    dest_var = regmap.get(dest_op)
    lhs_var = regmap.get(lhs_op)
    rhs_var = regmap.get(rhs_op) if isinstance(rhs_op, ptx.Register) else None
    if (
        dest_var is None
        or lhs_var is None
        or (isinstance(rhs_op, ptx.Register) and rhs_var is None)
    ):
        return None

    # Parse compare operator and type suffix
    parts = instr.opcode.split(".")
    if len(parts) < 3:
        return None
    cmp_token = parts[1]
    type_token = parts[2]

    # Determine comparison opcode (llvm-style)
    cmp_opcode_map_signed = {
        "lt": CompareOpcode.ICmpSLT,
        "le": CompareOpcode.ICmpSLE,
        "gt": CompareOpcode.ICmpSGT,
        "ge": CompareOpcode.ICmpSGE,
    }
    cmp_opcode_map_unsigned = {
        "lt": CompareOpcode.ICmpULT,
        "le": CompareOpcode.ICmpULE,
        "gt": CompareOpcode.ICmpUGT,
        "ge": CompareOpcode.ICmpUGE,
    }

    if cmp_token == "eq":
        cmp_opcode = CompareOpcode.ICmpEQ
    elif cmp_token == "ne" or cmp_token == "neu":
        cmp_opcode = CompareOpcode.ICmpNE
    elif cmp_token in cmp_opcode_map_signed:
        signedness = (
            "s"
            if type_token.startswith("s")
            else ("u" if type_token.startswith("u") else "u")
        )
        cmp_opcode = (
            cmp_opcode_map_signed[cmp_token]
            if signedness == "s"
            else cmp_opcode_map_unsigned[cmp_token]
        )
    else:
        return None

    # Determine operand type from suffix (integer only)
    m = re.search(r"(s|u|b)(\d+)", type_token)
    if not m:
        return None
    bits = int(m.group(2))
    is_signed = m.group(1) == "s"
    cmp_ty = CudaType(
        bitwidth=bits,
        type_id=CudaTypeId.Signed if is_signed else CudaTypeId.Unsigned,
        represents_predicate=False,
    )

    lhs_expr: Expr = _ensure_expr_type(lhs_var, cmp_ty)
    if isinstance(rhs_op, ptx.Register):
        rhs_expr: Expr = _ensure_expr_type(rhs_var, cmp_ty)  # type: ignore[arg-type]
    else:
        try:
            imm_val = int(rhs_op.value, 0)
        except ValueError:
            return None
        rhs_expr = ConstantInt(value=imm_val, ty=cmp_ty)

    cmp_expr = Compare(opcode=cmp_opcode, operand_a=lhs_expr, operand_b=rhs_expr)
    cmp_ty_result = cmp_expr.get_type()
    dest_ty = dest_var.get_type()
    rhs_final: Expr = cmp_expr
    if dest_ty is not None and cmp_ty_result is not None and dest_ty != cmp_ty_result:
        rhs_final = BitCast(new_type=dest_ty, operand=cmp_expr)

    return Assignment(lhs=dest_var, rhs=rhs_final)


def emit_cvt(
    instr: ptx.Instruction, regmap: Mapping[ptx.Register, Var]
) -> Optional[Assignment]:
    """
    Lower integer cvt.<dst>.<src> instructions into SignExt/ZeroExt/Trunc/BitCast as needed.
    """
    if instr.predicate is not None:
        return None
    if not instr.opcode.startswith("cvt."):
        return None
    if len(instr.operands) < 2:
        return None

    parts = instr.opcode.split(".")
    if len(parts) < 3:
        return None
    dst_tok, src_tok = parts[1], parts[2]

    def _parse_int_tok(tok: str) -> Optional[tuple[bool, int]]:
        m = re.match(r"(s|u|b)(\d+)", tok)
        if not m:
            return None
        return (m.group(1) == "s", int(m.group(2)))

    dst_parsed = _parse_int_tok(dst_tok)
    src_parsed = _parse_int_tok(src_tok)
    if dst_parsed is None or src_parsed is None:
        return None

    dst_signed, dst_bw = dst_parsed
    src_signed, src_bw = src_parsed

    dest_op, src_op = instr.operands[0], instr.operands[1]
    if not isinstance(dest_op, ptx.Register):
        return None
    dest_var = regmap.get(dest_op)
    if dest_var is None:
        return None

    dst_ty = CudaType(
        bitwidth=dst_bw,
        type_id=CudaTypeId.Signed if dst_signed else CudaTypeId.Unsigned,
        represents_predicate=False,
    )
    src_ty = CudaType(
        bitwidth=src_bw,
        type_id=CudaTypeId.Signed if src_signed else CudaTypeId.Unsigned,
        represents_predicate=False,
    )

    # Build source expression
    if isinstance(src_op, ptx.Register):
        src_var = regmap.get(src_op)
        if src_var is None:
            return None
        src_expr: Expr = _ensure_expr_type(src_var, src_ty)
    elif isinstance(src_op, ptx.Immediate):
        try:
            imm_val = int(src_op.value, 0)
        except ValueError:
            return None
        src_expr = ConstantInt(value=imm_val, ty=src_ty)
    else:
        return None

    rhs: Expr
    if dst_bw > src_bw:
        ext_cls = SignExt if dst_signed else ZeroExt
        rhs = ext_cls(operand=src_expr, new_type=dst_ty)
    elif dst_bw == src_bw:
        # Only need to adjust signedness if mismatched
        if src_ty != dst_ty:
            rhs = BitCast(new_type=dst_ty, operand=src_expr)
        else:
            rhs = src_expr
    else:
        rhs = Trunc(operand=src_expr, new_type=dst_ty)

    dest_ty = dest_var.get_type()
    if dest_ty is not None and dest_ty != dst_ty:
        rhs = BitCast(new_type=dest_ty, operand=rhs)

    return Assignment(lhs=dest_var, rhs=rhs)


def emit_assignment(
    instr: ptx.Instruction, regmap: Mapping[ptx.Register, Var]
) -> Optional[Assignment]:
    """
    Attempt to lower a PTX instruction into a pure CUDA assignment (BinaryOperator).
    Returns None if the opcode/shape is unsupported.
    """
    if instr.predicate is not None:
        return None
    if len(instr.operands) < 3:
        return None

    dest_op, src0_op, src1_op = instr.operands[0:3]
    if not isinstance(dest_op, ptx.Register):
        return None
    if not isinstance(src0_op, ptx.Register):
        return None
    if not isinstance(src1_op, (ptx.Register, ptx.Immediate)):
        return None

    dest = regmap.get(dest_op)
    src0 = regmap.get(src0_op)
    src1 = regmap.get(src1_op) if isinstance(src1_op, ptx.Register) else None
    if (
        dest is None
        or src0 is None
        or (isinstance(src1_op, ptx.Register) and src1 is None)
    ):
        return None

    rhs = emit_binary_expr(instr, regmap)
    if rhs is None:
        return None

    # Ensure output type matches RHS type
    dest_ty = dest.get_type()
    rhs_ty = rhs.get_type()
    if dest_ty is None or rhs_ty is None:
        return None
    if dest_ty != rhs_ty:
        rhs = BitCast(new_type=dest_ty, operand=rhs)

    return Assignment(lhs=dest, rhs=rhs)


def emit_binary_expr(
    instr: ptx.Instruction, regmap: Mapping[ptx.Register, Var]
) -> Optional[Expr]:
    """
    Lower a PTX instruction with two register sources into a BinaryOperator.
    Returns None if opcode/shape is unsupported.
    """
    if instr.predicate is not None:
        return None
    if len(instr.operands) < 3:
        return None

    src0_op, src1_op = instr.operands[1], instr.operands[2]
    if not isinstance(src0_op, ptx.Register):
        return None
    # src1 can be register or immediate int for some ops (e.g., shr)
    if not isinstance(src1_op, (ptx.Register, ptx.Immediate)):
        return None

    src0 = regmap.get(src0_op)
    if src0 is None:
        return None

    ty0 = src0.get_type()
    if ty0 is None:
        return None

    opcode_bw = _opcode_bitwidth(instr.opcode) or ty0.bitwidth
    mnemonic = _get_mnemonic(instr.opcode)
    if mnemonic == "mul.wide":
        parsed = _parse_int_suffix(instr.opcode, "mul.wide")
        if parsed is None:
            return None
        signed, input_bw = parsed
        target_bw = input_bw * 2
        input_ty = CudaType(
            bitwidth=input_bw,
            type_id=CudaTypeId.Signed if signed else CudaTypeId.Unsigned,
            represents_predicate=False,
        )
        target_ty = CudaType(
            bitwidth=target_bw,
            type_id=CudaTypeId.Signed if signed else CudaTypeId.Unsigned,
            represents_predicate=False,
        )

        src0_expr: Expr = _ensure_expr_type(src0, input_ty)
        if isinstance(src1_op, ptx.Register):
            src1_var = regmap.get(src1_op)
            if src1_var is None:
                return None
            src1_expr: Expr = _ensure_expr_type(src1_var, input_ty)
        else:
            try:
                imm_val = int(src1_op.value, 0)
            except ValueError:
                return None
            src1_expr = ConstantInt(value=imm_val, ty=target_ty)

        op_a = _widen_operand(src0_expr, input_ty, target_ty, signed)
        op_b: Expr = (
            src1_expr
            if isinstance(src1_expr, ConstantInt)
            else _widen_operand(src1_expr, input_ty, target_ty, signed)
        )

        return BinaryOperator(
            opcode=BinaryOpcode.Mul,
            operand_a=op_a,
            operand_b=op_b,
        )

    # Decide float/int based purely on opcode suffix, not operand types.
    is_float_opcode = any(part.startswith("f") for part in instr.opcode.split("."))
    # Preserve operand signedness when not float; opcode signedness is secondary.
    is_signed_opcode = any(part.startswith("s") for part in instr.opcode.split("."))
    target_ty = CudaType(
        bitwidth=opcode_bw,
        type_id=(
            CudaTypeId.Float
            if is_float_opcode
            else (CudaTypeId.Signed if ty0.is_signed else CudaTypeId.Unsigned)
        ),
        represents_predicate=False,
    )
    if target_ty.bitwidth == 16 and target_ty.type_id != CudaTypeId.Float:
        return None

    opcode_map: dict[str, BinaryOpcode] = {
        "add": BinaryOpcode.FAdd if is_float_opcode else BinaryOpcode.Add,
        "and": BinaryOpcode.And,
        "or": BinaryOpcode.Or,
        "xor": BinaryOpcode.Xor,
        "shl": BinaryOpcode.Shl,
        "mul": BinaryOpcode.FMul if is_float_opcode else BinaryOpcode.Mul,
        "mul.lo": BinaryOpcode.Mul,
        "shr": (
            BinaryOpcode.AShr if "s" in instr.opcode.split(".") else BinaryOpcode.LShr
        ),
    }

    op = opcode_map.get(mnemonic)
    if op is None:
        return None

    # Basic float/int consistency guard based on opcode class
    if is_float_opcode and op not in {BinaryOpcode.FAdd, BinaryOpcode.FMul}:
        return None
    if not is_float_opcode and op in {BinaryOpcode.FAdd, BinaryOpcode.FMul}:
        return None
    # Bail if we'd need to mix int16 with float16 (unsupported)
    if target_ty.bitwidth == 16 and target_ty.type_id == CudaTypeId.Float:
        if isinstance(src1_op, ptx.Register):
            src1_var = regmap.get(src1_op)
            if src1_var is None:
                return None
            src1_ty = src1_var.get_type()
            if (
                src1_ty is not None
                and src1_ty.bitwidth == 16
                and src1_ty.type_id != CudaTypeId.Float
            ):
                return None

    op_a_expr: Expr = src0 if src0.get_type() == target_ty else BitCast(target_ty, src0)
    if isinstance(src1_op, ptx.Register):
        src1_var = regmap.get(src1_op)
        if src1_var is None:
            return None
        op_b_expr: Expr = (
            src1_var
            if src1_var.get_type() == target_ty
            else BitCast(target_ty, src1_var)
        )
    else:
        # immediate
        try:
            imm_val = int(src1_op.value, 0)
        except ValueError:
            return None
        op_b_expr = ConstantInt(value=imm_val, ty=target_ty)

    return BinaryOperator(opcode=op, operand_a=op_a_expr, operand_b=op_b_expr)


def emit_ld_global(
    instr: ptx.Instruction, regmap: Mapping[ptx.Register, Var]
) -> Optional[Load]:
    """
    Lower ld.global.* into a Load IR node. Returns None if unsupported.
    """
    if instr.predicate is not None:
        return None
    if not instr.opcode.startswith("ld.global"):
        return None
    if len(instr.operands) < 2:
        return None

    dest_op = instr.operands[0]
    src_op = instr.operands[1]
    if not isinstance(dest_op, ptx.Register):
        return None
    if not isinstance(src_op, ptx.MemoryRef) or not isinstance(
        src_op.base, ptx.Register
    ):
        return None

    dest_var = regmap.get(dest_op)
    base_ptr = regmap.get(src_op.base)
    if dest_var is None or base_ptr is None:
        return None

    # Determine element type from opcode suffix
    op_suffix = instr.opcode.split(".")[-1]
    elem_type = op_suffix if op_suffix != "global" else "u32"
    _, bitwidth, is_float = type_info_for_datatype(elem_type)
    target_ty = CudaType(
        bitwidth=bitwidth,
        type_id=CudaTypeId.Float if is_float else CudaTypeId.Unsigned,
        represents_predicate=False,
    )

    elem_size = target_ty.bitwidth // 8
    offset = src_op.offset or 0
    if offset % elem_size != 0:
        return None
    index = offset // elem_size

    # Ensure pointer type matches element; bitcast pointer if needed
    base_expr: Expr = base_ptr
    base_ty = base_ptr.get_type()
    desired_ptr_ty = CudaPointerType(elem=target_ty)
    if not isinstance(base_ty, CudaPointerType) or base_ty.elem != target_ty:
        base_expr = BitCast(new_type=desired_ptr_ty, operand=base_ptr)

    return Load(
        ty=target_ty, dst=dest_var, src=base_expr, offset=offset, is_param=False
    )


def emit_st_global(
    instr: ptx.Instruction, regmap: Mapping[ptx.Register, Var]
) -> Optional[Store]:
    """
    Lower st.global.* into a Store IR node. Returns None if unsupported.
    """
    if instr.predicate is not None:
        return None
    if not instr.opcode.startswith("st.global"):
        return None
    if len(instr.operands) < 2:
        return None

    dest_ptr_op = instr.operands[0]
    value_op = instr.operands[1]
    if not isinstance(dest_ptr_op, ptx.MemoryRef) or not isinstance(
        dest_ptr_op.base, ptx.Register
    ):
        return None

    base_ptr = regmap.get(dest_ptr_op.base)
    if base_ptr is None:
        return None

    # Determine element type from opcode suffix
    op_suffix = instr.opcode.split(".")[-1]
    elem_type = op_suffix if op_suffix != "global" else "u32"
    _, bitwidth, is_float = type_info_for_datatype(elem_type)
    target_ty = CudaType(
        bitwidth=bitwidth,
        type_id=CudaTypeId.Float if is_float else CudaTypeId.Unsigned,
        represents_predicate=False,
    )
    elem_size = target_ty.bitwidth // 8

    offset = dest_ptr_op.offset or 0
    if offset % elem_size != 0:
        return None

    # Pointer expression, bitcast if necessary
    base_expr: Expr = base_ptr
    base_ty = base_ptr.get_type()
    desired_ptr_ty = CudaPointerType(elem=target_ty)
    if not isinstance(base_ty, CudaPointerType) or base_ty.elem != target_ty:
        base_expr = BitCast(new_type=desired_ptr_ty, operand=base_ptr)

    # Value expression
    value_expr: Expr
    if isinstance(value_op, ptx.Register):
        val_var = regmap.get(value_op)
        if val_var is None:
            return None
        val_ty = val_var.get_type()
        if val_ty != target_ty:
            # Do not attempt unsupported half/int bitcasts; fall back to asm.
            if (
                target_ty.bitwidth == 16
                and isinstance(val_ty, CudaType)
                and val_ty.is_float != target_ty.is_float
            ):
                return None
            value_expr = BitCast(target_ty, val_var)
        else:
            value_expr = val_var
    elif isinstance(value_op, ptx.Immediate):
        try:
            imm_val = int(value_op.value, 0)
        except ValueError:
            return None
        value_expr = ConstantInt(value=imm_val, ty=target_ty)
    else:
        return None

    return Store(pointer=base_expr, offset=offset, value=value_expr)


def emit_inline_asm(
    instr: ptx.Instruction, regmap: Mapping[ptx.Register, Var]
) -> InlineAsm:
    """
    Emit an InlineAsm for a PTX Instruction using a register-to-Var mapping.

    - Output register: first register (or first register in a vector) is treated as output;
      it also participates as an input if reused (handled by shared Var mapping).
    - Template uses %N placeholders in operand order.
    - InlineAsm.arguments are the Vars in placeholder order; outputs are the Vars for the
      selected output registers (if any).
    """
    if instr.predicate is not None:
        raise ValueError(
            "Predicated instructions are not yet lowered in emit_inline_asm"
        )

    args: list[Expr] = []
    idx = 0
    rendered_ops: list[str] = []
    for op in instr.operands:
        rendered, idx = render_operand_with_index(op, regmap, args, idx)
        rendered_ops.append(rendered)

    op_bw = _opcode_bitwidth(instr.opcode)
    if op_bw and op_bw != 64:
        for arg in args:
            if isinstance(arg, AddressOf) and arg.bitwidth is None:
                arg.bitwidth = op_bw

    out_regs = get_output_registers(instr)
    out_vars: list[Var] = []
    for r in out_regs:
        var = regmap.get(r)
        if var:
            out_vars.append(var)

    template = f"{instr.opcode} " + ", ".join(rendered_ops) + ";"

    clobbers_memory = any(isinstance(op, ptx.MemoryRef) for op in instr.operands)

    return InlineAsm(
        template=template,
        arguments=args,
        outputs=out_vars,
        clobbers_memory=clobbers_memory,
    )
