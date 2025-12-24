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
    ConstantInt,
)
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
    if parts[0] == "mul" and len(parts) > 1 and parts[1] in {"lo", "hi", "wide"}:
        return f"{parts[0]}.{parts[1]}"
    return parts[0]


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
    if dest is None or src0 is None or (isinstance(src1_op, ptx.Register) and src1 is None):
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
    # Decide float/int based purely on opcode suffix, not operand types.
    is_float_opcode = any(part.startswith("f") for part in instr.opcode.split("."))
    # Preserve operand signedness when not float; opcode signedness is secondary.
    is_signed_opcode = any(part.startswith("s") for part in instr.opcode.split("."))
    target_ty = CudaType(
        bitwidth=opcode_bw,
        is_float=is_float_opcode,
        represents_predicate=False,
        is_signed=ty0.is_signed if not is_float_opcode else False,
    )
    if target_ty.bitwidth == 16:
        return None

    mnemonic = _get_mnemonic(instr.opcode)
    if mnemonic == "mul.wide":
        return None

    opcode_map: dict[str, BinaryOpcode] = {
        "add": BinaryOpcode.FAdd if is_float_opcode else BinaryOpcode.Add,
        "and": BinaryOpcode.And,
        "or": BinaryOpcode.Or,
        "xor": BinaryOpcode.Xor,
        "shl": BinaryOpcode.Shl,
        "mul": BinaryOpcode.FMul if is_float_opcode else BinaryOpcode.Mul,
        "mul.lo": BinaryOpcode.Mul,
        "shr": BinaryOpcode.AShr if "s" in instr.opcode.split(".") else BinaryOpcode.LShr,
    }

    op = opcode_map.get(mnemonic)
    if op is None:
        return None

    # Basic float/int consistency guard based on opcode class
    if is_float_opcode and op not in {BinaryOpcode.FAdd, BinaryOpcode.FMul}:
        return None
    if not is_float_opcode and op in {BinaryOpcode.FAdd, BinaryOpcode.FMul}:
        return None

    op_a: Expr = src0 if src0.get_type() == target_ty else BitCast(target_ty, src0)
    if isinstance(src1_op, ptx.Register):
        src1_var = regmap.get(src1_op)
        if src1_var is None:
            return None
        op_b: Expr = (
            src1_var if src1_var.get_type() == target_ty else BitCast(target_ty, src1_var)
        )
    else:
        # immediate
        try:
            imm_val = int(src1_op.value, 0)
        except ValueError:
            return None
        op_b = ConstantInt(value=imm_val, ty=target_ty)

    return BinaryOperator(opcode=op, operand_a=op_a, operand_b=op_b)


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
        raise ValueError("Predicated instructions are not yet lowered in emit_inline_asm")

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
