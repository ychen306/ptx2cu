from __future__ import annotations

import ptx

from typing import Mapping

from .types import InlineAsm, Var
from .utils import collect_registers, render_operand_with_index


def get_output_registers(instr: ptx.Instruction) -> list[ptx.Register]:
    """
    Return the first register operand(s), if any, as the output registers.
    If the first operand is a vector, return all its registers.
    """
    for op in instr.operands:
        if isinstance(op, ptx.Register):
            return [op]
        if isinstance(op, ptx.Vector):
            return list(op.values)
    return []


def emit_inline_asm(instr: ptx.Instruction, regmap: Mapping[ptx.Register, Var]) -> InlineAsm:
    """
    Emit an InlineAsm for a PTX Instruction using a register-to-Var mapping.

    - Output register: first register (or first register in a vector) is treated as output;
      it also participates as an input if reused (handled by shared Var mapping).
    - Template uses %N placeholders in operand order.
    - InlineAsm.arguments are the Vars in placeholder order; outputs are the Vars for the
      selected output registers (if any).
    """
    args: list[Var] = []
    idx = 0
    rendered_ops: list[str] = []
    for op in instr.operands:
        rendered, idx = render_operand_with_index(op, regmap, args, idx)
        rendered_ops.append(rendered)

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
