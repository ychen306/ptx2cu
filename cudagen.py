from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import ptx

@dataclass(frozen=True)
class Var:
    name : str

@dataclass
class RegisterInfo:
    decl : ptx.RegisterDecl
    c_var : Var


@dataclass
class InlineAsm:
    template : str
    arguments : [Var]
    outputs : list[Var]
    constraints: dict[Var, str]
    clobbers_memory: bool = False

# Note: InlineAsm currently does not model clobbers or separate input/output lists.


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


def get_register_constraint(decl: ptx.RegisterDecl) -> str:
    """
    Map a register declaration to an inline asm constraint using its datatype.
    """
    dt = decl.datatype
    if dt.startswith("b8") or dt.startswith("u8") or dt.startswith("s8"):
        return "b"
    if dt.startswith("b16") or dt.startswith("u16") or dt.startswith("s16") or dt.startswith("f16"):
        return "h"
    if dt.startswith("b32") or dt.startswith("u32") or dt.startswith("s32"):
        return "r"
    if dt.startswith("f32"):
        return "f"
    if dt.startswith("b64") or dt.startswith("u64") or dt.startswith("s64"):
        return "l"
    if dt.startswith("f64"):
        return "d"
    return "r"

def _collect_registers(op: ptx.Operand) -> list[ptx.Register]:
    regs: list[ptx.Register] = []
    if isinstance(op, ptx.Register):
        regs.append(op)
    elif isinstance(op, ptx.Vector):
        regs.extend(op.values)
    elif isinstance(op, ptx.MemoryRef) and isinstance(op.base, ptx.Register):
        regs.append(op.base)
    return regs

def _render_operand_with_placeholders(
    operand: ptx.Operand,
    regmap: dict[ptx.Register, RegisterInfo],
    reg_to_placeholder: dict[ptx.Register, str],
) -> str:
    if isinstance(operand, ptx.Register):
        ph = reg_to_placeholder.get(operand)
        if ph is None:
            raise ValueError(f"Missing mapping for register {operand}")
        return ph
    if isinstance(operand, ptx.Vector):
        rendered = []
        for reg in operand.values:
            ph = reg_to_placeholder.get(reg)
            if ph is None:
                raise ValueError(f"Missing mapping for register {reg}")
            rendered.append(ph)
        return "{" + ", ".join(rendered) + "}"
    if isinstance(operand, ptx.Immediate):
        return str(operand.value)
    if isinstance(operand, ptx.MemoryRef):
        base_rendered = _render_operand_with_placeholders(
            operand.base, regmap, reg_to_placeholder
        )
        if operand.offset:
            return f"[{base_rendered}+{operand.offset}]"
        return f"[{base_rendered}]"
    if isinstance(operand, ptx.ParamRef):
        return operand.name
    raise ValueError(f"Unsupported operand type: {type(operand).__name__}")


def _render_operand_with_index(
    operand: ptx.Operand,
    regmap: dict[ptx.Register, RegisterInfo],
    constraints: dict[Var, str],
    args: list[Var],
    idx: int,
) -> tuple[str, int]:
    if isinstance(operand, ptx.Register):
        info = regmap.get(operand)
        if info is None:
            raise ValueError(f"Missing mapping for register {operand}")
        ph = f"%{idx}"
        args.append(info.c_var)
        constraints[info.c_var] = get_register_constraint(info.decl)
        return ph, idx + 1
    if isinstance(operand, ptx.Vector):
        rendered = []
        cur = idx
        for reg in operand.values:
            piece, cur = _render_operand_with_index(reg, regmap, constraints, args, cur)
            rendered.append(piece)
        return "{" + ", ".join(rendered) + "}", cur
    if isinstance(operand, ptx.Immediate):
        return str(operand.value), idx
    if isinstance(operand, ptx.MemoryRef):
        base_rendered, next_idx = _render_operand_with_index(operand.base, regmap, constraints, args, idx)
        if operand.offset:
            return f"[{base_rendered}+{operand.offset}]", next_idx
        return f"[{base_rendered}]", next_idx
    if isinstance(operand, ptx.ParamRef):
        return operand.name, idx
    raise ValueError(f"Unsupported operand type: {type(operand).__name__}")


def emit_inline_asm(instr: ptx.Instruction, regmap: dict[ptx.Register, RegisterInfo]) -> InlineAsm:
    """
    Emit an InlineAsm for a PTX Instruction using a register-to-Var mapping.

    - Output register: first register (or first register in a vector) is treated as output;
      it also participates as an input if reused (handled by shared Var mapping).
    - Template uses %N placeholders in operand order.
    - InlineAsm.arguments are the Vars in placeholder order; output is the Var for the
      selected output register (if any).
    """
    constraints: dict[Var, str] = {}

    args: list[Var] = []
    idx = 0
    rendered_ops: list[str] = []
    for op in instr.operands:
        rendered, idx = _render_operand_with_index(op, regmap, constraints, args, idx)
        rendered_ops.append(rendered)

    out_regs = get_output_registers(instr)
    out_vars: list[Var] = []
    for r in out_regs:
        info = regmap.get(r)
        if info:
            out_vars.append(info.c_var)

    template = f"{instr.opcode} " + ", ".join(rendered_ops) + ";"

    clobbers_memory = any(isinstance(op, ptx.MemoryRef) for op in instr.operands)

    return InlineAsm(
        template=template,
        arguments=args,
        outputs=out_vars,
        constraints=constraints,
        clobbers_memory=clobbers_memory,
    )


def emit_inline_asm_string(instr: ptx.Instruction, regmap: dict[ptx.Register, RegisterInfo]) -> str:
    """
    Emit a CUDA asm volatile string for a PTX instruction.
    """
    inline = emit_inline_asm(instr, regmap)

    def escape(s: str) -> str:
        return s.replace("\\", "\\\\").replace('"', '\\"')

    # Map Vars back to declarations
    var_to_decl = {
        info.c_var: info.decl
        for info in regmap.values()
    }
    placeholder_for_var = {var: f"%{idx}" for idx, var in enumerate(inline.arguments)}

    pred_vars = {
        var: decl
        for var, decl in var_to_decl.items()
        if decl.datatype.startswith("pred")
    }

    pred_temps = {var: f"ptmp{idx}" for idx, var in enumerate(pred_vars)}

    main_template = inline.template
    # Replace predicate placeholders in main template with temp predicate registers
    for var, temp in pred_temps.items():
        ph = placeholder_for_var.get(var)
        if ph:
            main_template = main_template.replace(ph, f"%{temp}")

    pre_lines: list[str] = []
    post_lines: list[str] = []

    if pred_temps:
        pre_lines.append(".reg .pred " + ", ".join(f"%{t}" for t in pred_temps.values()) + ";")
        for var, temp in pred_temps.items():
            ph = placeholder_for_var[var]
            pre_lines.append(f"setp.ne.u32 %{temp}, {ph}, 0;")

    for out_var in inline.outputs:
        if out_var in pred_temps:
            temp = pred_temps[out_var]
            ph_out = placeholder_for_var[out_var]
            post_lines.append(f"selp.u32 {ph_out}, 1, 0, %{temp};")

    template_parts = []
    if pre_lines or post_lines:
        template_parts.append("{")
        template_parts.extend(pre_lines)
        template_parts.append(main_template)
        template_parts.extend(post_lines)
        template_parts.append("}")
        final_template = " ".join(template_parts)
    else:
        final_template = main_template

    outputs = []
    for out_var in inline.outputs:
        c = inline.constraints.get(out_var, "r")
        outputs.append(f'"+{c}"({out_var.name})')

    inputs = []
    for arg in inline.arguments:
        if arg in inline.outputs:
            continue
        c = inline.constraints.get(arg, "r")
        inputs.append(f'"{c}"({arg.name})')

    clobbers = ['"memory"'] if inline.clobbers_memory else []

    outs_str = ", ".join(outputs)
    ins_str = ", ".join(inputs)
    clob_str = ", ".join(clobbers)

    return f'asm volatile("{escape(final_template)}" : {outs_str} : {ins_str} : {clob_str});'
