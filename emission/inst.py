from __future__ import annotations

from typing import Mapping

import ptx

from cudagen.render_inst import emit_inline_asm
from cudagen.types import Var


def emit_inline_asm_string(instr: ptx.Instruction, regmap: Mapping[ptx.Register, Var]) -> str:
    """
    Emit a CUDA asm volatile string for a PTX instruction.
    """
    inline = emit_inline_asm(instr, regmap)

    def escape(s: str) -> str:
        return s.replace("\\", "\\\\").replace('"', '\\"')

    placeholder_for_var = {var: f"%{idx}" for idx, var in enumerate(inline.arguments + inline.outputs)}

    pred_vars = [var for var in placeholder_for_var if var.represents_predicate]

    final_template = inline.template
    pre_lines: list[str] = []
    post_lines: list[str] = []

    # Only apply predicate bridging if any predicate vars are present
    if pred_vars:
        pred_temps = {var: f"ptmp{idx}" for idx, var in enumerate(pred_vars)}

        main_template = inline.template
        for var, temp in pred_temps.items():
            ph = placeholder_for_var.get(var)
            if ph:
                main_template = main_template.replace(ph, f"%{temp}")

        pre_lines.append(".reg .pred " + ", ".join(f"%{t}" for t in pred_temps.values()) + ";")
        for var, temp in pred_temps.items():
            ph = placeholder_for_var[var]
            pre_lines.append(f"setp.ne.u32 %{temp}, {ph}, 0;")

        for out_var in inline.outputs:
            if out_var in pred_temps:
                temp = pred_temps[out_var]
                ph_out = placeholder_for_var[out_var]
                post_lines.append(f"selp.u32 {ph_out}, 1, 0, %{temp};")

        template_parts = ["{"] + pre_lines + [main_template] + post_lines + ["}"]
        final_template = " ".join(template_parts)

    def constraint_for(var: Var) -> str:
        if var.is_float:
            if var.bitwidth == 64:
                return "d"
            return "f"
        if var.bitwidth == 64:
            return "l"
        if var.bitwidth == 32:
            return "r"
        if var.bitwidth == 16:
            return "h"
        if var.bitwidth == 8:
            return "b"
        return "r"

    outputs = []
    for out_var in inline.outputs:
        c = constraint_for(out_var)
        outputs.append(f'"+{c}"({out_var.name})')

    inputs = []
    for arg in inline.arguments:
        if arg in inline.outputs:
            continue
        c = constraint_for(arg)
        inputs.append(f'"{c}"({arg.name})')

    clobbers = ['"memory"'] if inline.clobbers_memory else []

    outs_str = ", ".join(outputs)
    ins_str = ", ".join(inputs)
    clob_str = ", ".join(clobbers)

    return f'asm volatile("{escape(final_template)}" : {outs_str} : {ins_str} : {clob_str});'
