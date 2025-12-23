from __future__ import annotations

from typing import Mapping

import ptx

from cudagen.render_inst import emit_inline_asm
from cudagen.types import Var, Expr, AddressOf


def emit_inline_asm_string(
    instr: ptx.Instruction, regmap: Mapping[ptx.Register, Var]
) -> str:
    """
    Emit a CUDA asm volatile string for a PTX instruction.
    """
    inline = emit_inline_asm(instr, regmap)

    def escape(s: str) -> str:
        return s.replace("\\", "\\\\").replace('"', '\\"')

    placeholder_for_expr = []
    for idx, expr in enumerate(inline.arguments + inline.outputs):
        if isinstance(expr, (Var, AddressOf)):
            placeholder_for_expr.append((expr, f"%{idx}"))
    placeholder_for_expr_dict = {}
    for expr, ph in placeholder_for_expr:
        placeholder_for_expr_dict[id(expr)] = ph

    pred_vars = [
        var
        for var, _ in placeholder_for_expr
        if isinstance(var, Var) and var.represents_predicate
    ]

    final_template = inline.template
    pre_lines: list[str] = []
    post_lines: list[str] = []

    # Only apply predicate bridging if any predicate vars are present
    if pred_vars:
        pred_temps = {var: f"ptmp{idx}" for idx, var in enumerate(pred_vars)}

        main_template = inline.template
        for var, temp in pred_temps.items():
            ph = placeholder_for_expr_dict.get(id(var))
            if ph:
                main_template = main_template.replace(ph, f"%{temp}")

        pre_lines.append(
            ".reg .pred " + ", ".join(f"%{t}" for t in pred_temps.values()) + ";"
        )
        for var, temp in pred_temps.items():
            ph = placeholder_for_expr_dict[id(var)]
            pre_lines.append(f"setp.ne.u32 %{temp}, {ph}, 0;")

        for out_var in inline.outputs:
            if out_var in pred_temps:
                temp = pred_temps[out_var]
                ph_out = placeholder_for_expr_dict[id(out_var)]
                post_lines.append(f"selp.u32 {ph_out}, 1, 0, %{temp};")

        template_parts = ["{"] + pre_lines + [main_template] + post_lines + ["}"]
        final_template = " ".join(template_parts)

    def constraint_for(expr: Expr) -> str:
        if isinstance(expr, Var):
            if expr.is_float:
                if expr.bitwidth == 64:
                    return "d"
                return "f"
            if expr.bitwidth == 64:
                return "l"
            if expr.bitwidth == 32:
                return "r"
            if expr.bitwidth == 16:
                return "h"
            if expr.bitwidth == 8:
                return "b"
            return "r"
        if isinstance(expr, AddressOf):
            # Always take the address as a 64-bit input; downcast handled in-template if needed.
            return "l"
        return "r"

    outputs = []
    for out_var in inline.outputs:
        c = constraint_for(out_var)
        outputs.append(f'"+{c}"({out_var.name})')

    inputs = []
    addr_prep: list[str] = []
    for arg in inline.arguments:
        if isinstance(arg, Var):
            if arg in inline.outputs:
                continue
            c = constraint_for(arg)
            inputs.append(f'"{c}"({arg.name})')
        elif isinstance(arg, AddressOf):
            bw = arg.bitwidth or 64
            if bw != 64:
                # only support shared for downcast path for now
                if arg.symbol.decl.memory_type != ptx.MemoryType.Shared:
                    raise ValueError(
                        "AddressOf downcast only supported for shared memory symbols"
                    )
                tmp64 = f"ptr64_{len(addr_prep)}"
                tmpN = f"ptr{bw}_{len(addr_prep)}"
                addr_prep.append(f".reg .u64 %{tmp64};")
                addr_prep.append(f".reg .u{bw} %{tmpN};")
                ph = placeholder_for_expr_dict[id(arg)]
                addr_prep.append(f"cvta.shared.u64 %{tmp64}, {ph};")
                addr_prep.append(f"cvt.u{bw}.u64 %{tmpN}, %{tmp64};")
                final_template = final_template.replace(ph, f"%{tmpN}")
                c = constraint_for(arg)
                inputs.append(f'"{c}"(&{arg.symbol.decl.name})')
            else:
                c = constraint_for(arg)
                inputs.append(f'"{c}"(&{arg.symbol.decl.name})')

    clobbers = ['"memory"'] if inline.clobbers_memory else []

    outs_str = ", ".join(outputs)
    ins_str = ", ".join(inputs)
    clob_str = ", ".join(clobbers)

    if addr_prep:
        final_template = "{ " + " ".join(addr_prep) + " " + final_template + " }"

    return f'asm volatile("{escape(final_template)}" : {outs_str} : {ins_str} : {clob_str});'
