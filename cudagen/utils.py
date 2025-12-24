from typing import Mapping

import ptx

from .types import Var, Expr, AddressOf


def collect_registers(op: ptx.Operand) -> list[ptx.Register]:
    regs: list[ptx.Register] = []
    if isinstance(op, ptx.Register):
        regs.append(op)
    elif isinstance(op, ptx.Vector):
        regs.extend(op.values)
    elif isinstance(op, ptx.MemoryRef) and isinstance(op.base, ptx.Register):
        regs.append(op.base)
    return regs


def render_operand_with_index(
    operand: ptx.Operand | ptx.ParamRef,
    regmap: Mapping[ptx.Register, Var],
    args: list[Expr],
    idx: int,
) -> tuple[str, int]:
    if isinstance(operand, ptx.Register):
        var = regmap.get(operand)
        if var is None:
            raise ValueError(f"Missing mapping for register {operand}")
        args.append(var)
        return f"%{len(args)-1}", idx + 1
    if isinstance(operand, ptx.Vector):
        rendered = []
        cur = idx
        for reg in operand.values:
            piece, cur = render_operand_with_index(reg, regmap, args, cur)
            rendered.append(piece)
        return "{" + ", ".join(rendered) + "}", cur
    if isinstance(operand, ptx.Immediate):
        return str(operand.value), idx
    if isinstance(operand, ptx.MemoryRef):
        base_rendered, next_idx = render_operand_with_index(
            operand.base, regmap, args, idx
        )
        if operand.offset:
            return f"[{base_rendered}+{operand.offset}]", next_idx
        return f"[{base_rendered}]", next_idx
    if isinstance(operand, ptx.MemorySymbol):
        addr = AddressOf(symbol=operand, bitwidth=None)
        args.append(addr)
        return f"%{len(args)-1}", idx + 1
    if isinstance(operand, ptx.ParamRef):
        return operand.name, idx
    raise ValueError(f"Unsupported operand type: {type(operand).__name__}")
