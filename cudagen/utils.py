import ptx

from .types import RegisterInfo, Var
from .constraints import get_register_constraint


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
            piece, cur = render_operand_with_index(reg, regmap, constraints, args, cur)
            rendered.append(piece)
        return "{" + ", ".join(rendered) + "}", cur
    if isinstance(operand, ptx.Immediate):
        return str(operand.value), idx
    if isinstance(operand, ptx.MemoryRef):
        base_rendered, next_idx = render_operand_with_index(operand.base, regmap, constraints, args, idx)
        if operand.offset:
            return f"[{base_rendered}+{operand.offset}]", next_idx
        return f"[{base_rendered}]", next_idx
    if isinstance(operand, ptx.ParamRef):
        return operand.name, idx
    raise ValueError(f"Unsupported operand type: {type(operand).__name__}")
