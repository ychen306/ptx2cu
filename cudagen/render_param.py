
from __future__ import annotations

import ptx

from .datatype import type_info_for_datatype
from .types import Load, MemoryDecl, Var


def emit_ld_param(
    instr: ptx.Instruction,
    regmap: dict[ptx.Register, Var],
    param_map: dict[str, MemoryDecl],
) -> Load:
    """
    Emit C code to load a param into a mapped variable using a PTX ld.param instruction.
    """
    if not instr.opcode.startswith("ld.param"):
        raise ValueError("emit_ld_param expects an ld.param instruction")
    if len(instr.operands) < 2:
        raise ValueError("ld.param instruction requires at least dest and source operands")

    dest = instr.operands[0]
    if not isinstance(dest, ptx.Register):
        raise ValueError("ld.param destination must be a register")

    src = instr.operands[1]
    if not isinstance(src, ptx.MemoryRef) or not isinstance(src.base, ptx.ParamRef):
        raise ValueError("ld.param source must be a param memory reference")

    decl = param_map.get(src.base.name)
    if decl is None:
        raise ValueError(f"Unknown param {src.base.name}")
    if decl.memory_type != ptx.MemoryType.Param:
        raise ValueError("emit_ld_param only supports param memory")

    ctype, bitwidth, is_float = type_info_for_datatype(decl.datatype)
    size = bitwidth // 8
    offset = src.offset or 0
    if offset % size != 0:
        raise ValueError(f"Offset {offset} not aligned to {size}-byte element")
    index = offset // size

    dest_var = regmap.get(dest)
    if dest_var is None:
        raise ValueError(f"Missing mapping for dest register {dest}")
    lhs = dest_var

    src_var = Var(name=decl.name, bitwidth=bitwidth, is_float=is_float, represents_predicate=False)

    return Load(bitwidth=bitwidth, is_float=is_float, dst=lhs, src=src_var, offset=offset)
