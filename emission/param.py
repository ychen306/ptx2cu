from __future__ import annotations

import ptx

from cudagen.datatype import ctype_for_datatype, sizeof_datatype
from cudagen.types import Load, MemoryDecl


def get_type_decl_for_param(param: MemoryDecl) -> tuple[str | None, str]:
    """
    Given a param MemoryDecl, return (struct_def, type_name).
    struct_def is None for scalars; otherwise a struct definition string.
    type_name is the type to use in a function signature.
    """
    if param.memory_type != ptx.MemoryType.Param:
        raise ValueError("Expected a param MemoryDecl")

    ctype = ctype_for_datatype(param.datatype)
    if param.num_elements == 1:
        return None, ctype

    struct_name = f"Param_{param.datatype}_x_{param.num_elements}"
    struct_def = f"struct {struct_name} {{ {ctype} buf[{param.num_elements}]; }};"
    return struct_def, struct_name


def emit_load(load: Load) -> str:
    """
    Emit C code for a Load IR node as an assignment string.
    """
    elem_size = load.ty.bitwidth // 8
    if load.offset % elem_size != 0:
        raise ValueError("Unaligned load offset")
    idx = load.offset // elem_size

    if load.ty.is_float:
        ctype = (
            "double"
            if load.ty.bitwidth == 64
            else ("float" if load.ty.bitwidth == 32 else "__half")
        )
    else:
        if load.ty.bitwidth == 64:
            ctype = "unsigned long long"
        elif load.ty.bitwidth == 32:
            ctype = "unsigned int"
        elif load.ty.bitwidth == 16:
            ctype = "unsigned short"
        elif load.ty.bitwidth == 8:
            ctype = "unsigned char"
        else:
            ctype = "unsigned int"

    base_ptr = f"reinterpret_cast<{ctype}*>({('&' + load.src.name) if load.is_param else load.src.name})"
    rhs = f"{base_ptr}[{idx}]"
    return f"{load.dst.name} = {rhs};"
