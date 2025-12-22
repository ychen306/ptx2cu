
from __future__ import annotations

import ptx

from .types import Load, MemoryDecl, RegisterInfo, Var


def _ctype_for_datatype(datatype: str) -> str:
    dt = datatype
    if dt.startswith(("u8", "b8")):
        return "unsigned char"
    if dt.startswith("s8"):
        return "signed char"
    if dt.startswith(("u16", "b16")):
        return "unsigned short"
    if dt.startswith("s16"):
        return "short"
    if dt.startswith("f16"):
        return "__half"
    if dt.startswith(("u32", "b32")):
        return "unsigned int"
    if dt.startswith("s32"):
        return "int"
    if dt.startswith("f32"):
        return "float"
    if dt.startswith(("u64", "b64")):
        return "unsigned long long"
    if dt.startswith("s64"):
        return "long long"
    if dt.startswith("f64"):
        return "double"
    return "unsigned int"


def _type_info_for_datatype(datatype: str) -> tuple[str, int, bool]:
    """
    Return (ctype, bitwidth, is_float) for a PTX datatype string.
    """
    ctype = _ctype_for_datatype(datatype)
    size = _sizeof_datatype(datatype)
    bitwidth = size * 8
    is_float = datatype.startswith("f")
    return ctype, bitwidth, is_float


def _sizeof_datatype(datatype: str) -> int:
    if datatype.startswith(("u8", "b8", "s8")):
        return 1
    if datatype.startswith(("u16", "b16", "s16", "f16")):
        return 2
    if datatype.startswith(("u32", "b32", "s32", "f32")):
        return 4
    if datatype.startswith(("u64", "b64", "s64", "f64")):
        return 8
    return 4


def get_type_decl_for_param(param: MemoryDecl) -> tuple[str | None, str]:
    """
    Given a param MemoryDecl, return (struct_def, type_name).
    struct_def is None for scalars; otherwise a struct definition string.
    type_name is the type to use in a function signature.
    """
    if param.memory_type != ptx.MemoryType.Param:
        raise ValueError("Expected a param MemoryDecl")

    ctype = _ctype_for_datatype(param.datatype)
    if param.num_elements == 1:
        return None, ctype

    struct_name = f"Param_{param.datatype}_x_{param.num_elements}"
    struct_def = f"struct {struct_name} {{ {ctype} buf[{param.num_elements}]; }};"
    return struct_def, struct_name


def emit_ld_param(
    instr: ptx.Instruction,
    regmap: dict[ptx.Register, RegisterInfo],
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

    ctype, bitwidth, is_float = _type_info_for_datatype(decl.datatype)
    size = _sizeof_datatype(decl.datatype)
    offset = src.offset or 0
    if offset % size != 0:
        raise ValueError(f"Offset {offset} not aligned to {size}-byte element")
    index = offset // size

    dest_info = regmap.get(dest)
    if dest_info is None:
        raise ValueError(f"Missing mapping for dest register {dest}")
    lhs = dest_info.c_var

    src_var = Var(name=decl.name, bitwidth=bitwidth, is_float=is_float, represents_predicate=False)

    return Load(bitwidth=bitwidth, is_float=is_float, dst=lhs, src=src_var, offset=offset)


def emit_load(load: Load) -> str:
    """
    Emit C code for a Load IR node as an assignment string.
    """
    elem_size = load.bitwidth // 8
    if load.offset % elem_size != 0:
        raise ValueError("Unaligned load offset")
    idx = load.offset // elem_size

    if load.is_float:
        ctype = "double" if load.bitwidth == 64 else ("float" if load.bitwidth == 32 else "__half")
    else:
        if load.bitwidth == 64:
            ctype = "unsigned long long"
        elif load.bitwidth == 32:
            ctype = "unsigned int"
        elif load.bitwidth == 16:
            ctype = "unsigned short"
        elif load.bitwidth == 8:
            ctype = "unsigned char"
        else:
            ctype = "unsigned int"

    base_ptr = f"reinterpret_cast<{ctype}*>(&{load.src.name})"
    rhs = f"{base_ptr}[{idx}]"
    return f"{load.dst.name} = {rhs};"
