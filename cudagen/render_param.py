
import ptx

from .types import MemoryDecl


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
