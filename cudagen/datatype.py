from __future__ import annotations


def ctype_for_datatype(datatype: str) -> str:
    """
    Map a PTX datatype string (e.g., u32, f16) to a C/CUDA type name.
    """
    dt = datatype
    if dt.startswith(("u8", "b8")):
        return "unsigned char"
    if dt.startswith("s8"):
        return "signed char"
    if dt.startswith("b16"):
        return "__half"
    if dt.startswith("u16"):
        return "__half"
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
    if dt.startswith(("u64", "b64", "s64", "f64")):
        return (
            "double"
            if dt.startswith("f64")
            else ("long long" if dt.startswith("s64") else "unsigned long long")
        )
    return "unsigned int"


def sizeof_datatype(datatype: str) -> int:
    """
    Size in bytes for a PTX datatype string.
    """
    if datatype.startswith(("u8", "b8", "s8")):
        return 1
    if datatype.startswith(("u16", "b16", "s16", "f16")):
        return 2
    if datatype.startswith(("u32", "b32", "s32", "f32")):
        return 4
    if datatype.startswith(("u64", "b64", "s64", "f64")):
        return 8
    return 4


def type_info_for_datatype(datatype: str) -> tuple[str, int, bool]:
    """
    Return (ctype, bitwidth, is_float) for a PTX datatype string.
    """
    ctype = ctype_for_datatype(datatype)
    size = sizeof_datatype(datatype)
    bitwidth = size * 8
    is_float = datatype.startswith("f") or datatype.startswith("b16") or datatype.startswith("u16")
    return ctype, bitwidth, is_float
