import ptx


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
