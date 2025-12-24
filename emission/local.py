from __future__ import annotations

from cudagen.types import Var


def ctype_for_var(var: Var) -> str:
    ty = var.get_type()
    if ty is None:
        return "unsigned int"
    if ty.represents_predicate:
        return "unsigned int"
    if ty.is_float:
        if ty.bitwidth == 64:
            return "double"
        if ty.bitwidth == 16:
            return "__half"
        return "float"
    if ty.bitwidth == 64:
        return "unsigned long long"
    if ty.bitwidth == 32:
        return "unsigned int"
    if ty.bitwidth == 16:
        return "unsigned short"
    if ty.bitwidth == 8:
        return "unsigned char"
    return "unsigned int"


def declare_local(var: Var) -> str:
    """
    Declare a local C variable for the given Var.
    """
    ctype = ctype_for_var(var)
    return f"{ctype} {var.name};"
