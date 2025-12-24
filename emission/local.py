from __future__ import annotations

from cudagen.types import Var


def ctype_for_var(var: Var) -> str:
    if var.ty.represents_predicate:
        return "unsigned int"
    if var.ty.is_float:
        if var.ty.bitwidth == 64:
            return "double"
        if var.ty.bitwidth == 16:
            return "__half"
        return "float"
    if var.ty.bitwidth == 64:
        return "unsigned long long"
    if var.ty.bitwidth == 32:
        return "unsigned int"
    if var.ty.bitwidth == 16:
        return "unsigned short"
    if var.ty.bitwidth == 8:
        return "unsigned char"
    return "unsigned int"


def declare_local(var: Var) -> str:
    """
    Declare a local C variable for the given Var.
    """
    ctype = ctype_for_var(var)
    return f"{ctype} {var.name};"
