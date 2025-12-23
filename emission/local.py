from __future__ import annotations

from cudagen.types import Var


def ctype_for_var(var: Var) -> str:
    if var.represents_predicate:
        return "unsigned int"
    if var.is_float:
        if var.bitwidth == 64:
            return "double"
        if var.bitwidth == 16:
            return "__half"
        return "float"
    if var.bitwidth == 64:
        return "unsigned long long"
    if var.bitwidth == 32:
        return "unsigned int"
    if var.bitwidth == 16:
        return "unsigned short"
    if var.bitwidth == 8:
        return "unsigned char"
    return "unsigned int"


def declare_local(var: Var) -> str:
    """
    Declare a local C variable for the given Var.
    """
    ctype = ctype_for_var(var)
    return f"{ctype} {var.name};"
