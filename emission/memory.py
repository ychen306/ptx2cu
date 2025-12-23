from __future__ import annotations

from cudagen.datatype import ctype_for_datatype
from cudagen.types import MemoryDecl
import ptx


def declare_memory(mem: MemoryDecl) -> str:
    """
    Declare a non-param memory object as a CUDA-side array.
    Handles global/shared memory; params are rejected.
    """
    if mem.memory_type == ptx.MemoryType.Param:
        raise ValueError("declare_memory does not handle param memory")

    if mem.memory_type == ptx.MemoryType.Shared:
        scope_kw = "__shared__"
    elif mem.memory_type == ptx.MemoryType.Global:
        scope_kw = "__device__"
    else:
        scope_kw = "__device__"

    extern_kw = "extern " if mem.num_elements == 0 else ""
    align_kw = f"__align__({mem.alignment}) " if mem.alignment else ""
    ctype = ctype_for_datatype(mem.datatype)

    size_part = "[]" if mem.num_elements == 0 else f"[{mem.num_elements}]"

    return f"{extern_kw}{scope_kw} {align_kw}{ctype} {mem.name}{size_part};"
