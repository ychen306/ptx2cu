from __future__ import annotations

from cudagen.datatype import ctype_for_datatype
from cudagen.types import MemoryDecl, Store, CudaPointerType, CudaType
import ptx
from emission.expr import emit_expr


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

    align_kw = f"__align__({mem.alignment}) " if mem.alignment else ""
    ctype = ctype_for_datatype(mem.datatype)

    size_part = "[]" if mem.num_elements == 0 else f"[{mem.num_elements}]"

    extern_kw = "extern " if mem.num_elements == 0 else ""
    return f'extern "C" {extern_kw}{scope_kw} {align_kw}{ctype} {mem.name}{size_part};'


def emit_store(store: Store) -> str:
    """
    Emit a C statement for a Store IR node.
    """
    ptr_ty = store.pointer.get_type()
    if not isinstance(ptr_ty, CudaPointerType):
        raise ValueError("Store requires a typed pointer")

    elem_size = ptr_ty.elem.bitwidth // 8
    if store.offset % elem_size != 0:
        raise ValueError("Unaligned store offset")
    index = store.offset // elem_size

    ptr_expr = emit_expr(store.pointer)
    val_expr = emit_expr(store.value)
    return f"{ptr_expr}[{index}] = {val_expr};"
