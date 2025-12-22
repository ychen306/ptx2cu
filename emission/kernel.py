import ptx

from .param import get_type_decl_for_param


def declare_kernel(entry: ptx.EntryDirective, kernel_name: str):
    """
    Declare a CUDA kernel for the given EntryDirective.
    Returns (struct_defs, kernel_decl) where struct_defs is a list of
    helper struct definitions (for array params) and kernel_decl is the
    function signature string.
    """
    struct_defs: list[str] = []
    arg_parts: list[str] = []
    for param in entry.params:
        struct_def, type_name = get_type_decl_for_param(param)
        if struct_def:
            struct_defs.append(struct_def)
        arg_parts.append(f"{type_name} {param.name}")

    kernel_decl = f"__global__ void {kernel_name}(" + ", ".join(arg_parts) + ")"
    return struct_defs, kernel_decl
