from __future__ import annotations

from emission.inst import emit_inline_asm_ir
from emission.param import emit_load
from emission.branch import emit_branch_string
from emission.local import declare_local
from cudagen.types import CudaKernel, InlineAsm, Load, CudaBranch, CudaLabel


def emit_kernel(kernel: CudaKernel) -> str:
    """
    Emit a CUDA kernel definition string from a CudaKernel IR.
    """
    # Signature
    arg_parts = []
    for var, decl in kernel.arguments:
        # use decl to get type name
        from emission.param import get_type_decl_for_param  # lazy import to avoid cycle

        struct_def, type_name = get_type_decl_for_param(decl)
        arg_parts.append(f"{type_name} {var.name}")
    signature = f"__global__ void {kernel.name}(" + ", ".join(arg_parts) + ")"

    lines = [signature, "{"]
    indent = "  "
    # local declarations
    for v in kernel.var_decls:
        lines.append(indent + declare_local(v))

    for item in kernel.body:
        if isinstance(item, InlineAsm):
            lines.append(indent + emit_inline_asm_ir(item))
        elif isinstance(item, Load):
            lines.append(indent + emit_load(item))
        elif isinstance(item, CudaBranch):
            lines.append(indent + emit_branch_string(item))
        elif isinstance(item, CudaLabel):
            lines.append(f"{item.name}:")
        else:
            raise ValueError(f"Unsupported kernel item: {type(item).__name__}")

    lines.append("}")
    return "\n".join(lines)
