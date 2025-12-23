from __future__ import annotations

from typing import List, Set

from cudagen.types import CudaModule, CudaKernel
from emission.memory import declare_memory
from emission.kernel_body import emit_kernel
from emission.param import get_type_decl_for_param


def emit_cuda_module(module: CudaModule) -> str:
    """
    Emit a full CUDA translation unit string from a CudaModule.
    Always includes <cuda_fp16.h>.
    """
    lines: List[str] = ["#include <cuda_fp16.h>", ""]

    # Globals
    for mem in module.global_vars:
        lines.append(declare_memory(mem))

    if module.global_vars:
        lines.append("")

    # Collect kernels and struct defs
    struct_defs: List[str] = []
    seen_structs: Set[str] = set()
    kernel_strings: List[str] = []

    for kernel in module.kernels:
        # collect struct defs from arguments
        for _, decl in kernel.arguments:
            struct_def, type_name = get_type_decl_for_param(decl)
            if struct_def and struct_def not in seen_structs:
                seen_structs.add(struct_def)
                struct_defs.append(struct_def)
        kernel_strings.append(emit_kernel(kernel))

    for sd in struct_defs:
        lines.append(sd)
    if struct_defs:
        lines.append("")

    lines.extend(kernel_strings)

    return "\n".join(lines)
