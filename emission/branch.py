from __future__ import annotations

from cudagen.types import CudaBranch


def emit_branch_string(branch: CudaBranch) -> str:
    """
    Render a CudaBranch into a CUDA C string statement.
    Conditional branches become 'if (cond != 0) goto label;'
    Unconditional branches become 'goto label;'
    """
    label = branch.target.name
    if branch.cond is None:
        return f"goto {label};"
    return f"if ({branch.cond.name} != 0) goto {label};"
