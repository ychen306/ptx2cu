from __future__ import annotations

import ptx

from .types import CudaBranch, CudaLabel, RegisterInfo, Var


def emit_branch(branch: ptx.Branch, regmap: dict[ptx.Register, RegisterInfo]) -> CudaBranch:
    """
    Lower a PTX Branch into a CudaBranch. Uniformity is ignored in this IR.
    """
    cond: Var | None = None
    if branch.predicate:
        info = regmap.get(branch.predicate)
        if info is None:
            raise ValueError(f"Missing mapping for predicate register {branch.predicate}")
        cond = info.c_var

    label = CudaLabel(name=branch.target.name)
    return CudaBranch(cond=cond, target=label)
