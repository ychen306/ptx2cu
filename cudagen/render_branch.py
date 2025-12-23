from __future__ import annotations

from typing import Mapping

import ptx

from .types import CudaBranch, CudaLabel, Var


def emit_branch(branch: ptx.Branch, regmap: Mapping[ptx.Register, Var]) -> CudaBranch:
    """
    Lower a PTX Branch into a CudaBranch. Uniformity is ignored in this IR.
    """
    cond: Var | None = None
    if branch.predicate:
        var = regmap.get(branch.predicate)
        if var is None:
            raise ValueError(
                f"Missing mapping for predicate register {branch.predicate}"
            )
        cond = var

    label = CudaLabel(name=branch.target.name)
    return CudaBranch(cond=cond, target=label)
