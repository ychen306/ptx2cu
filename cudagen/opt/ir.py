from __future__ import annotations

from dataclasses import dataclass
from typing import List, Union

from .. import types as cg_types


# IR for optimized control-flow representation.


# Reuse the existing low-level statements.
OptInstruction = Union[cg_types.Assignment, cg_types.Load, cg_types.Store]


@dataclass
class OptTerminator:
    """Base class for block terminators."""


@dataclass
class Return(OptTerminator):
    """Terminator that returns from the kernel (no value)."""


@dataclass
class OptBranch(OptTerminator):
    target: "OptBasicBlock"


@dataclass
class OptCondBranch(OptTerminator):
    condition: cg_types.Expr
    true_target: "OptBasicBlock"
    false_target: "OptBasicBlock"


@dataclass
class OptBasicBlock:
    """A basic block: instructions followed by a mandatory terminator."""

    id: str
    instructions: List[OptInstruction]
    terminator: OptTerminator


@dataclass
class OptKernel:
    """Kernel with CFG entry plus metadata matching CudaKernel signature."""

    name: str
    arguments: List[cg_types.MemoryDecl]
    var_decls: List[cg_types.Var]
    entry: OptBasicBlock
import dataclasses import dataclass
from abc import ABC
