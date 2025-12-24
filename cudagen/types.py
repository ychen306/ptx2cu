from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, List
from abc import ABC
from enum import IntEnum
import ptx


# cuda expression
class Expr(ABC):
    pass


@dataclass(frozen=True)
class Var(Expr):
    name: str
    bitwidth: int
    is_float: bool
    represents_predicate: bool = False

class BinaryOpcode(IntEnum):
    pass


@dataclass
class BinaryOperator(Expr):
    opcode : BinaryOpcode


class KernelItem(ABC):
    pass


@dataclass
class AddressOf(Expr):
    symbol: ptx.MemorySymbol
    bitwidth: Optional[int]


@dataclass
class InlineAsm(KernelItem):
    template: str
    arguments: list[Expr]
    outputs: list[Var]
    clobbers_memory: bool = False


@dataclass
class Load(KernelItem):
    """
    Models: dest = src[offset]
    assuming everything is byte-typed
    """

    bitwidth: int
    is_float: bool
    dst: Var
    src: Var
    offset: int


@dataclass
class CudaBranch(KernelItem):
    cond: Optional[Var]
    target: CudaLabel


@dataclass
class CudaLabel(KernelItem):
    name: str


@dataclass
class Return(KernelItem):
    pass


@dataclass
class CudaKernel:
    name: str
    arguments: list[Tuple[Var, ptx.MemoryDecl]]
    var_decls: list[Var]
    body: list[KernelItem]


@dataclass
class CudaModule:
    global_vars: list[ptx.MemoryDecl]
    kernels: list[CudaKernel]


# Re-export MemoryDecl for convenience
MemoryDecl = ptx.MemoryDecl
