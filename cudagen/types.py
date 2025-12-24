from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, List
from abc import ABC
from enum import Enum, auto
import ptx


# cuda expression
class Expr(ABC):
    def get_type(self) -> Optional["CudaType"]:
        return None


@dataclass(frozen=True)
class CudaType:
    bitwidth: int
    is_float: bool
    represents_predicate: bool = False


@dataclass(frozen=True)
class Var(Expr):
    name: str
    ty: CudaType

    def get_type(self) -> Optional[CudaType]:
        return self.ty

class BinaryOpcode(Enum):
    # integer 
    Add = auto()
    Sub = auto()
    SDiv = auto()
    UDiv = auto()

    # float
    FAdd = auto()
    FMul = auto()
    FDiv = auto()

    # bitwise
    Or = auto()
    And = auto()
    Shl = auto()
    LShr = auto()
    AShr = auto()
    Xor = auto()

@dataclass
class BinaryOperator(Expr):
    opcode : BinaryOpcode
    operand_a  : Expr
    operand_b : Expr

    def get_type(self):
        assert self.operand_a.get_type() == self.operand_b.get_type()
        return self.operand_a.get_type()


class KernelItem(ABC):
    pass


@dataclass
class AddressOf(Expr):
    symbol: ptx.MemorySymbol
    bitwidth: Optional[int]

    def get_type(self) -> Optional[CudaType]:
        # AddressOf itself has no numeric value type; caller should treat as None.
        return None


@dataclass
class InlineAsm(KernelItem):
    template: str
    arguments: list[Expr]
    outputs: list[Var]
    clobbers_memory: bool = False

@dataclass
class Assignment(KernelItem):
    lhs : Var
    rhs : Expr


@dataclass
class Load(KernelItem):
    """
    Models: dest = src[offset]
    assuming everything is byte-typed
    """

    ty: CudaType
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
