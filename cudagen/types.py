from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, List
from abc import ABC
from enum import Enum, auto
import ptx


# cuda expression
class Expr(ABC):
    def get_type(self) -> Optional["CudaType | CudaPointerType"]:
        return None


class CudaTypeId(Enum):
    Signed = "signed"
    Unsigned = "unsigned"
    Float = "float"
    Pointer = "pointer"


@dataclass(frozen=True)
class CudaType:
    bitwidth: int
    type_id: CudaTypeId
    represents_predicate: bool = False

    @property
    def is_float(self) -> bool:
        return self.type_id == CudaTypeId.Float

    @property
    def is_signed(self) -> bool:
        return self.type_id == CudaTypeId.Signed

    @property
    def is_pointer(self) -> bool:
        return self.type_id == CudaTypeId.Pointer


@dataclass(frozen=True)
class CudaPointerType:
    elem: CudaType
    bitwidth: int
    type_id: CudaTypeId = CudaTypeId.Pointer
    represents_predicate: bool = False

    def __init__(self, elem: CudaType, bitwidth: int = 64) -> None:
        object.__setattr__(self, "elem", elem)
        object.__setattr__(self, "bitwidth", bitwidth)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, CudaPointerType):
            return False
        return self.bitwidth == other.bitwidth and self.elem == other.elem

    def __hash__(self) -> int:
        return hash((self.bitwidth, self.type_id, self.elem))

    @property
    def is_float(self) -> bool:
        return False

    @property
    def is_signed(self) -> bool:
        return False

    @property
    def is_pointer(self) -> bool:
        return True


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
    Mul = auto()
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
    opcode: BinaryOpcode
    operand_a: Expr
    operand_b: Expr

    def get_type(self):
        assert self.operand_a.get_type() == self.operand_b.get_type()
        return self.operand_a.get_type()


@dataclass
class ConstantInt(Expr):
    value: int
    ty: CudaType

    def get_type(self) -> Optional[CudaType]:
        return self.ty


@dataclass
class BitCast(Expr):
    new_type: CudaType | CudaPointerType
    operand: Expr

    def get_type(self) -> Optional[CudaType | CudaPointerType]:
        return self.new_type


@dataclass
class SignExt(Expr):
    operand: Expr
    new_type: CudaType

    def get_type(self) -> Optional[CudaType]:
        return self.new_type


@dataclass
class ZeroExt(Expr):
    operand: Expr
    new_type: CudaType

    def get_type(self) -> Optional[CudaType]:
        return self.new_type


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
    lhs: Var
    rhs: Expr


@dataclass
class Load(KernelItem):
    """
    Models: dest = src[offset]
    assuming everything is byte-typed
    """

    ty: CudaType
    dst: Var
    src: Expr
    offset: int
    is_param: bool = False


@dataclass
class Store(KernelItem):
    """pointer[offset] = value"""
    pointer: Expr
    offset: int
    value: Expr


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
