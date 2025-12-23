from __future__ import annotations

from abc import ABC
from dataclasses import dataclass
from typing import Optional


class MemoryType:
    Param = 0
    Global = 1
    Shared = 2


class Statement(ABC):
    """Base class for top-level statements."""


@dataclass
class MemoryDecl(ABC):
    alignment: Optional[int]
    datatype: str
    name: str
    num_elements: int
    memory_type: MemoryType


@dataclass
class RegisterDecl:
    datatype: str
    prefix: str
    num_regs: int


@dataclass(frozen=True)
class Register:
    prefix: str
    idx: Optional[int] = None


class BlockItem(ABC):
    """Base class for items inside a scoped block."""


@dataclass
class InstructionBase(BlockItem):
    predicate: Optional[Register]


class Operand(ABC):
    """Base class for an instruction operand"""


@dataclass
class Immediate(Operand):
    value: int


@dataclass
class Vector(Operand):
    values: list[Register]


@dataclass
class ParamRef:
    name: str


@dataclass
class MemoryRef(Operand):
    base: Register | ParamRef
    offset: int


@dataclass(frozen=True)
class MemorySymbol(Operand):
    decl: MemoryDecl


@dataclass
class Instruction(InstructionBase):
    opcode: str
    operands: list[Operand]


@dataclass
class Branch(InstructionBase):
    is_uniform: bool
    target: "Label"


@dataclass
class Label(BlockItem):
    name: str


@dataclass
class ScopedBlock(BlockItem):
    registers: list[RegisterDecl]
    body: list[BlockItem]


@dataclass
class Opaque(Statement):
    content: str


@dataclass
class Module:
    statements: list[Statement]


@dataclass
class EntryDirective(Statement):
    name: str
    params: list[MemoryDecl]
    directives: list[Opaque]
    body: ScopedBlock
