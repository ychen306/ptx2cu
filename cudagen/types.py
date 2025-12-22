from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import ptx


@dataclass(frozen=True)
class Var:
    name: str
    bitwidth: int
    is_float: bool
    represents_predicate: bool = False


@dataclass
class RegisterInfo:
    decl: ptx.RegisterDecl
    c_var: Var


@dataclass
class InlineAsm:
    template: str
    arguments: list[Var]
    outputs: list[Var]
    clobbers_memory: bool = False

@dataclass
class Load:
    '''
    Models: dest = src[offset]
    assuming everything is byte-typed
    '''
    bitwidth: int
    is_float: bool
    dst: Var
    src: Var
    offset: int

@dataclass
class CudaBranch:
    cond : Optional[Var]
    target : CudaLabel

@dataclass
class CudaLabel:
    name : str


# Re-export MemoryDecl for convenience
MemoryDecl = ptx.MemoryDecl
