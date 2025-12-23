from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple
from abc import ABC
import ptx


@dataclass(frozen=True)
class Var:
    name: str
    bitwidth: int
    is_float: bool
    represents_predicate: bool = False

class KernelItem(ABC):
    pass


@dataclass
class InlineAsm(KernelItem):
    template: str
    arguments: list[Var]
    outputs: list[Var]
    clobbers_memory: bool = False

@dataclass
class Load(KernelItem):
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
class CudaBranch(KernelItem):
    cond : Optional[Var]
    target : CudaLabel

@dataclass
class CudaLabel(KernelItem):
    name : str

@dataclass
@dataclass
class CudaKernel:
    arguments: list[Tuple[Var, ptx.MemoryDecl]]
    # this include kernel arguments (i.e., params)
    var_decls: list[Var]
    body: list[KernelItem]

# Re-export MemoryDecl for convenience
MemoryDecl = ptx.MemoryDecl
