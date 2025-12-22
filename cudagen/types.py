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


# Re-export MemoryDecl for convenience
MemoryDecl = ptx.MemoryDecl
