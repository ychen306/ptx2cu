from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import ptx


@dataclass(frozen=True)
class Var:
    name: str


@dataclass
class RegisterInfo:
    decl: ptx.RegisterDecl
    c_var: Var


@dataclass
class InlineAsm:
    template: str
    arguments: list[Var]
    outputs: list[Var]
    constraints: dict[Var, str]
    clobbers_memory: bool = False


# Re-export MemoryDecl for convenience
MemoryDecl = ptx.MemoryDecl
