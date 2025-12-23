from __future__ import annotations

from collections import ChainMap
from typing import List

import ptx

from .types import CudaKernel, CudaLabel, Var
from .render_branch import emit_branch
from .render_inst import emit_inline_asm
from .render_param import emit_ld_param
from .datatype import type_info_for_datatype


class CudaGen:
    """
    Driver for lowering PTX IR into cudagen IR.

    reg_map holds the current mapping from PTX registers to CUDA Vars.
    It is a ChainMap to support scoped shadowing during traversal.
    """

    def __init__(self):
        self.reg_map: ChainMap[ptx.Register, Var] = ChainMap()
        self._name_counters: dict[str, int] = {}
        self.var_decls: list[Var] = []
        self.param_map: dict[str, ptx.MemoryDecl] = {}
        self._init_special_regs()

    def _init_special_regs(self):
        """
        Seed the root reg_map with PTX special registers mapped to CUDA built-ins.
        These are inputs-only and are not added to var_decls or name counters.
        """
        special = {
            ptx.Register(prefix="ctaid.x", idx=None): Var("blockIdx.x", 32, False),
            ptx.Register(prefix="ctaid.y", idx=None): Var("blockIdx.y", 32, False),
            ptx.Register(prefix="ctaid.z", idx=None): Var("blockIdx.z", 32, False),
            ptx.Register(prefix="tid.x", idx=None): Var("threadIdx.x", 32, False),
            ptx.Register(prefix="tid.y", idx=None): Var("threadIdx.y", 32, False),
            ptx.Register(prefix="tid.z", idx=None): Var("threadIdx.z", 32, False),
        }
        self.reg_map = ChainMap(special)

    def _alloc_var(self, decl: ptx.RegisterDecl, reg: ptx.Register) -> Var:
        """
        Allocate a Var for the given PTX register with a unique C name.
        """
        prefix = reg.prefix
        idx = self._name_counters.get(prefix, 0)
        self._name_counters[prefix] = idx + 1
        name = f"{prefix}{idx}"

        # Determine bitwidth and type flags
        dt = decl.datatype
        if dt == "pred":
            bitwidth = 32
            is_float = False
            represents_predicate = True
        elif dt.startswith("f"):
            # float types
            if dt.startswith("f16"):
                bitwidth = 16
            elif dt.startswith("f64"):
                bitwidth = 64
            else:
                bitwidth = 32
            is_float = True
            represents_predicate = False
        else:
            # integer/bit types
            if dt.endswith("8"):
                bitwidth = 8
            elif dt.endswith("16"):
                bitwidth = 16
            elif dt.endswith("64"):
                bitwidth = 64
            else:
                bitwidth = 32
            is_float = False
            represents_predicate = False

        var = Var(name=name, bitwidth=bitwidth, is_float=is_float, represents_predicate=represents_predicate)
        self.var_decls.append(var)
        return var

    def _walk_block(self, block: ptx.ScopedBlock, items: list) -> None:
        """
        Recursively walk a ScopedBlock and append lowered KernelItems to items.
        """
        self.enter_scope(block)
        try:
            for node in block.body:
                if isinstance(node, ptx.Label):
                    items.append(CudaLabel(name=node.name))
                elif isinstance(node, ptx.Branch):
                    items.append(emit_branch(node, self.reg_map))
                elif isinstance(node, ptx.Instruction):
                    if node.opcode.startswith("ld.param"):
                        items.append(emit_ld_param(node, self.reg_map, self.param_map))
                    else:
                        items.append(emit_inline_asm(node, self.reg_map))
                elif isinstance(node, ptx.ScopedBlock):
                    self._walk_block(node, items)
                # ignore other directive/opaque nodes for now
        finally:
            self.exit_scope()

    def run(self, entry: ptx.EntryDirective) -> "CudaKernel":
        """
        Lower a PTX EntryDirective into a CudaKernel.
        """
        # reset state
        self._init_special_regs()
        self._name_counters = {}
        self.var_decls = []
        self.param_map = {p.name: p for p in entry.params}

        arguments: list[tuple[Var, ptx.MemoryDecl]] = []
        for p in entry.params:
            _, bitwidth, is_float = type_info_for_datatype(p.datatype)
            arg_var = Var(name=p.name, bitwidth=bitwidth, is_float=is_float, represents_predicate=False)
            arguments.append((arg_var, p))

        body_items: list = []
        self._walk_block(entry.body, body_items)

        return CudaKernel(arguments=arguments, var_decls=self.var_decls, body=body_items)

    def enter_scope(self, block: ptx.ScopedBlock) -> None:
        """
        Push a new register scope, expanding register declarations into Var mappings.
        """
        scope_map: dict[ptx.Register, Var] = {}
        for decl in block.registers:
            for i in range(decl.num_regs):
                reg_idx = i if decl.num_regs > 1 else (None if decl.prefix == "p" else i)
                reg = ptx.Register(prefix=decl.prefix, idx=reg_idx)
                scope_map[reg] = self._alloc_var(decl, reg)

        self.reg_map = self.reg_map.new_child(scope_map)

    def exit_scope(self) -> None:
        """
        Pop the current register scope.
        """
        if self.reg_map.parents is None:
            raise RuntimeError("Cannot exit root scope")
        self.reg_map = self.reg_map.parents
