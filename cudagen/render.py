from __future__ import annotations

from collections import ChainMap

import ptx

from .types import Var


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
