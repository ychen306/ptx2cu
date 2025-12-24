import ptx
import pytest

from cudagen import CudaGen, Var
from cudagen.types import CudaType

t32 = CudaType(32, False)
t_pred = CudaType(32, False, True)


def test_enter_and_exit_scope_basic():
    gen = CudaGen()
    block = ptx.ScopedBlock(
        registers=[ptx.RegisterDecl(datatype="b32", prefix="r", num_regs=2)], body=[]
    )
    gen.enter_scope(block)
    r0 = ptx.Register(prefix="r", idx=0)
    r1 = ptx.Register(prefix="r", idx=1)
    assert gen.reg_map[r0] == Var("r0", t32)
    assert gen.reg_map[r1] == Var("r1", t32)
    assert gen.var_decls == [Var("r0", t32), Var("r1", t32)]
    gen.exit_scope()
    with pytest.raises(KeyError):
        _ = gen.reg_map[r0]
    # var_decls persists across scope exits
    assert gen.var_decls == [Var("r0", t32), Var("r1", t32)]


def test_nested_scopes_shadowing():
    gen = CudaGen()
    outer = ptx.ScopedBlock(
        registers=[ptx.RegisterDecl(datatype="b32", prefix="r", num_regs=1)], body=[]
    )
    inner = ptx.ScopedBlock(
        registers=[ptx.RegisterDecl(datatype="b32", prefix="r", num_regs=1)], body=[]
    )
    gen.enter_scope(outer)
    outer_reg = ptx.Register(prefix="r", idx=0)
    outer_var = gen.reg_map[outer_reg]
    gen.enter_scope(inner)
    inner_reg = ptx.Register(prefix="r", idx=0)
    inner_var = gen.reg_map[inner_reg]
    assert outer_var != inner_var  # unique names
    gen.exit_scope()
    assert gen.reg_map[outer_reg] == outer_var
    # Allocation order preserved and persisted
    assert gen.var_decls[0] == outer_var
    assert gen.var_decls[1] == inner_var


def test_predicate_registers_get_predicate_flag():
    gen = CudaGen()
    block = ptx.ScopedBlock(
        registers=[ptx.RegisterDecl(datatype="pred", prefix="p", num_regs=1)], body=[]
    )
    gen.enter_scope(block)
    preg = ptx.Register(prefix="p", idx=None)
    var = gen.reg_map[preg]
    assert var.ty.represents_predicate is True
    assert var.ty.bitwidth == 32
