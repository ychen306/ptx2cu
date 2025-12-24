import ptx
from cudagen.render_inst import emit_assignment
from cudagen.types import (
    CudaType,
    Var,
    CudaTypeId,
    CudaPointerType,
    BitCast,
    Load,
    Store,
)
from emission.expr import emit_assignment_stmt
from emission.param import emit_load
from emission.memory import emit_store


t_i32 = CudaType(32, CudaTypeId.Unsigned)
t_f32 = CudaType(32, CudaTypeId.Float)


def test_emit_assignment_from_add_s32():
    regmap = {
        ptx.Register(prefix="r", idx=1): Var("r1", t_i32),
        ptx.Register(prefix="r", idx=2): Var("r2", t_i32),
        ptx.Register(prefix="r", idx=3): Var("r3", t_i32),
    }
    instr = ptx.Instruction(
        predicate=None,
        opcode="add.s32",
        operands=[
            ptx.Register(prefix="r", idx=1),
            ptx.Register(prefix="r", idx=2),
            ptx.Register(prefix="r", idx=3),
        ],
    )
    assignment = emit_assignment(instr, regmap)
    assert assignment is not None
    assert emit_assignment_stmt(assignment) == "r1 = (r2 + r3);"


def test_emit_assignment_from_add_f32_with_int_inputs_bitcasts():
    regmap = {
        ptx.Register(prefix="r", idx=1): Var("r1", t_i32),
        ptx.Register(prefix="r", idx=2): Var("r2", t_i32),
        ptx.Register(prefix="r", idx=3): Var("r3", t_i32),
    }
    instr = ptx.Instruction(
        predicate=None,
        opcode="add.f32",
        operands=[
            ptx.Register(prefix="r", idx=1),
            ptx.Register(prefix="r", idx=2),
            ptx.Register(prefix="r", idx=3),
        ],
    )
    assignment = emit_assignment(instr, regmap)
    assert assignment is not None
    # Inputs are bitcast to float, sum is float, then cast back to dest int type
    assert (
        emit_assignment_stmt(assignment)
        == "r1 = __float_as_uint((__uint_as_float(r2) + __uint_as_float(r3)));"
    )


def test_emit_load_from_ld_global_direct():
    # Simulate a Load coming from ld.global with matching pointer type
    ptr_ty = CudaPointerType(elem=t_i32)
    load = Load(
        ty=t_i32,
        dst=Var("r1", t_i32),
        src=Var("rd1", ptr_ty),
        offset=4,
        is_param=False,
    )
    assert emit_load(load) == "r1 = rd1[1];"


def test_emit_load_from_ld_global_with_bitcast():
    # Pointer elem type mismatched; src is a BitCast
    ptr_ty = CudaPointerType(elem=CudaType(32, CudaTypeId.Signed))
    load = Load(
        ty=t_i32,
        dst=Var("r1", t_i32),
        src=BitCast(new_type=ptr_ty, operand=Var("rd1", CudaPointerType(elem=t_i32))),
        offset=0,
        is_param=False,
    )
    assert emit_load(load) == "r1 = reinterpret_cast<int*>(rd1)[0];"


def test_emit_store_basic():
    ptr_ty = CudaPointerType(elem=t_i32)
    store = Store(
        pointer=Var("rd1", ptr_ty),
        offset=4,
        value=Var("r2", t_i32),
    )
    assert emit_store(store) == "rd1[1] = r2;"


def test_emit_assignment_from_add_f16():
    thalf = CudaType(16, CudaTypeId.Float)
    regmap = {
        ptx.Register(prefix="rs", idx=1): Var("rs1", thalf),
        ptx.Register(prefix="rs", idx=2): Var("rs2", thalf),
        ptx.Register(prefix="rs", idx=3): Var("rs3", thalf),
    }
    instr = ptx.Instruction(
        predicate=None,
        opcode="add.f16",
        operands=[
            ptx.Register(prefix="rs", idx=1),
            ptx.Register(prefix="rs", idx=2),
            ptx.Register(prefix="rs", idx=3),
        ],
    )
    assignment = emit_assignment(instr, regmap)
    assert assignment is not None
    assert emit_assignment_stmt(assignment) == "rs1 = (rs2 + rs3);"
