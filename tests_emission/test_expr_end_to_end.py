import ptx
from cudagen.render_inst import emit_assignment, emit_mov, emit_mad_lo, emit_predicate
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
t_s32 = CudaType(32, CudaTypeId.Signed)
t_f32 = CudaType(32, CudaTypeId.Float)
t_u64 = CudaType(64, CudaTypeId.Unsigned)


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


def test_emit_mov_assignment_end_to_end():
    regmap = {
        ptx.Register(prefix="r", idx=1): Var("r1", t_i32),
        ptx.Register(prefix="r", idx=2): Var("r2", t_f32),
    }
    instr = ptx.Instruction(
        predicate=None,
        opcode="mov.u32",
        operands=[
            ptx.Register(prefix="r", idx=1),
            ptx.Register(prefix="r", idx=2),
        ],
    )
    assignment = emit_mov(instr, regmap)
    assert assignment is not None
    assert emit_assignment_stmt(assignment) == "r1 = __float_as_uint(r2);"


def test_emit_mad_lo_s32_end_to_end():
    regmap = {
        ptx.Register(prefix="r", idx=1): Var("r1", t_s32),
        ptx.Register(prefix="r", idx=2): Var("r2", t_s32),
        ptx.Register(prefix="r", idx=3): Var("r3", t_s32),
        ptx.Register(prefix="r", idx=4): Var("r4", t_s32),
    }
    instr = ptx.Instruction(
        predicate=None,
        opcode="mad.lo.s32",
        operands=[
            ptx.Register(prefix="r", idx=1),
            ptx.Register(prefix="r", idx=2),
            ptx.Register(prefix="r", idx=3),
            ptx.Register(prefix="r", idx=4),
        ],
    )
    assignment = emit_mad_lo(instr, regmap)
    assert assignment is not None
    assert (
        emit_assignment_stmt(assignment)
        == "r1 = ((int)(((long long)(r2) * (long long)(r3))) + r4);"
    )


def test_emit_predicate_end_to_end():
    t_s32 = CudaType(32, CudaTypeId.Signed)
    regmap = {
        ptx.Register(prefix="p", idx=1): Var(
            "p1", CudaType(32, CudaTypeId.Unsigned, True)
        ),
        ptx.Register(prefix="r", idx=2): Var("r2", t_s32),
        ptx.Register(prefix="r", idx=3): Var("r3", t_s32),
    }
    instr = ptx.Instruction(
        predicate=None,
        opcode="setp.ge.s32",
        operands=[
            ptx.Register(prefix="p", idx=1),
            ptx.Register(prefix="r", idx=2),
            ptx.Register(prefix="r", idx=3),
        ],
    )
    assignment = emit_predicate(instr, regmap)
    assert assignment is not None
    # Should produce a signed compare
    assert emit_assignment_stmt(assignment) == "p1 = (r2 >= r3);"


def test_emit_assignment_from_mul_wide_u32():
    regmap = {
        ptx.Register(prefix="rd", idx=1): Var("rd1", t_u64),
        ptx.Register(prefix="r", idx=2): Var("r2", t_i32),
        ptx.Register(prefix="r", idx=3): Var("r3", t_i32),
    }
    instr = ptx.Instruction(
        predicate=None,
        opcode="mul.wide.u32",
        operands=[
            ptx.Register(prefix="rd", idx=1),
            ptx.Register(prefix="r", idx=2),
            ptx.Register(prefix="r", idx=3),
        ],
    )
    assignment = emit_assignment(instr, regmap)
    assert assignment is not None
    assert (
        emit_assignment_stmt(assignment)
        == "rd1 = ((unsigned long long)(r2) * (unsigned long long)(r3));"
    )
