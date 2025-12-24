import ptx
from cudagen.render_inst import emit_assignment
from cudagen.types import CudaType, Var, CudaTypeId
from emission.expr import emit_assignment_stmt


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
