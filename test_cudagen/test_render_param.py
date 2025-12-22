import ptx
from cudagen import Load, emit_ld_param
from cudagen.types import MemoryDecl, RegisterInfo, Var


def test_emit_ld_param_scalar():
    instr = ptx.Instruction(
        predicate=None,
        opcode="ld.param.u32",
        operands=[
            ptx.Register(prefix="r", idx=1),
            ptx.MemoryRef(base=ptx.ParamRef(name="p0"), offset=0),
        ],
    )
    regmap = {
        ptx.Register(prefix="r", idx=1): RegisterInfo(
            decl=ptx.RegisterDecl(datatype="b32", prefix="r", num_regs=1), c_var=Var("r1", 32, False)
        )
    }
    param_map = {
        "p0": MemoryDecl(alignment=None, datatype="u32", name="p0", num_elements=1, memory_type=ptx.MemoryType.Param)
    }
    load = emit_ld_param(instr, regmap, param_map)
    assert isinstance(load, Load)
    assert load.dst == Var("r1", 32, False)
    assert load.src == Var("p0", 32, False)
    assert load.offset == 0


def test_emit_ld_param_array_offset():
    instr = ptx.Instruction(
        predicate=None,
        opcode="ld.param.f16",
        operands=[
            ptx.Register(prefix="r", idx=2),
            ptx.MemoryRef(base=ptx.ParamRef(name="arr"), offset=4),
        ],
    )
    regmap = {
        ptx.Register(prefix="r", idx=2): RegisterInfo(
            decl=ptx.RegisterDecl(datatype="b16", prefix="r", num_regs=1), c_var=Var("r2", 32, False)
        )
    }
    param_map = {
        "arr": MemoryDecl(alignment=None, datatype="f16", name="arr", num_elements=4, memory_type=ptx.MemoryType.Param)
    }
    load = emit_ld_param(instr, regmap, param_map)
    assert load.dst == Var("r2", 32, False)
    assert load.src == Var("arr", 16, True)
    assert load.offset == 4
