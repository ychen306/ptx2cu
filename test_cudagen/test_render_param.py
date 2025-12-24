import ptx
from cudagen import Load, emit_ld_param
from cudagen.types import MemoryDecl, Var, CudaType

t32 = CudaType(32, False)
tf16 = CudaType(16, True)


def test_emit_ld_param_scalar():
    instr = ptx.Instruction(
        predicate=None,
        opcode="ld.param.u32",
        operands=[
            ptx.Register(prefix="r", idx=1),
            ptx.MemoryRef(base=ptx.ParamRef(name="p0"), offset=0),
        ],
    )
    regmap = {ptx.Register(prefix="r", idx=1): Var("r1", t32)}
    param_map = {
        "p0": MemoryDecl(
            alignment=None,
            datatype="u64",
            name="p0",
            num_elements=1,
            memory_type=ptx.MemoryType.Param,
        )
    }
    load = emit_ld_param(instr, regmap, param_map)
    assert isinstance(load, Load)
    assert load.dst == Var("r1", t32)
    assert load.src == Var("p0", t32)
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
    regmap = {ptx.Register(prefix="r", idx=2): Var("r2", t32)}
    param_map = {
        "arr": MemoryDecl(
            alignment=None,
            datatype="f16",
            name="arr",
            num_elements=4,
            memory_type=ptx.MemoryType.Param,
        )
    }
    load = emit_ld_param(instr, regmap, param_map)
    assert load.dst == Var("r2", t32)
    assert load.src == Var("arr", tf16)
    assert load.offset == 4


def test_emit_ld_param_opcode_bitwidth_overrides_decl():
    instr = ptx.Instruction(
        predicate=None,
        opcode="ld.param.u32",
        operands=[
            ptx.Register(prefix="r", idx=3),
            ptx.MemoryRef(base=ptx.ParamRef(name="buf"), offset=0),
        ],
    )
    regmap = {ptx.Register(prefix="r", idx=3): Var("r3", t32)}
    param_map = {
        "buf": MemoryDecl(
            alignment=None,
            datatype="b8",
            name="buf",
            num_elements=8,
            memory_type=ptx.MemoryType.Param,
        )
    }
    load = emit_ld_param(instr, regmap, param_map)
    # Even though the decl is bytes, opcode says u32
    assert load.ty.bitwidth == 32
    assert load.ty.is_float is False
