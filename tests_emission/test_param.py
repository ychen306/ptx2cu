import ptx
from cudagen import Load, emit_ld_param
from cudagen.types import MemoryDecl, Var
from emission.param import emit_load, get_type_decl_for_param


def test_get_type_decl_scalar():
    decl = MemoryDecl(alignment=None, datatype="u64", name="p0", num_elements=1, memory_type=ptx.MemoryType.Param)
    struct_def, type_name = get_type_decl_for_param(decl)
    assert struct_def is None
    assert type_name == "unsigned long long"


def test_get_type_decl_array():
    decl = MemoryDecl(alignment=None, datatype="f16", name="p1", num_elements=4, memory_type=ptx.MemoryType.Param)
    struct_def, type_name = get_type_decl_for_param(decl)
    assert struct_def == "struct Param_f16_x_4 { __half buf[4]; };"
    assert type_name == "Param_f16_x_4"


def test_emit_load_scalar_from_ld_param():
    instr = ptx.Instruction(
        predicate=None,
        opcode="ld.param.u32",
        operands=[
            ptx.Register(prefix="r", idx=1),
            ptx.MemoryRef(base=ptx.ParamRef(name="p0"), offset=0),
        ],
    )
    regmap = {
        ptx.Register(prefix="r", idx=1): Var("r1", 32, False)
    }
    param_map = {
        "p0": MemoryDecl(alignment=None, datatype="u32", name="p0", num_elements=1, memory_type=ptx.MemoryType.Param)
    }
    load = emit_ld_param(instr, regmap, param_map)
    assert isinstance(load, Load)
    assert emit_load(load) == "r1 = reinterpret_cast<unsigned int*>(&p0)[0];"


def test_emit_load_array_offset():
    instr = ptx.Instruction(
        predicate=None,
        opcode="ld.param.f16",
        operands=[
            ptx.Register(prefix="r", idx=2),
            ptx.MemoryRef(base=ptx.ParamRef(name="arr"), offset=4),
        ],
    )
    regmap = {
        ptx.Register(prefix="r", idx=2): Var("r2", 32, False)
    }
    param_map = {
        "arr": MemoryDecl(alignment=None, datatype="f16", name="arr", num_elements=4, memory_type=ptx.MemoryType.Param)
    }
    load = emit_ld_param(instr, regmap, param_map)
    # offset 4 bytes -> index 2 for f16 (2-byte elements)
    assert emit_load(load) == "r2 = reinterpret_cast<__half*>(&arr)[2];"
