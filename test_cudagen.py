import pytest

import ptx
from cudagen import InlineAsm, RegisterInfo, Var, emit_inline_asm, emit_inline_asm_string


def test_emit_inline_asm_basic():
    regmap = {
        ptx.Register(prefix="r", idx=1): RegisterInfo(
            decl=ptx.RegisterDecl(datatype="b32", prefix="r", num_regs=1), c_var=Var("r1")
        ),
        ptx.Register(prefix="r", idx=2): RegisterInfo(
            decl=ptx.RegisterDecl(datatype="b32", prefix="r", num_regs=1), c_var=Var("r2")
        ),
    }
    instr = ptx.Instruction(
        predicate=None,
        opcode="add.s32",
        operands=[
            ptx.Register(prefix="r", idx=1),
            ptx.Register(prefix="r", idx=1),
            ptx.Register(prefix="r", idx=2),
        ],
    )
    asm = emit_inline_asm(instr, regmap)
    assert isinstance(asm, InlineAsm)
    assert asm.template == "add.s32 %0, %1, %2;"
    assert asm.arguments == [Var("r1"), Var("r1"), Var("r2")]
    assert asm.outputs == [Var("r1")]
    assert asm.constraints == {Var("r1"): "r", Var("r2"): "r"}


def test_emit_inline_asm_vector_and_memory():
    regmap = {
        ptx.Register(prefix="r", idx=1): RegisterInfo(
            decl=ptx.RegisterDecl(datatype="b32", prefix="r", num_regs=1), c_var=Var("r1")
        ),
        ptx.Register(prefix="r", idx=2): RegisterInfo(
            decl=ptx.RegisterDecl(datatype="b32", prefix="r", num_regs=1), c_var=Var("r2")
        ),
        ptx.Register(prefix="rd", idx=0): RegisterInfo(
            decl=ptx.RegisterDecl(datatype="b64", prefix="rd", num_regs=1), c_var=Var("rd0")
        ),
    }
    instr = ptx.Instruction(
        predicate=None,
        opcode="st.global.v2.u32",
        operands=[
            ptx.MemoryRef(base=ptx.Register(prefix="rd", idx=0), offset=16),
            ptx.Vector(values=[ptx.Register(prefix="r", idx=1), ptx.Register(prefix="r", idx=2)]),
        ],
    )
    asm = emit_inline_asm(instr, regmap)
    assert asm.template.startswith("st.global.v2.u32")
    assert "[%0+16]" in asm.template
    assert "{%1, %2}" in asm.template
    assert asm.constraints[Var("rd0")] == "l"
    assert asm.constraints[Var("r1")] == "r"
    assert asm.constraints[Var("r2")] == "r"
    assert asm.clobbers_memory is True


def test_emit_inline_asm_missing_register():
    regmap = {}
    instr = ptx.Instruction(
        predicate=None,
        opcode="add.s32",
        operands=[ptx.Register(prefix="r", idx=0)],
    )
    with pytest.raises(ValueError):
        emit_inline_asm(instr, regmap)


def test_emit_inline_asm_wgmma_vector():
    # Mirrors the wgmma operand structure tested in parser tests
    regs = [ptx.Register(prefix="r", idx=i) for i in range(1, 5)]
    regmap = {
        r: RegisterInfo(
            decl=ptx.RegisterDecl(datatype="b32", prefix="r", num_regs=1), c_var=Var(f"{r.prefix}{r.idx}")
        )
        for r in regs
    }
    regmap.update(
        {
            ptx.Register(prefix="rd", idx=74): RegisterInfo(
                decl=ptx.RegisterDecl(datatype="b64", prefix="rd", num_regs=1), c_var=Var("rd74")
            ),
            ptx.Register(prefix="rd", idx=79): RegisterInfo(
                decl=ptx.RegisterDecl(datatype="b64", prefix="rd", num_regs=1), c_var=Var("rd79")
            ),
        }
    )
    regmap[ptx.Register(prefix="p", idx=None)] = RegisterInfo(
        decl=ptx.RegisterDecl(datatype="pred", prefix="p", num_regs=1), c_var=Var("p")
    )

    instr = ptx.Instruction(
        predicate=None,
        opcode="wgmma.mma_async.sync.aligned.m64n64k16.f16.f16.f16",
        operands=[
            ptx.Vector(values=regs),
            ptx.Register(prefix="rd", idx=74),
            ptx.Register(prefix="rd", idx=79),
            ptx.Register(prefix="p", idx=None),
            ptx.Immediate(1),
            ptx.Immediate(1),
            ptx.Immediate(1),
            ptx.Immediate(1),
        ],
    )
    asm = emit_inline_asm(instr, regmap)
    assert asm.template.startswith("wgmma.mma_async.sync.aligned")
    assert asm.constraints[Var("rd74")] == "l"
    assert asm.constraints[Var("rd79")] == "l"
    for r in regs:
        assert asm.constraints[Var(f"{r.prefix}{r.idx}")] == "r"


def test_emit_inline_asm_string_basic():
    regmap = {
        ptx.Register(prefix="r", idx=1): RegisterInfo(
            decl=ptx.RegisterDecl(datatype="b32", prefix="r", num_regs=1), c_var=Var("r1")
        ),
        ptx.Register(prefix="r", idx=2): RegisterInfo(
            decl=ptx.RegisterDecl(datatype="b32", prefix="r", num_regs=1), c_var=Var("r2")
        ),
    }
    instr = ptx.Instruction(
        predicate=None,
        opcode="add.s32",
        operands=[
            ptx.Register(prefix="r", idx=1),
            ptx.Register(prefix="r", idx=1),
            ptx.Register(prefix="r", idx=2),
        ],
    )
    s = emit_inline_asm_string(instr, regmap)
    assert (
        s
        == 'asm volatile("add.s32 %0, %1, %2;" : "+r"(r1) : "r"(r2) : );'
    )


def test_emit_inline_asm_string_wgmma():
    regs = [ptx.Register(prefix="r", idx=i) for i in range(1, 5)]
    regmap = {
        r: RegisterInfo(
            decl=ptx.RegisterDecl(datatype="b32", prefix="r", num_regs=1), c_var=Var(f"{r.prefix}{r.idx}")
        )
        for r in regs
    }
    regmap.update(
        {
            ptx.Register(prefix="rd", idx=74): RegisterInfo(
                decl=ptx.RegisterDecl(datatype="b64", prefix="rd", num_regs=1), c_var=Var("rd74")
            ),
            ptx.Register(prefix="rd", idx=79): RegisterInfo(
                decl=ptx.RegisterDecl(datatype="b64", prefix="rd", num_regs=1), c_var=Var("rd79")
            ),
        }
    )
    regmap[ptx.Register(prefix="p", idx=None)] = RegisterInfo(
        decl=ptx.RegisterDecl(datatype="pred", prefix="p", num_regs=1), c_var=Var("p")
    )

    instr = ptx.Instruction(
        predicate=None,
        opcode="wgmma.mma_async.sync.aligned.m64n64k16.f16.f16.f16",
        operands=[
            ptx.Vector(values=regs),
            ptx.Register(prefix="rd", idx=74),
            ptx.Register(prefix="rd", idx=79),
            ptx.Register(prefix="p", idx=None),
            ptx.Immediate(1),
            ptx.Immediate(1),
            ptx.Immediate(1),
            ptx.Immediate(1),
        ],
    )
    s = emit_inline_asm_string(instr, regmap)
    assert (
        s
        == 'asm volatile("{ .reg .pred %ptmp0; setp.ne.u32 %ptmp0, %6, 0; wgmma.mma_async.sync.aligned.m64n64k16.f16.f16.f16 {%0, %1, %2, %3}, %4, %5, %ptmp0, 1, 1, 1, 1; }" : "+r"(r1), "+r"(r2), "+r"(r3), "+r"(r4) : "l"(rd74), "l"(rd79), "r"(p) : );'
    )


def test_parse_and_emit_wgmma():
    line = (
        "wgmma.mma_async.sync.aligned.m64n64k16.f16.f16.f16 "
        "{%r1253, %r1252, %r1251, %r1250, %r1249, %r1248, %r1247, %r1246, "
        "%r1245, %r1244, %r1243, %r1242, %r1241, %r1240, %r1239, %r1238}, "
        "%rd86, %rd91, p, 1, 1, 0, 0;"
    )
    from parser import parse_instruction

    instr = parse_instruction(line)
    regs = list(instr.operands[0].values)
    regmap = {
        r: RegisterInfo(
            decl=ptx.RegisterDecl(datatype="b32", prefix="r", num_regs=1), c_var=Var(f"{r.prefix}{r.idx}")
        )
        for r in regs
    }
    regmap[ptx.Register(prefix="rd", idx=86)] = RegisterInfo(
        decl=ptx.RegisterDecl(datatype="b64", prefix="rd", num_regs=1), c_var=Var("rd86")
    )
    regmap[ptx.Register(prefix="rd", idx=91)] = RegisterInfo(
        decl=ptx.RegisterDecl(datatype="b64", prefix="rd", num_regs=1), c_var=Var("rd91")
    )
    regmap[ptx.Register(prefix="p", idx=None)] = RegisterInfo(
        decl=ptx.RegisterDecl(datatype="pred", prefix="p", num_regs=1), c_var=Var("p")
    )

    s = emit_inline_asm_string(instr, regmap)
    assert (
        s
        == 'asm volatile("{ .reg .pred %ptmp0; setp.ne.u32 %ptmp0, %18, 0; wgmma.mma_async.sync.aligned.m64n64k16.f16.f16.f16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, %16, %17, %ptmp0, 1, 1, 0, 0; }" : "+r"(r1253), "+r"(r1252), "+r"(r1251), "+r"(r1250), "+r"(r1249), "+r"(r1248), "+r"(r1247), "+r"(r1246), "+r"(r1245), "+r"(r1244), "+r"(r1243), "+r"(r1242), "+r"(r1241), "+r"(r1240), "+r"(r1239), "+r"(r1238) : "l"(rd86), "l"(rd91), "r"(p) : );'
    )
