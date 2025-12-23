import pytest

import ptx
from cudagen import InlineAsm, Var, emit_inline_asm
from ptx import MemoryDecl, MemorySymbol, MemoryType


def test_emit_inline_asm_basic():
    regmap = {
        ptx.Register(prefix="r", idx=1): Var("r1", 32, False),
        ptx.Register(prefix="r", idx=2): Var("r2", 32, False),
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
    assert asm.arguments == [
        Var("r1", 32, False),
        Var("r1", 32, False),
        Var("r2", 32, False),
    ]
    assert asm.outputs == [Var("r1", 32, False)]


def test_emit_inline_asm_vector_and_memory():
    regmap = {
        ptx.Register(prefix="r", idx=1): Var("r1", 32, False),
        ptx.Register(prefix="r", idx=2): Var("r2", 32, False),
        ptx.Register(prefix="rd", idx=0): Var("rd0", 64, False),
    }
    instr = ptx.Instruction(
        predicate=None,
        opcode="st.global.v2.u32",
        operands=[
            ptx.MemoryRef(base=ptx.Register(prefix="rd", idx=0), offset=16),
            ptx.Vector(
                values=[
                    ptx.Register(prefix="r", idx=1),
                    ptx.Register(prefix="r", idx=2),
                ]
            ),
        ],
    )
    asm = emit_inline_asm(instr, regmap)
    assert asm.template.startswith("st.global.v2.u32")
    assert "[%0+16]" in asm.template
    assert "{%1, %2}" in asm.template
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
    regmap = {r: Var(f"{r.prefix}{r.idx}", 32, False) for r in regs}
    regmap.update(
        {
            ptx.Register(prefix="rd", idx=74): Var("rd74", 64, False),
            ptx.Register(prefix="rd", idx=79): Var("rd79", 64, False),
        }
    )
    regmap[ptx.Register(prefix="p", idx=None)] = Var("p", 32, False, True)

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


def test_emit_inline_asm_with_memory_symbol():
    regmap = {ptx.Register(prefix="r", idx=1): Var("r1", 32, False)}
    mem_decl = MemoryDecl(
        alignment=None,
        datatype="u32",
        name="shared_memory",
        num_elements=0,
        memory_type=MemoryType.Shared,
    )
    instr = ptx.Instruction(
        predicate=None,
        opcode="mov.u32",
        operands=[
            ptx.Register(prefix="r", idx=1),
            MemorySymbol(decl=mem_decl),
        ],
    )
    asm = emit_inline_asm(instr, regmap)
    assert asm.template == "mov.u32 %0, %1;"
    assert asm.arguments[1].symbol.decl is mem_decl
    assert asm.arguments[1].bitwidth == 32
