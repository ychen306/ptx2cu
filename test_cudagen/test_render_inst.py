import pytest

import ptx
from cudagen import InlineAsm, Var, emit_inline_asm
from cudagen.types import (
    CudaType,
    CudaTypeId,
    Assignment,
    BinaryOperator,
    BinaryOpcode,
    BitCast,
    ConstantInt,
)
from cudagen.render_inst import emit_assignment
from ptx import MemoryDecl, MemorySymbol, MemoryType

t32 = CudaType(32, CudaTypeId.Unsigned)
t64 = CudaType(64, CudaTypeId.Unsigned)
tpred = CudaType(32, CudaTypeId.Unsigned, True)


def test_emit_inline_asm_basic():
    regmap = {
        ptx.Register(prefix="r", idx=1): Var("r1", t32),
        ptx.Register(prefix="r", idx=2): Var("r2", t32),
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
        Var("r1", t32),
        Var("r1", t32),
        Var("r2", t32),
    ]
    assert asm.outputs == [Var("r1", t32)]


def test_emit_inline_asm_vector_and_memory():
    regmap = {
        ptx.Register(prefix="r", idx=1): Var("r1", t32),
        ptx.Register(prefix="r", idx=2): Var("r2", t32),
        ptx.Register(prefix="rd", idx=0): Var("rd0", t64),
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
    regmap = {r: Var(f"{r.prefix}{r.idx}", t32) for r in regs}
    regmap.update(
        {
            ptx.Register(prefix="rd", idx=74): Var("rd74", t64),
            ptx.Register(prefix="rd", idx=79): Var("rd79", t64),
        }
    )
    regmap[ptx.Register(prefix="p", idx=None)] = Var("p", tpred)

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
    regmap = {ptx.Register(prefix="r", idx=1): Var("r1", t32)}
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


def test_emit_assignment_add():
    regmap = {
        ptx.Register(prefix="r", idx=1): Var("r1", t32),
        ptx.Register(prefix="r", idx=2): Var("r2", t32),
        ptx.Register(prefix="r", idx=3): Var("r3", t32),
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
    assert isinstance(assignment, Assignment)
    assert assignment.lhs == Var("r1", t32)
    assert assignment.rhs == BinaryOperator(
        opcode=BinaryOpcode.Add, operand_a=Var("r2", t32), operand_b=Var("r3", t32)
    )


def test_emit_assignment_reject_mul_wide():
    regmap = {
        ptx.Register(prefix="r", idx=1): Var("r1", t32),
        ptx.Register(prefix="r", idx=2): Var("r2", t32),
        ptx.Register(prefix="r", idx=3): Var("r3", t32),
    }
    instr = ptx.Instruction(
        predicate=None,
        opcode="mul.wide.s32",
        operands=[
            ptx.Register(prefix="r", idx=1),
            ptx.Register(prefix="r", idx=2),
            ptx.Register(prefix="r", idx=3),
        ],
    )
    assert emit_assignment(instr, regmap) is None


def test_emit_assignment_bitcasts_int_inputs_for_float_opcode():
    # Integer-typed registers used with a float opcode should be bitcast to float
    regmap = {
        ptx.Register(prefix="r", idx=1): Var("r1", t32),
        ptx.Register(prefix="r", idx=2): Var("r2", t32),
        ptx.Register(prefix="r", idx=3): Var("r3", t32),
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
    assert isinstance(assignment, Assignment)
    # RHS should be bitcast to the dest type, with inner BinaryOperator and BitCast operands to float
    rhs = assignment.rhs
    assert isinstance(rhs, BitCast)
    inner = rhs.operand
    assert isinstance(inner, BinaryOperator)
    assert inner.opcode == BinaryOpcode.FAdd
    assert isinstance(inner.operand_a, BitCast)
    assert isinstance(inner.operand_b, BitCast)


def test_emit_assignment_with_immediate_shift():
    regmap = {
        ptx.Register(prefix="r", idx=1): Var("r1", t32),
        ptx.Register(prefix="r", idx=2): Var("r2", t32),
    }
    instr = ptx.Instruction(
        predicate=None,
        opcode="shr.s32",
        operands=[
            ptx.Register(prefix="r", idx=1),
            ptx.Register(prefix="r", idx=2),
            ptx.Immediate("6"),
        ],
    )
    assignment = emit_assignment(instr, regmap)
    assert isinstance(assignment, Assignment)
    rhs = assignment.rhs
    assert isinstance(rhs, BinaryOperator)
    assert rhs.opcode in (BinaryOpcode.AShr, BinaryOpcode.LShr)
    assert isinstance(rhs.operand_b, ConstantInt)
    assert rhs.operand_b.value == 6
