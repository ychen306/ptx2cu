import ptx
from cudagen.types import Var, CudaType, CudaTypeId
from ptx import MemoryDecl, MemorySymbol, MemoryType
from emission.inst import emit_inline_asm_ir

t32 = CudaType(32, CudaTypeId.Unsigned)
t64 = CudaType(64, CudaTypeId.Unsigned)
tf32 = CudaType(32, CudaTypeId.Float)
tpred = CudaType(32, CudaTypeId.Unsigned, True)


def test_emit_inline_asm_string_basic():
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
    from cudagen.render_inst import emit_inline_asm

    s = emit_inline_asm_ir(emit_inline_asm(instr, regmap))
    assert s == 'asm volatile("add.s32 %0, %1, %2;" : "+r"(r1) : "r"(r2) : );'


def test_emit_inline_asm_string_wgmma():
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
    from cudagen.render_inst import emit_inline_asm

    s = emit_inline_asm_ir(emit_inline_asm(instr, regmap))
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
    regmap = {r: Var(f"{r.prefix}{r.idx}", t32) for r in regs}
    regmap[ptx.Register(prefix="rd", idx=86)] = Var("rd86", t64)
    regmap[ptx.Register(prefix="rd", idx=91)] = Var("rd91", t64)
    regmap[ptx.Register(prefix="p", idx=None)] = Var("p", tpred)

    from cudagen.render_inst import emit_inline_asm

    s = emit_inline_asm_ir(emit_inline_asm(instr, regmap))
    assert (
        s
        == 'asm volatile("{ .reg .pred %ptmp0; setp.ne.u32 %ptmp0, %18, 0; wgmma.mma_async.sync.aligned.m64n64k16.f16.f16.f16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, %16, %17, %ptmp0, 1, 1, 0, 0; }" : "+r"(r1253), "+r"(r1252), "+r"(r1251), "+r"(r1250), "+r"(r1249), "+r"(r1248), "+r"(r1247), "+r"(r1246), "+r"(r1245), "+r"(r1244), "+r"(r1243), "+r"(r1242), "+r"(r1241), "+r"(r1240), "+r"(r1239), "+r"(r1238) : "l"(rd86), "l"(rd91), "r"(p) : );'
    )


def test_emit_inline_asm_string_with_memory_symbol():
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
        operands=[ptx.Register(prefix="r", idx=1), MemorySymbol(decl=mem_decl)],
    )
    from cudagen.render_inst import emit_inline_asm

    asm_ir = emit_inline_asm(instr, regmap)
    from cudagen.render_inst import emit_inline_asm

    s = emit_inline_asm_ir(emit_inline_asm(instr, regmap))
    assert (
        s
        == 'asm volatile("{ .reg .u64 %ptr64_0; .reg .u32 %ptr32_0; cvta.shared.u64 %ptr64_0, %1; cvt.u32.u64 %ptr32_0, %ptr64_0; mov.u32 %0, %ptr32_0; }" : "+r"(r1) : "l"(&shared_memory) : );'
    )


def test_emit_inline_asm_string_with_memory_symbol_32bit_addr():
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
        operands=[ptx.Register(prefix="r", idx=1), MemorySymbol(decl=mem_decl)],
    )
    from cudagen.render_inst import emit_inline_asm

    asm_ir = emit_inline_asm(instr, regmap)
    # Manually tweak AddressOf bitwidth to exercise downcast path
    for arg in asm_ir.arguments:
        if hasattr(arg, "symbol"):
            arg.bitwidth = 32
    s = emit_inline_asm_ir(asm_ir)
    assert "cvta.shared.u64" in s
    assert "cvt.u32.u64" in s


def test_emit_inline_asm_string_predicate_placeholder_not_reused():
    # Regression: ensure placeholder is assigned once per Var
    regmap = {
        ptx.Register(prefix="p", idx=None): Var("p1", tpred),
        ptx.Register(prefix="r", idx=1): Var("r1", t32),
        ptx.Register(prefix="r", idx=2): Var("r2", t32),
    }
    instr = ptx.Instruction(
        predicate=None,
        opcode="setp.ge.s32",
        operands=[
            ptx.Register(prefix="p", idx=None),
            ptx.Register(prefix="r", idx=1),
            ptx.Register(prefix="r", idx=2),
        ],
    )
    from cudagen.render_inst import emit_inline_asm

    s = emit_inline_asm_ir(emit_inline_asm(instr, regmap))
    assert s.count("%") >= 3  # placeholders exist
    # ensure no extra placeholder was generated for the predicate var
    assert "%3" not in s  # only %0-%2 should appear for three operands


def test_emit_inline_asm_string_store_has_inputs_only():
    regmap = {
        ptx.Register(prefix="f", idx=4): Var("f4", tf32),
        ptx.Register(prefix="rd", idx=7): Var("rd7", t64),
    }
    instr = ptx.Instruction(
        predicate=None,
        opcode="st.global.f32",
        operands=[
            ptx.MemoryRef(base=ptx.Register(prefix="rd", idx=7), offset=0),
            ptx.Register(prefix="f", idx=4),
        ],
    )
    from cudagen.render_inst import emit_inline_asm

    s = emit_inline_asm_ir(emit_inline_asm(instr, regmap))
    # Expect both operands as inputs, no outputs
    assert "st.global.f32 [%0], %1;" in s
    assert ':+f"(f4)' not in s
    assert '"f"(f4)' in s
    assert '"l"(rd7)' in s


def test_emit_inline_asm_half_operand_uses_short_cast():
    thalf = CudaType(16, CudaTypeId.Float)
    regmap = {
        ptx.Register(prefix="rs", idx=1): Var("rs1", thalf),
        ptx.Register(prefix="rs", idx=2): Var("rs2", thalf),
        ptx.Register(prefix="rs", idx=3): Var("rs3", thalf),
    }
    instr = ptx.Instruction(
        predicate=None,
        opcode="mul.f16",
        operands=[
            ptx.Register(prefix="rs", idx=1),
            ptx.Register(prefix="rs", idx=2),
            ptx.Register(prefix="rs", idx=3),
        ],
    )
    from cudagen.render_inst import emit_inline_asm

    s = emit_inline_asm_ir(emit_inline_asm(instr, regmap))
    # Both inputs and output should use *(short *)&var to satisfy h constraint
    assert "*(short *)&rs1" in s
    assert "*(short *)&rs2" in s
    assert "*(short *)&rs3" in s
