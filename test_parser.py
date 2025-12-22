import pytest

from parser import (
    Immediate,
    Instruction,
    Label,
    MemoryDecl,
    MemoryType,
    MemoryRef,
    ParamRef,
    Register,
    RegisterDecl,
    ScopedBlock,
    Vector,
    Branch,
    Opaque,
    EntryDirective,
    parse_branch,
    parse_module,
    parse_entry_directive,
    parse_instruction,
    parse_label,
    parse_memory_directive,
    parse_param_directive,
    parse_register_decl,
    parse_scoped_block,
)


def test_parse_param_with_align_and_array():
    pd = parse_param_directive(".param .align 4 .b8 foo[12],")
    assert pd == MemoryDecl(
        alignment=4, datatype="b8", name="foo", num_elements=12, memory_type=MemoryType.Param
    )


def test_parse_param_without_align_sets_alignment_none():
    pd = parse_param_directive("  .param .u64 bar")
    assert pd == MemoryDecl(
        alignment=None, datatype="u64", name="bar", num_elements=1, memory_type=MemoryType.Param
    )


def test_parse_param_strips_comments_and_trailing_punctuation():
    pd = parse_param_directive(".param .align 2 .b16 baz[3]; // comment here")
    assert pd == MemoryDecl(
        alignment=2, datatype="b16", name="baz", num_elements=3, memory_type=MemoryType.Param
    )


def test_parse_memory_directive_global_and_shared():
    g = parse_memory_directive(".global .align 1 .b8 foo[4];")
    assert g == MemoryDecl(
        alignment=1,
        datatype="b8",
        name="foo",
        num_elements=4,
        memory_type=MemoryType.Global,
    )
    s = parse_memory_directive(".extern .shared .u64 bar")
    assert s == MemoryDecl(
        alignment=None,
        datatype="u64",
        name="bar",
        num_elements=1,
        memory_type=MemoryType.Shared,
    )


def test_parse_param_invalid_raises():
    with pytest.raises(ValueError):
        parse_param_directive(".param foo")


def test_parse_register_decl_basic():
    decl = parse_register_decl(".reg .b32 %r<1251>;")
    assert decl == RegisterDecl(datatype="b32", prefix="r", num_regs=1251)


def test_parse_register_decl_pred_and_comments():
    decl = parse_register_decl(".reg .pred %p<7>; // predicate regs")
    assert decl == RegisterDecl(datatype="pred", prefix="p", num_regs=7)

def test_parse_register_decl_without_count():
    decl = parse_register_decl(".reg .pred p;")
    assert decl == RegisterDecl(datatype="pred", prefix="p", num_regs=1)


def test_parse_register_decl_invalid():
    with pytest.raises(ValueError):
        parse_register_decl(".reg %r<4>")


def test_parse_instruction_with_param_load():
    inst = parse_instruction("ld.param.u32 %r5, [foo];")
    assert inst == Instruction(
        predicate=None,
        opcode="ld.param.u32",
        operands=[
            Register(prefix="r", idx=5),
            MemoryRef(base=ParamRef(name="foo"), offset=0),
        ],
    )


def test_parse_instruction_predicated_and_immediate():
    inst = parse_instruction("@%p1 add.s32 %r1, %r2, 4;")
    assert inst == Instruction(
        predicate=Register(prefix="p", idx=1),
        opcode="add.s32",
        operands=[
            Register(prefix="r", idx=1),
            Register(prefix="r", idx=2),
            Immediate(value=4),
        ],
    )


def test_parse_instruction_predicate_without_percent():
    inst = parse_instruction("@p add.s32 %r1, %r2, 4;")
    assert inst.predicate == Register(prefix="p", idx=None)


def test_parse_instruction_hex_immediate_without_prefix():
    inst = parse_instruction("add.s32 %r1, %r2, 0f00000000;")
    assert isinstance(inst.operands[2], Immediate)
    assert inst.operands[2].value == int("0f00000000", 16)


def test_parse_instruction_vector_and_mem_offset():
    inst = parse_instruction("st.global.v2.u32 [%rd1+16], {%r2,%r3}")
    assert inst == Instruction(
        predicate=None,
        opcode="st.global.v2.u32",
        operands=[
            MemoryRef(base=Register(prefix="rd", idx=1), offset=16),
            Vector(values=[Register(prefix="r", idx=2), Register(prefix="r", idx=3)]),
        ],
    )


def test_parse_instruction_branch_rejected():
    with pytest.raises(ValueError):
        parse_instruction("bra $L1;")

def test_parse_branch_basic():
    br = parse_branch("bra $L1;")
    assert br.is_uniform is False
    assert br.predicate is None
    assert br.target.name == "L1"


def test_parse_branch_uniform_and_predicated():
    br = parse_branch("@%p1 bra.uni $L__BB0_4; // comment")
    assert br.is_uniform is True
    assert br.predicate == Register(prefix="p", idx=1)
    assert br.target.name == "L__BB0_4"


def test_parse_branch_predicated_without_percent():
    br = parse_branch("@p2 bra $L_next;")
    assert br.is_uniform is False
    assert br.predicate == Register(prefix="p", idx=2)
    assert br.target.name == "L_next"


def test_parse_branch_invalid_opcode():
    with pytest.raises(ValueError):
        parse_branch("jmp $L1;")


def test_label_and_branch_same_line():
    text = "L0: bra $L1;"
    block = parse_scoped_block(text)
    assert len(block.body) == 2
    assert isinstance(block.body[0], Label)
    assert block.body[0].name == "L0"
    assert isinstance(block.body[1], Branch)
    assert block.body[1].target.name == "L1"


def test_label_predicated_branch_same_line():
    text = "L2: @%p1 bra.uni $L3;"
    block = parse_scoped_block(text)
    assert len(block.body) == 2
    lbl, br = block.body
    assert isinstance(lbl, Label) and lbl.name == "L2"
    assert isinstance(br, Branch)
    assert br.is_uniform is True
    assert br.predicate == Register(prefix="p", idx=1)
    assert br.target.name == "L3"


def test_parse_label_basic():
    lbl = parse_label("$L__BB0_4:")
    assert lbl.name == "L__BB0_4"


def test_parse_label_invalid():
    with pytest.raises(ValueError):
        parse_label("not_a_label")


def test_parse_scoped_block_inline_braces():
    text = "{ .reg .b32 %r<2>; L0: add.s32 %r1, %r2, 1; }"
    root = parse_scoped_block(text)
    assert len(root.body) == 1
    inner = root.body[0]
    assert isinstance(inner, ScopedBlock)
    assert inner.registers == [RegisterDecl(datatype="b32", prefix="r", num_regs=2)]
    assert isinstance(inner.body[0], Label)
    assert inner.body[0].name == "L0"
    assert isinstance(inner.body[1], Instruction)
    assert inner.body[1].opcode == "add.s32"


def test_parse_scoped_block_nested_and_mixed_lines():
    text = """
    .reg .b32 %r<1>;
    outer:
    { .reg .pred %p<1>;
      @%p0 bra.uni $L1;
      { mov.u32 %r0, %r0; }
      $L1:
    }
    """
    root = parse_scoped_block(text)
    assert root.registers == [RegisterDecl(datatype="b32", prefix="r", num_regs=1)]
    assert isinstance(root.body[0], Label) and root.body[0].name == "outer"
    outer_block = root.body[1]
    assert isinstance(outer_block, ScopedBlock)
    assert outer_block.registers == [RegisterDecl(datatype="pred", prefix="p", num_regs=1)]
    assert isinstance(outer_block.body[0], Branch)
    nested = outer_block.body[1]
    assert isinstance(nested, ScopedBlock)
    assert isinstance(nested.body[0], Instruction)
    assert isinstance(outer_block.body[-1], Label) and outer_block.body[-1].name == "L1"


def test_parse_scoped_block_unmatched_brace():
    with pytest.raises(ValueError):
        parse_scoped_block("}")


def test_parse_scoped_block_with_vector_operand():
    text = """
    {
      {
        wgmma.mma_async.sync.aligned.m64n64k16.f16.f16.f16 {%r1, %r2, %r3, %r4}, %rd1, %rd2, p, 1, 1, 1, 1;
      }
    }
    """
    root = parse_scoped_block(text)
    inner = root.body[0]
    assert isinstance(inner, ScopedBlock)
    nested = inner.body[0]
    assert isinstance(nested, ScopedBlock)
    inst = nested.body[0]
    assert isinstance(inst, Instruction)
    assert inst.opcode.startswith("wgmma.mma_async")
    vec = inst.operands[0]
    assert isinstance(vec, Vector)
    assert len(vec.values) == 4


def test_parse_entry_directive_basic():
    text = """
.entry foo(
    .param .u64 param0,
    .param .align 4 .b8 param1[4]
)
.maxntid 128, 1, 1
{
  L0:
  ret;
}
"""
    entry = parse_entry_directive(text)
    assert entry.name == "foo"
    assert [p.name for p in entry.params] == ["param0", "param1"]
    assert entry.params[0].datatype == "u64"
    assert entry.params[1].num_elements == 4
    # Directives captured before body
    assert entry.directives == [Opaque(content=".maxntid 128, 1, 1")]
    # Body root should contain an inner block with label/instruction
    inner = entry.body.body[0]
    assert isinstance(inner, ScopedBlock)
    assert isinstance(inner.body[0], Label)
    assert inner.body[0].name == "L0"
    assert isinstance(inner.body[1], Instruction)
    assert inner.body[1].opcode == "ret"


def test_parse_module_top_globals_and_entry():
    text = """
.version 9.0
.target sm_90a
.address_size 64
.global .align 1 .b8 foo[4];
.shared .u32 shmem;

.entry bar(
    .param .u64 p0
)
{
  ret;
}
"""
    module = parse_module(text)
    assert len(module.statements) == 6
    assert module.statements[0] == Opaque(content=".version 9.0")
    assert module.statements[1] == Opaque(content=".target sm_90a")
    assert module.statements[2] == Opaque(content=".address_size 64")
    assert isinstance(module.statements[3], MemoryDecl)
    assert module.statements[3].name == "foo"
    assert module.statements[4].memory_type == MemoryType.Shared
    assert isinstance(module.statements[5], EntryDirective)
    assert module.statements[5].name == "bar"


def test_parse_module_fixture_file():
    from pathlib import Path

    fixture = Path(__file__).parent / "tests" / "fixtures" / "x.ptx"
    text = fixture.read_text()
    module = parse_module(text)
    assert len(module.statements) > 0


def test_parse_instruction_wgmma_vector_operands():
    line = (
        "wgmma.mma_async.sync.aligned.m64n64k16.f16.f16.f16 "
        "{%r1248, %r1247, %r1246, %r1245, %r1244, %r1243, %r1242, %r1241, "
        " %r1240, %r1239, %r1238, %r1237, %r1236, %r1235, %r1234, %r1233}, "
        "%rd74, %rd79, p, 1, 1, 1, 1;"
    )
    inst = parse_instruction(line)
    assert inst.opcode == "wgmma.mma_async.sync.aligned.m64n64k16.f16.f16.f16"
    # First operand is a vector of 16 registers
    vec = inst.operands[0]
    assert isinstance(vec, Vector)
    assert len(vec.values) == 16
    # Next operands are registers, a param ref, and immediates
    assert inst.operands[1] == Register(prefix="rd", idx=74)
    assert inst.operands[2] == Register(prefix="rd", idx=79)
    assert inst.operands[3] == Register(prefix="p", idx=None)
    assert [op.value for op in inst.operands[4:]] == [1, 1, 1, 1]
