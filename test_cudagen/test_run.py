import ptx

from cudagen import CudaGen, Var
from cudagen.types import InlineAsm, Load, CudaBranch, CudaLabel


def test_run_entry_lowering():
    entry = ptx.EntryDirective(
        name="kernel",
        params=[
            ptx.MemoryDecl(
                alignment=None,
                datatype="u32",
                name="p0",
                num_elements=1,
                memory_type=ptx.MemoryType.Param,
            ),
        ],
        directives=[],
        body=ptx.ScopedBlock(
            registers=[ptx.RegisterDecl(datatype="b32", prefix="r", num_regs=1)],
            body=[
                ptx.Instruction(
                    predicate=None,
                    opcode="ld.param.u32",
                    operands=[
                        ptx.Register(prefix="r", idx=0),
                        ptx.MemoryRef(base=ptx.ParamRef(name="p0"), offset=0),
                    ],
                ),
                ptx.Label(name="L0"),
                ptx.Instruction(
                    predicate=None,
                    opcode="add.s32",
                    operands=[
                        ptx.Register(prefix="r", idx=0),
                        ptx.Register(prefix="r", idx=0),
                        ptx.Register(prefix="r", idx=0),
                    ],
                ),
                ptx.Branch(
                    predicate=None, is_uniform=False, target=ptx.Label(name="L0")
                ),
            ],
        ),
    )

    gen = CudaGen()
    module = ptx.Module(statements=[entry])
    cuda_module = gen.run(module)
    kernel = cuda_module.kernels[0]
    assert kernel.name == "kernel"
    assert cuda_module.global_vars == []

    # Params are arguments, not var_decls
    assert kernel.arguments[0][0].name == "p0"
    assert kernel.var_decls  # registers allocated
    # Special registers are not in var_decls but available in reg_map
    assert ptx.Register(prefix="ctaid.x", idx=None) in gen.reg_map
    assert gen.reg_map[ptx.Register(prefix="ctaid.x", idx=None)].name == "blockIdx.x"
    # Body contains Load, Label, InlineAsm, CudaBranch in order
    assert isinstance(kernel.body[0], Load)
    assert isinstance(kernel.body[1], CudaLabel)
    assert isinstance(kernel.body[2], InlineAsm)
    assert isinstance(kernel.body[3], CudaBranch)


def test_special_registers_seeded_and_usable_in_inline_asm():
    gen = CudaGen()
    entry = ptx.EntryDirective(
        name="kernel",
        params=[],
        directives=[],
        body=ptx.ScopedBlock(
            registers=[ptx.RegisterDecl(datatype="b32", prefix="r", num_regs=1)],
            body=[
                ptx.Instruction(
                    predicate=None,
                    opcode="add.s32",
                    operands=[
                        ptx.Register(prefix="r", idx=0),
                        ptx.Register(prefix="ctaid.x", idx=None),
                        ptx.Register(prefix="tid.y", idx=None),
                    ],
                )
            ],
        ),
    )
    gdecl = ptx.MemoryDecl(
        alignment=None,
        datatype="u32",
        name="g0",
        num_elements=1,
        memory_type=ptx.MemoryType.Global,
    )
    module = ptx.Module(statements=[gdecl, entry])
    cuda_module = gen.run(module)
    kernel = cuda_module.kernels[0]
    assert cuda_module.global_vars == [gdecl]
    assert kernel.var_decls  # includes r0
    # Ensure all special regs are seeded
    for reg_prefix, name in [
        ("ctaid.x", "blockIdx.x"),
        ("ctaid.y", "blockIdx.y"),
        ("ctaid.z", "blockIdx.z"),
        ("tid.x", "threadIdx.x"),
        ("tid.y", "threadIdx.y"),
        ("tid.z", "threadIdx.z"),
    ]:
        reg = ptx.Register(prefix=reg_prefix, idx=None)
        assert reg in gen.reg_map
        assert gen.reg_map[reg].name == name
    # Body should contain the lowered inline asm using the seeded specials as arguments
    asm = next(item for item in kernel.body if isinstance(item, InlineAsm))
    arg_names = [v.name for v in asm.arguments]
    assert "blockIdx.x" in arg_names
    assert "threadIdx.y" in arg_names
