import ptx
from cudagen.types import CudaKernel, Var, InlineAsm, Load, CudaBranch, CudaLabel
from emission.kernel_body import emit_kernel
from emission.param import get_type_decl_for_param
from emission.branch import emit_branch_string


def test_emit_kernel_simple():
    arg_decl = ptx.MemoryDecl(
        alignment=None,
        datatype="u32",
        name="p0",
        num_elements=1,
        memory_type=ptx.MemoryType.Param,
    )
    kernel = CudaKernel(
        name="k",
        arguments=[(Var("p0", 32, False), arg_decl)],
        var_decls=[Var("r0", 32, False)],
        body=[
            Load(
                bitwidth=32,
                is_float=False,
                dst=Var("r0", 32, False),
                src=Var("p0", 32, False),
                offset=0,
            ),
            CudaLabel(name="L0"),
            InlineAsm(
                template="add.s32 %0, %1, %2;",
                arguments=[
                    Var("r0", 32, False),
                    Var("r0", 32, False),
                    Var("r0", 32, False),
                ],
                outputs=[Var("r0", 32, False)],
            ),
            CudaBranch(cond=None, target=CudaLabel(name="L0")),
        ],
    )
    out = emit_kernel(kernel)
    expected = (
        'extern "C" __global__ void k(unsigned int p0)\n'
        "{\n"
        "  unsigned int r0;\n"
        "  r0 = reinterpret_cast<unsigned int*>(&p0)[0];\n"
        "L0:\n"
        '  asm volatile("add.s32 %0, %1, %2;" : "+r"(r0) :  : );\n'
        "  goto L0;\n"
        "}"
    )
    assert out == expected
