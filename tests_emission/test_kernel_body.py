import ptx
from cudagen.types import (
    CudaKernel,
    Var,
    InlineAsm,
    Load,
    CudaBranch,
    CudaLabel,
    CudaType,
    CudaTypeId,
)
from emission.kernel_body import emit_kernel
from emission.param import get_type_decl_for_param
from emission.branch import emit_branch_string


t32 = CudaType(32, CudaTypeId.Unsigned)


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
        arguments=[(Var("p0", t32), arg_decl)],
        var_decls=[Var("r0", t32)],
        body=[
            Load(
                ty=t32,
                dst=Var("r0", t32),
                src=Var("p0", t32),
                offset=0,
                is_param=True,
            ),
            CudaLabel(name="L0"),
            InlineAsm(
                template="add.s32 %0, %1, %2;",
                arguments=[
                    Var("r0", t32),
                    Var("r0", t32),
                    Var("r0", t32),
                ],
                outputs=[Var("r0", t32)],
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
