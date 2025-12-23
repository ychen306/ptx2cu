import ptx
from emission.kernel import declare_kernel


def test_declare_kernel_scalar_and_array():
    entry = ptx.EntryDirective(
        name="_Z6kernelPi",
        params=[
            ptx.MemoryDecl(
                alignment=None,
                datatype="u32",
                name="p0",
                num_elements=1,
                memory_type=ptx.MemoryType.Param,
            ),
            ptx.MemoryDecl(
                alignment=None,
                datatype="f16",
                name="p1",
                num_elements=4,
                memory_type=ptx.MemoryType.Param,
            ),
        ],
        directives=[],
        body=[],
    )
    structs, decl = declare_kernel(entry, kernel_name="kernel")
    assert decl == "__global__ void kernel(unsigned int p0, Param_f16_x_4 p1)"
    assert structs == ["struct Param_f16_x_4 { __half buf[4]; };"]
