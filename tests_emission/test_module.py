import ptx
from cudagen.types import CudaModule, CudaKernel, Var
from emission.module import emit_cuda_module


def test_emit_cuda_module_with_globals_and_kernel():
    gdecl = ptx.MemoryDecl(
        alignment=None,
        datatype="u32",
        name="g0",
        num_elements=4,
        memory_type=ptx.MemoryType.Global,
    )
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
        var_decls=[],
        body=[],
    )
    mod = CudaModule(global_vars=[gdecl], kernels=[kernel])
    out = emit_cuda_module(mod)
    assert "#include <cuda_fp16.h>" in out.splitlines()[0]
    assert "__device__ unsigned int g0[4];" in out
    assert "__global__ void k(unsigned int p0)" in out
