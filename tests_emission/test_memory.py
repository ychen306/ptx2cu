import ptx
from emission.memory import declare_memory
from cudagen.types import MemoryDecl


def test_declare_global_memory():
    mem = MemoryDecl(alignment=None, datatype="u32", name="g0", num_elements=4, memory_type=ptx.MemoryType.Global)
    assert declare_memory(mem) == 'extern "C" __device__ unsigned int g0[4];'


def test_declare_shared_extern_memory_with_align():
    mem = MemoryDecl(alignment=16, datatype="u8", name="shared_mem", num_elements=0, memory_type=ptx.MemoryType.Shared)
    assert declare_memory(mem) == 'extern "C" extern __shared__ __align__(16) unsigned char shared_mem[];'
