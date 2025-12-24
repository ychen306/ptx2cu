from cudagen.types import Var, CudaType, CudaTypeId
from emission.local import declare_local


def test_declare_local_ints_and_predicate():
    assert declare_local(Var("a", CudaType(32, CudaTypeId.Unsigned))) == "unsigned int a;"
    assert (
        declare_local(Var("b", CudaType(64, CudaTypeId.Unsigned)))
        == "unsigned long long b;"
    )
    assert (
        declare_local(Var("p", CudaType(32, CudaTypeId.Unsigned, True)))
        == "unsigned int p;"
    )


def test_declare_local_floats():
    assert declare_local(Var("f", CudaType(32, CudaTypeId.Float))) == "float f;"
    assert declare_local(Var("d", CudaType(64, CudaTypeId.Float))) == "double d;"
    assert declare_local(Var("h", CudaType(16, CudaTypeId.Float))) == "__half h;"
