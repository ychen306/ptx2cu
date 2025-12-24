from cudagen.types import Var, CudaType
from emission.local import declare_local


def test_declare_local_ints_and_predicate():
    assert declare_local(Var("a", CudaType(32, False))) == "unsigned int a;"
    assert declare_local(Var("b", CudaType(64, False))) == "unsigned long long b;"
    assert declare_local(Var("p", CudaType(32, False, True))) == "unsigned int p;"


def test_declare_local_floats():
    assert declare_local(Var("f", CudaType(32, True))) == "float f;"
    assert declare_local(Var("d", CudaType(64, True))) == "double d;"
    assert declare_local(Var("h", CudaType(16, True))) == "__half h;"
