from cudagen.types import Var
from emission.local import declare_local


def test_declare_local_ints_and_predicate():
    assert declare_local(Var("a", 32, False)) == "unsigned int a;"
    assert declare_local(Var("b", 64, False)) == "unsigned long long b;"
    assert declare_local(Var("p", 32, False, True)) == "unsigned int p;"


def test_declare_local_floats():
    assert declare_local(Var("f", 32, True)) == "float f;"
    assert declare_local(Var("d", 64, True)) == "double d;"
    assert declare_local(Var("h", 16, True)) == "__half h;"
