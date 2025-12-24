from cudagen.types import CudaBranch, CudaLabel, Var, CudaType, CudaTypeId
from emission.branch import emit_branch_string


def test_emit_branch_string_unconditional():
    br = CudaBranch(cond=None, target=CudaLabel(name="L1"))
    assert emit_branch_string(br) == "goto L1;"


def test_emit_branch_string_conditional():
    br = CudaBranch(
        cond=Var("p1", CudaType(32, CudaTypeId.Unsigned, True)),
        target=CudaLabel(name="L2"),
    )
    assert emit_branch_string(br) == "if (p1 != 0) goto L2;"
