import ptx
import pytest
from cudagen import CudaBranch, CudaLabel, Var, emit_branch


def test_emit_branch_unconditional():
    br = ptx.Branch(predicate=None, is_uniform=False, target=ptx.Label(name="L1"))
    regmap = {}
    out = emit_branch(br, regmap)
    assert out == CudaBranch(cond=None, target=CudaLabel(name="L1"))


def test_emit_branch_predicated():
    pred = ptx.Register(prefix="p", idx=1)
    br = ptx.Branch(predicate=pred, is_uniform=True, target=ptx.Label(name="L2"))
    regmap = {
        pred: Var("p1", 32, False, True)
    }
    out = emit_branch(br, regmap)
    assert out == CudaBranch(cond=Var("p1", 32, False, True), target=CudaLabel(name="L2"))


def test_emit_branch_predicated_without_percent():
    # Predicate registers can be parsed without an explicit index (idx=None)
    pred = ptx.Register(prefix="p", idx=None)
    br = ptx.Branch(predicate=pred, is_uniform=False, target=ptx.Label(name="L3"))
    regmap = {
        pred: Var("p", 32, False, True)
    }
    out = emit_branch(br, regmap)
    assert out == CudaBranch(cond=Var("p", 32, False, True), target=CudaLabel(name="L3"))


def test_emit_branch_uniform_ignored():
    # Uniformity flag should not affect lowering result
    br = ptx.Branch(predicate=None, is_uniform=True, target=ptx.Label(name="L4"))
    out = emit_branch(br, {})
    assert out == CudaBranch(cond=None, target=CudaLabel(name="L4"))


def test_emit_branch_missing_predicate_mapping():
    pred = ptx.Register(prefix="p", idx=0)
    br = ptx.Branch(predicate=pred, is_uniform=False, target=ptx.Label(name="L3"))
    with pytest.raises(ValueError):
        emit_branch(br, {})
