from __future__ import annotations

import pytest

from cudagen import types as cg_types
from cudagen.opt.cfg import discover_block_boundaries


def _label(name: str) -> cg_types.CudaLabel:
    return cg_types.CudaLabel(name=name)


def _branch(label: str) -> cg_types.CudaBranch:
    return cg_types.CudaBranch(cond=None, target=cg_types.CudaLabel(name=label))


def _cond_branch(label: str, cond: cg_types.Expr) -> cg_types.CudaBranch:
    # For boundary discovery we only care that it's a CudaBranch; reuse the same type.
    return cg_types.CudaBranch(cond=cond, target=cg_types.CudaLabel(name=label))


def _ret() -> cg_types.Return:
    return cg_types.Return()


def _assign() -> cg_types.Assignment:
    # Minimal placeholder assignment for testing; actual fields are unused by boundary finder.
    dummy_var = cg_types.Var(
        name="r1", ty=cg_types.CudaType(bitwidth=32, type_id=cg_types.CudaTypeId.Unsigned)
    )
    return cg_types.Assignment(lhs=dummy_var, rhs=dummy_var)


def _kernel(items: list[cg_types.KernelItem]) -> cg_types.CudaKernel:
    return cg_types.CudaKernel(
        name="k",
        arguments=[],
        var_decls=[],
        body=items,
    )


def test_single_basic_block_no_terminator():
    k = _kernel([_assign(), _assign()])
    assert discover_block_boundaries(k) == [(0, 1)]


def test_label_splits_block():
    k = _kernel([_assign(), _label("L1"), _assign()])
    assert discover_block_boundaries(k) == [(0, 2)]


def test_branch_ends_block():
    k = _kernel([_assign(), _branch("L1"), _assign()])
    assert discover_block_boundaries(k) == [(0, 1), (2, 2)]


def test_return_ends_block():
    k = _kernel([_assign(), _ret(), _assign()])
    assert discover_block_boundaries(k) == [(0, 1), (2, 2)]


def test_consecutive_labels_allow_empty_block():
    k = _kernel([_label("L0"), _label("L1"), _assign()])
    assert discover_block_boundaries(k) == [(0, 2)]


def test_label_after_terminator_starts_new_block():
    k = _kernel([_assign(), _branch("L1"), _label("L1"), _assign()])
    assert discover_block_boundaries(k) == [(0, 1), (2, 3)]


def test_diamond_shape_boundaries():
    # entry -> then/else -> join
    items: list[cg_types.KernelItem] = [
        _label("entry"),  # 0
        _assign(),  # 1
        _cond_branch("then", cond=_assign().lhs),  # 2
        _label("else"),  # 3
        _assign(),  # 4
        _branch("join"),  # 5
        _label("then"),  # 6
        _assign(),  # 7
        _branch("join"),  # 8
        _label("join"),  # 9
        _ret(),  # 10
    ]
    k = _kernel(items)
    assert discover_block_boundaries(k) == [(0, 2), (3, 5), (6, 8), (9, 10)]
