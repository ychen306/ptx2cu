from __future__ import annotations

from typing import List, Tuple, Dict, Set, Optional

from .. import types as cg_types
from . import ir


def _is_label(item: cg_types.KernelItem) -> bool:
    return isinstance(item, cg_types.CudaLabel)


def _is_terminator(item: cg_types.KernelItem) -> bool:
    return isinstance(item, (cg_types.CudaBranch, cg_types.Return))


def discover_block_boundaries(kernel: cg_types.CudaKernel) -> List[Tuple[int, int]]:
    """
    Discover basic block boundaries in a linear CudaKernel body.

    Returns a list of (begin, end) index pairs, both inclusive, into kernel.body.
    """

    items = kernel.body
    n = len(items)

    # Pass 1: collect label positions and branch targets/returns.
    label_pos: Dict[str, int] = {}
    for idx, item in enumerate(items):
        if isinstance(item, cg_types.CudaLabel):
            label_pos[item.name] = idx

    branch_targets: Set[int] = set()
    branch_idxs: Set[int] = set()
    return_idxs: Set[int] = set()

    for idx, item in enumerate(items):
        if isinstance(item, cg_types.CudaBranch):
            branch_idxs.add(idx)
            tgt_idx = label_pos.get(item.target.name)
            if tgt_idx is not None:
                branch_targets.add(tgt_idx)
        elif isinstance(item, cg_types.Return):
            return_idxs.add(idx)

    # Pass 2: derive starts and ends (inclusive end).
    starts: Set[int] = {0}
    ends: Set[int] = set()

    for tgt in branch_targets:
        starts.add(tgt)
        if tgt > 0:
            ends.add(tgt - 1)

    for ret_idx in return_idxs:
        ends.add(ret_idx)
        if ret_idx + 1 < n:
            starts.add(ret_idx + 1)

    for br_idx in branch_idxs:
        ends.add(br_idx)
        if br_idx + 1 < n:
            starts.add(br_idx + 1)

    ends.add(n - 1)

    cuts = set(starts)
    cuts.update(e + 1 for e in ends if e + 1 <= n)
    cuts.add(n)

    ordered = sorted(cuts)
    return [(ordered[i], ordered[i + 1] - 1) for i in range(len(ordered) - 1)]


def build_opt_kernel(kernel: cg_types.CudaKernel) -> ir.OptKernel:
    """
    Build an OptKernel with explicit basic blocks and terminators from a CudaKernel.
    """

    body = kernel.body
    boundaries = discover_block_boundaries(kernel)

    # Map label name -> block id (first index of its range).
    label_to_block: Dict[str, str] = {}
    blocks: List[ir.OptBasicBlock] = []

    # First pass: build blocks with placeholder targets.
    placeholder_targets: List[Optional[str]] = []

    for begin, end in boundaries:
        items = body[begin : end + 1]
        block_id = f"B{begin}"

        if items and isinstance(items[0], cg_types.CudaLabel):
            label_to_block[items[0].name] = block_id

        instrs: List[ir.OptInstruction] = []
        branch_target: Optional[str] = None
        terminator: Optional[ir.OptTerminator] = None

        for it in items:
            if isinstance(it, cg_types.Assignment):
                instrs.append(it)
            elif isinstance(it, cg_types.Load):
                instrs.append(it)
            elif isinstance(it, cg_types.Store):
                instrs.append(it)
            elif isinstance(it, cg_types.CudaBranch):
                branch_target = it.target.name
                if it.cond is None:
                    terminator = ir.OptBranch(target=None)  # to be resolved
                else:
                    terminator = ir.OptCondBranch(condition=it.cond, true_target=None, false_target=None)  # type: ignore[arg-type]
            elif isinstance(it, cg_types.Return):
                terminator = ir.Return()
            elif isinstance(it, cg_types.CudaLabel):
                continue
            else:
                raise TypeError(f"Unsupported kernel item in CFG construction: {type(it)}")

        if terminator is None:
            raise ValueError(f"Missing terminator in block starting at {begin}")

        blocks.append(ir.OptBasicBlock(id=block_id, instructions=instrs, terminator=terminator))
        placeholder_targets.append(branch_target)

    id_to_block = {blk.id: blk for blk in blocks}

    # Second pass: resolve branch targets.
    for idx, blk in enumerate(blocks):
        term = blk.terminator
        if isinstance(term, ir.OptBranch):
            target_label = placeholder_targets[idx]
            if target_label is None:
                raise ValueError(f"Unconditional branch without target in block {blk.id}")
            target_block_id = label_to_block.get(target_label)
            if target_block_id is None:
                raise KeyError(f"Unknown branch target label {target_label}")
            blk.terminator = ir.OptBranch(target=id_to_block[target_block_id])
        elif isinstance(term, ir.OptCondBranch):
            target_label = placeholder_targets[idx]
            if target_label is None:
                raise ValueError(f"Conditional branch without target in block {blk.id}")
            true_block_id = label_to_block.get(target_label)
            if true_block_id is None:
                raise KeyError(f"Unknown branch target label {target_label}")
            false_block = blocks[idx + 1] if idx + 1 < len(blocks) else None
            if false_block is None:
                raise ValueError(f"Missing fallthrough block for conditional branch in {blk.id}")
            blk.terminator = ir.OptCondBranch(
                condition=term.condition,
                true_target=id_to_block[true_block_id],
                false_target=false_block,
            )

    if not blocks:
        raise ValueError("Kernel has no blocks")

    return ir.OptKernel(
        name=kernel.name,
        arguments=list(kernel.arguments),
        var_decls=list(kernel.var_decls),
        entry=blocks[0],
    )
