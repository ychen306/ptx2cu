Plan: CudaKernel → OptKernel
================================

Inputs/assumptions
------------------
- Start from a `CudaKernel` (name, arguments, var_decls, body of kernel items including labels/branches/returns).
- Opt IR supports only `Assignment | Load | Store` inside blocks and explicit terminators (`OptBranch/OptCondBranch/Return`).

Pass 1: Identify block boundaries
---------------------------------
- Seed an entry block at the start of the kernel body.
- Start a new block at each label and immediately after any terminator-like item (branch/return).
- Allow empty blocks (label-only), but every block must end with a terminator.

Pass 2: Collect instructions
----------------------------
- For each block, accumulate linear kernel items until the next boundary.
- Keep only `Assignment/Load/Store` as block instructions; decide how to handle/forbid other item kinds (InlineAsm, etc.)—either extend supported set or raise for now.

Pass 3: Assign terminators
--------------------------
- For branch items: create `OptBranch` (unconditional) or `OptCondBranch` (conditional) with placeholder targets keyed by label names.
- For `Return` items: create `Return`.
- If a block ends without an explicit terminator, insert an implicit fallthrough `OptBranch` to the next block; the final block without a terminator should end in `Return`.

Pass 4: Resolve targets
-----------------------
- Build a label→block map (block IDs can reuse label names or generated IDs).
- Replace placeholder label references in terminators with actual `OptBasicBlock` objects.

Assemble OptKernel
------------------
- Use the original kernel `name`, `arguments`, `var_decls`.
- Set `entry` to the first block created.
- Ensure each block has a unique `id` (label-derived when available, otherwise generated).
