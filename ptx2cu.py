"""
CLI entrypoint for parsing PTX. Currently accepts a single positional PTX file.
"""

import argparse
from pathlib import Path

from parser import parse_module
from cudagen import CudaGen


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse a PTX file into a Module AST")
    parser.add_argument("ptx_file", type=Path, help="Path to PTX file to parse")
    parser.add_argument("--entry", required=True, help="Name of the PTX entry directive to translate")
    parser.add_argument(
        "--kernel-name",
        required=True,
        help="New CUDA kernel name to emit",
    )
    args = parser.parse_args()

    text = args.ptx_file.read_text()
    module = parse_module(text)
    entry = next((stmt for stmt in module.statements if getattr(stmt, "name", None) == args.entry), None)
    if entry is None:
        raise SystemExit(f"Entry {args.entry!r} not found in module")

    gen = CudaGen()
    cuda_kernel = gen.run(entry)

    # For now, just print a summary count and kernel info
    print(
        f"Parsed module with {len(module.statements)} statements from {args.ptx_file}; "
        f"entry={args.entry} -> kernel={args.kernel_name}; "
        f"lowered kernel has {len(cuda_kernel.arguments)} args, "
        f"{len(cuda_kernel.var_decls)} vars, {len(cuda_kernel.body)} items"
    )


if __name__ == "__main__":
    main()
