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
    args = parser.parse_args()

    text = args.ptx_file.read_text()
    module = parse_module(text)
    gen = CudaGen()
    cuda_module = gen.run(module)

    # For now, just print a summary count and kernel info
    print(
        f"Parsed module with {len(module.statements)} statements from {args.ptx_file}; "
        f"lowered {len(cuda_module.kernels)} kernel(s)"
    )


if __name__ == "__main__":
    main()
