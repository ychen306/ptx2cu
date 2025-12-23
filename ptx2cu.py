"""
CLI entrypoint for parsing PTX. Currently accepts a single positional PTX file.
"""

import argparse
from pathlib import Path

from parser import parse_module
from cudagen import CudaGen
from emission import emit_cuda_module


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse a PTX file into a Module AST")
    parser.add_argument("ptx_file", type=Path, help="Path to PTX file to parse")
    args = parser.parse_args()

    text = args.ptx_file.read_text()
    module = parse_module(text)
    gen = CudaGen()
    cuda_module = gen.run(module)

    cuda_src = emit_cuda_module(cuda_module)

    # For now, just print the generated CUDA source
    print(cuda_src)


if __name__ == "__main__":
    main()
