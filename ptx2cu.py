"""
CLI entrypoint for parsing PTX. Currently accepts a single positional PTX file.
"""

import argparse
from pathlib import Path

from parser import parse_module


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse a PTX file into a Module AST")
    parser.add_argument("ptx_file", type=Path, help="Path to PTX file to parse")
    args = parser.parse_args()

    text = args.ptx_file.read_text()
    module = parse_module(text)
    # For now, just print a summary count
    print(f"Parsed module with {len(module.statements)} statements from {args.ptx_file}")


if __name__ == "__main__":
    main()
