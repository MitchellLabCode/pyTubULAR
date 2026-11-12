"""Command-line access to the same services intended for a future viewer."""

import argparse
import json
from pathlib import Path

from .analysis import analyze_mesh
from .grid import cylinder_mesh
from .io import load_mat_mesh, load_mesh, save_mesh


def main(argv=None):
    parser = argparse.ArgumentParser(description="TubULAR Python experimental mesh tools")
    commands = parser.add_subparsers(dest="command", required=True)
    demo = commands.add_parser("demo", help="write a synthetic cylinder mesh")
    demo.add_argument("output", type=Path)
    inspect = commands.add_parser("inspect", help="validate and summarize a mesh")
    inspect.add_argument("input", type=Path)
    inspect.add_argument("--mat-variable", help="mesh struct name in a MATLAB file")
    inspect.add_argument("--mat-length-unit", default="unknown")
    inspect.add_argument("--output", type=Path, help="write JSON to a new file")
    args = parser.parse_args(argv)
    try:
        if args.command == "demo":
            save_mesh(args.output, cylinder_mesh(length_unit="um"))
            print(str(args.output))
        else:
            suffix = args.input.suffix.lower()
            if suffix == ".mat":
                mesh = load_mat_mesh(args.input, variable=args.mat_variable,
                                     length_unit=args.mat_length_unit)
            elif suffix == ".npz":
                mesh = load_mesh(args.input)
            else:
                raise ValueError("supported mesh inputs are .mat and .npz")
            output = json.dumps(analyze_mesh(mesh), indent=2, allow_nan=False) + "\n"
            if args.output:
                with args.output.open("x") as stream:
                    stream.write(output)
            else:
                print(output, end="")
    except (OSError, ValueError, NotImplementedError) as exc:
        parser.exit(2, f"tubular: {exc}\n")
    return 0
