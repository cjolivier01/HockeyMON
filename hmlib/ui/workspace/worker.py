"""Subprocess entry points for potentially long desktop workflows."""

from __future__ import annotations

import argparse
from pathlib import Path

from .files import read_json, write_json
from .highlights import render_plan


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=["highlights", "concatenate"])
    parser.add_argument("request", type=Path)
    args = parser.parse_args()
    request = read_json(args.request)
    if args.operation == "concatenate":
        from .concatenate import concatenate

        result = concatenate(request, args.request.parent)
    else:
        result = render_plan(
            request["plan"],
            Path(request["game"]),
            Path(request["output"]),
            preview=request["preview"],
        )
    write_json(args.request.parent / "result.json", result)


if __name__ == "__main__":
    main()
