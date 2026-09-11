"""Inspect or merge portable telemetry recordings."""

import argparse
import json

from hmlib.telemetry.database import discover_runs, merge_databases


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    inspect = commands.add_parser("inspect")
    inspect.add_argument("inputs", nargs="+")
    merge = commands.add_parser("merge")
    merge.add_argument("destination")
    merge.add_argument("inputs", nargs="+")
    args = parser.parse_args(argv)
    result = (
        merge_databases(args.destination, args.inputs)
        if args.command == "merge"
        else discover_runs(args.inputs)
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
