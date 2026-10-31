"""Infrastructure checks; scientific CLI commands arrive in M1 and later."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import sys

from dual_target import __version__


def doctor() -> dict[str, str]:
    """Report installed dependencies without asserting scientific readiness."""
    packages = ("numpy", "pandas", "pyarrow", "rdkit")
    return {
        "status": "ok",
        "package_version": __version__,
        "python_version": sys.version.split()[0],
        **{package: importlib.metadata.version(package) for package in packages},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Dual-target scientific utilities")
    command = parser.add_subparsers(dest="command", required=True)
    command.add_parser("doctor", help="Check Python runtime and core dependencies")
    args = parser.parse_args()

    if args.command == "doctor":
        print(json.dumps(doctor(), indent=2, sort_keys=True))
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
