"""Dispatch a canonical variant config to its declared Python module."""

from __future__ import annotations

import argparse
import subprocess
import sys

import yaml

from pointforward.variants import resolve_entrypoint


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--mode", required=True, choices=("train", "infer"))
    args, remaining = parser.parse_known_args()
    with open(args.config, "r") as handle:
        config = yaml.safe_load(handle)
    module = resolve_entrypoint(config, args.mode)
    command = [sys.executable, "-m", module, "--config", args.config, *remaining]
    raise SystemExit(subprocess.run(command, check=False).returncode)


if __name__ == "__main__":
    main()