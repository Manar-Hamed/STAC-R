from __future__ import annotations

import _bootstrap  # noqa: F401  (puts the package root on the import path)

import argparse

from stacr.training import run_config
from stacr.utils import load_yaml, write_csv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", default="outputs/result.csv")
    parser.add_argument("--steps", type=int, default=None)
    args = parser.parse_args()
    cfg = load_yaml(args.config)
    if args.steps is not None:
        cfg["steps"] = args.steps
    row = run_config(cfg)
    write_csv(args.output, [row])
    print(row)


if __name__ == "__main__":
    main()
