from __future__ import annotations

import _bootstrap  # noqa: F401  (puts the package root on the import path)

import argparse
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    args = parser.parse_args()
    frame = pd.read_csv(args.csv)
    print(frame.to_string(index=False))


if __name__ == "__main__":
    main()
