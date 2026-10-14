from __future__ import annotations

import _bootstrap  # noqa: F401  (puts the package root on the import path)

import argparse

import torch

from comparators.per_batch_factors import factorization_benchmark


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--rows", type=int, default=64)
    parser.add_argument("--features", type=int, default=512)
    args = parser.parse_args()
    x = torch.randn(args.rows, args.features)
    for row in factorization_benchmark(x, rank=args.rank):
        print(row)


if __name__ == "__main__":
    main()
