from __future__ import annotations

import _bootstrap  # noqa: F401  (puts the package root on the import path)

import argparse

import torch

from comparators.prac import PRACCompressor, PRACConfig
from stacr.compressor import STACRCompressor, STACRConfig
from stacr.metrics import cosine_similarity, relative_mse


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--rows", type=int, default=64)
    parser.add_argument("--features", type=int, default=512)
    args = parser.parse_args()
    x = torch.randn(args.rows, args.features)
    stacr = STACRCompressor(
        STACRConfig(rank=args.rank, residual_budget=0.01, min_features=1, min_numel=1)
    )
    packed = stacr.pack(x)
    x_stacr = stacr.unpack(packed)
    prac = PRACCompressor(PRACConfig(rank=args.rank)).reconstruct(x)
    print(
        {
            "stacr_cosine": float(cosine_similarity(x, x_stacr)),
            "stacr_relative_mse": float(relative_mse(x, x_stacr)),
            "prac_cosine": float(cosine_similarity(x, prac)),
            "prac_relative_mse": float(relative_mse(x, prac)),
        }
    )


if __name__ == "__main__":
    main()
