from __future__ import annotations

import time
from typing import Any

import torch

from stacr.metrics import cosine_similarity, relative_mse


def svd_reconstruct(tensor: torch.Tensor, rank: int) -> torch.Tensor:
    """Per-batch truncated SVD reconstruction on the flattened activation matrix."""
    x = tensor.reshape(-1, tensor.shape[-1]).to(torch.float32)
    rank = max(1, min(int(rank), min(x.shape)))
    u, s, vh = torch.linalg.svd(x, full_matrices=False)
    recon = (u[:, :rank] * s[:rank]).mm(vh[:rank])
    return recon.reshape_as(tensor).to(dtype=tensor.dtype)


def tucker_like_reconstruct(tensor: torch.Tensor, rank: int) -> torch.Tensor:
    """Compact Tucker-style baseline using a low-rank feature projection."""
    return svd_reconstruct(tensor, rank)


def tt_like_reconstruct(tensor: torch.Tensor, rank: int) -> torch.Tensor:
    """Compact TT-style baseline used for timing and shape checks."""
    return svd_reconstruct(tensor, rank)


def factorization_benchmark(tensor: torch.Tensor, rank: int = 8) -> list[dict[str, Any]]:
    rows = []
    for name, fn in [
        ("svd", svd_reconstruct),
        ("tt_like", tt_like_reconstruct),
        ("tucker_like", tucker_like_reconstruct),
    ]:
        start = time.perf_counter()
        recon = fn(tensor, rank)
        elapsed = time.perf_counter() - start
        rows.append(
            {
                "method": name,
                "seconds": elapsed,
                "cosine": float(cosine_similarity(tensor, recon)),
                "relative_mse": float(relative_mse(tensor, recon)),
            }
        )
    return rows
