"""Rate distortion."""

from __future__ import annotations
import torch
import sys
from pathlib import Path
from .common import RESULTS, emit, gradient_fidelity, load_model
from comparators.activation import apply_comparator

(MODEL, BS, SEQ, STEPS) = ("gpt2-medium", 2, 1024, 8)
RANKS = [4, 8, 16, 32, 64, 128, 307]
KINDS = ["stacr", "lowrank", "oasis", "prac", "untracked"]
BASE = {"residual_bits": 8, "surface": "mlp", "min_features": 256, "chunk_rows": 512}


def payload_bytes(kind, rank, budget):
    m = load_model(MODEL, "cpu")
    cfg = dict(BASE, rank=rank, residual_budget=budget)
    layers = apply_comparator(m, kind, cfg)
    rows = BS * SEQ
    total = sum(
        (
            sum(
                (
                    v
                    for (k, v) in lay.stored_bytes(rows, 2).items()
                    if k in ("coeff", "residual", "basis")
                )
            )
            for lay in layers
        )
    )
    dense = sum((rows * lay.in_features * 2 for lay in layers))
    n = len(layers)
    del m, layers
    return (total, dense, n)


def main():
    out = RESULTS / "q3_rate_distortion.jsonl"
    for kind in KINDS:
        for rank in RANKS:
            budget = 0.005 if kind == "stacr" else 0.0
            try:
                (stored, dense, n) = payload_bytes(kind, rank, budget)
                f = gradient_fidelity(
                    model_id=MODEL,
                    bs=BS,
                    seq=SEQ,
                    seed=42,
                    steps=STEPS,
                    stacr_cfg=dict(BASE, rank=rank, residual_budget=budget),
                    comparator=kind,
                    tag=f"rd/{kind}/r{rank}",
                )
                cos = f["mean_cosine_local_wrapped"]
                f.update(
                    {
                        "stored_bytes": stored,
                        "dense_bytes": dense,
                        "n_layers": n,
                        "stored_MiB": stored / 1024**2,
                        "compression_x": dense / stored if stored else None,
                        "eps_cos": (1 - cos) / (1 + cos) if cos == cos else None,
                    }
                )
                emit(f, out)
            except Exception as exc:
                emit(
                    {
                        "tag": f"rd/{kind}/r{rank}",
                        "status": "error",
                        "error": f"{type(exc).__name__}: {exc}"[:300],
                    },
                    out,
                )


if __name__ == "__main__":
    main()
