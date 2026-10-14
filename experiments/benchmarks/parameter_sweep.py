"""Parameter sweep."""

from __future__ import annotations
import torch
import sys
from pathlib import Path
from .common import RESULTS, emit, run_config

SEEDS = [42, 123, 456]
(MODEL, BS, SEQ, STEPS) = ("gpt2-medium", 8, 1024, 300)
BASE = {"residual_budget": 0.005, "residual_bits": 8, "min_features": 256, "chunk_rows": 512}
SURFACES = [("mlp", "mlp"), ("mlp+attention", "mlp+attention"), ("attention", "attention")]
RANKS = [8, 32, 128]


def main():
    out = RESULTS / "r3_recipe.jsonl"
    base = None
    for seed in SEEDS:
        r = run_config(
            model_id=MODEL, bs=BS, seq=SEQ, steps=STEPS, seed=seed, method="baseline", tag="recipe"
        )
        if base is None:
            base = r["peak_bytes"]
        emit(r, out)
    for sname, surface in SURFACES:
        for rank in RANKS:
            for seed in SEEDS:
                try:
                    r = run_config(
                        model_id=MODEL,
                        bs=BS,
                        seq=SEQ,
                        steps=STEPS,
                        seed=seed,
                        method=f"{sname}_r{rank}",
                        stacr_cfg=dict(BASE, surface=surface, rank=rank),
                        baseline_peak_bytes=base,
                        tag="recipe",
                    )
                except Exception as exc:
                    emit(
                        {
                            "tag": "recipe",
                            "method": f"{sname}_r{rank}",
                            "seed": seed,
                            "status": "error",
                            "error": str(exc)[:200],
                        },
                        out,
                    )
                    continue
                r["measured_saving_pct"] = 100.0 * (base - r["peak_bytes"]) / base
                emit(r, out)


if __name__ == "__main__":
    main()
