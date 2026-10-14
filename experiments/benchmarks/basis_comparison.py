"""Basis comparison."""

from __future__ import annotations
import torch
import sys
from pathlib import Path
from .common import RESULTS, emit, run_config, gradient_fidelity

SEEDS = [42, 123, 456]
(MODEL, BS, SEQ, STEPS) = ("gpt2-medium", 8, 1024, 300)
BASE = {"rank": 8, "residual_bits": 8, "surface": "mlp", "min_features": 256, "chunk_rows": 512}
KINDS = ["stacr", "lowrank", "untracked", "oasis", "prac"]


def main():
    out = RESULTS / "e3_baselines.jsonl"
    base_peak = None
    for seed in SEEDS:
        r = run_config(
            model_id=MODEL, bs=BS, seq=SEQ, steps=STEPS, seed=seed, method="baseline", tag="cmp"
        )
        if base_peak is None:
            base_peak = r["peak_bytes"]
        emit(r, out)
    for kind in KINDS:
        for seed in SEEDS:
            try:
                r = run_config(
                    model_id=MODEL,
                    bs=BS,
                    seq=SEQ,
                    steps=STEPS,
                    seed=seed,
                    method=f"cmp_{kind}",
                    stacr_cfg=dict(BASE),
                    comparator=kind,
                    baseline_peak_bytes=base_peak,
                    tag="cmp",
                )
            except Exception as exc:
                emit(
                    {
                        "tag": "cmp",
                        "method": f"cmp_{kind}",
                        "seed": seed,
                        "status": "error",
                        "error": f"{type(exc).__name__}: {exc}"[:300],
                    },
                    out,
                )
                continue
            r["measured_saving_pct"] = 100.0 * (base_peak - r["peak_bytes"]) / base_peak
            emit(r, out)
    fid = RESULTS / "e3_fidelity.jsonl"
    for kind in KINDS:
        try:
            emit(
                gradient_fidelity(
                    model_id=MODEL,
                    bs=2,
                    seq=1024,
                    seed=42,
                    steps=12,
                    stacr_cfg=dict(BASE),
                    comparator=kind,
                    tag="cmp_fid",
                ),
                fid,
            )
        except Exception as exc:
            emit(
                {
                    "tag": "cmp_fid",
                    "comparator": kind,
                    "status": "error",
                    "error": f"{type(exc).__name__}: {exc}"[:300],
                },
                fid,
            )


if __name__ == "__main__":
    main()
