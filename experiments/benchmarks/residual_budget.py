"""Residual budget."""

from __future__ import annotations
import torch
import sys
from pathlib import Path
from .common import RESULTS, emit, run_config, gradient_fidelity

SEEDS = [42, 123, 456]
(MODEL, BS, SEQ, STEPS) = ("gpt2-medium", 8, 1024, 300)
BASE = {"rank": 8, "residual_bits": 8, "surface": "mlp", "min_features": 256, "chunk_rows": 512}


def main():
    out = RESULTS / "e6_residual.jsonl"
    base_peak = None
    for seed in SEEDS:
        r = run_config(
            model_id=MODEL, bs=BS, seq=SEQ, steps=STEPS, seed=seed, method="baseline", tag="resid"
        )
        if base_peak is None:
            base_peak = r["peak_bytes"]
        emit(r, out)
    for label, budget in [
        ("residual_on_0.5pct", 0.005),
        ("residual_off", 0.0),
        ("residual_1pct", 0.01),
    ]:
        for seed in SEEDS:
            try:
                r = run_config(
                    model_id=MODEL,
                    bs=BS,
                    seq=SEQ,
                    steps=STEPS,
                    seed=seed,
                    method=label,
                    stacr_cfg=dict(BASE, residual_budget=budget),
                    baseline_peak_bytes=base_peak,
                    tag="resid",
                )
            except Exception as exc:
                emit(
                    {
                        "tag": "resid",
                        "method": label,
                        "seed": seed,
                        "status": "error",
                        "error": str(exc)[:300],
                    },
                    out,
                )
                continue
            r["measured_saving_pct"] = 100.0 * (base_peak - r["peak_bytes"]) / base_peak
            emit(r, out)
    fid = RESULTS / "e6_residual_fidelity.jsonl"
    for rank in [8, 32, 128, 307]:
        for budget in [0.005, 0.0]:
            try:
                emit(
                    gradient_fidelity(
                        model_id=MODEL,
                        bs=2,
                        seq=1024,
                        seed=42,
                        steps=12,
                        stacr_cfg=dict(BASE, rank=rank, residual_budget=budget),
                        tag=f"resid_r{rank}_b{budget}",
                    ),
                    fid,
                )
            except Exception as exc:
                emit(
                    {"tag": f"resid_r{rank}_b{budget}", "status": "error", "error": str(exc)[:300]},
                    fid,
                )


if __name__ == "__main__":
    main()
