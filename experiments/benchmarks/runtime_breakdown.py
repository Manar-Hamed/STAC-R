"""Runtime breakdown."""

from __future__ import annotations
import torch
import sys
from pathlib import Path
from .common import RESULTS, emit, run_config

SEEDS = [42, 123, 456]
(MODEL, BS, SEQ, STEPS) = ("gpt2-medium", 8, 1024, 300)
B = {"rank": 8, "residual_budget": 0.005, "surface": "mlp", "min_features": 256, "chunk_rows": 512}
VARIANTS = [
    ("full_ui1_fp32", dict(B, residual_bits=8, update_interval=1)),
    ("full_ui10_fp32", dict(B, residual_bits=8, update_interval=10)),
    ("full_ui100_fp32", dict(B, residual_bits=8, update_interval=100)),
    ("full_ui100_fp16", dict(B, residual_bits=8, update_interval=100, basis_dtype="float16")),
    (
        "paper_cfg_frozen",
        dict(
            B,
            residual_bits=8,
            update_interval=100,
            basis_dtype="float16",
            freeze_after_warmup=True,
            warmup_steps=20,
        ),
    ),
    (
        "paper_cfg_frozen_q4",
        dict(
            B,
            residual_bits=4,
            update_interval=100,
            basis_dtype="float16",
            freeze_after_warmup=True,
            warmup_steps=20,
        ),
    ),
    ("turbo_after_2", dict(B, residual_bits=8, turbo_after_step=2)),
]


def main():
    out = RESULTS / "r4_overhead.jsonl"
    base_t = base_p = None
    for seed in SEEDS:
        r = run_config(
            model_id=MODEL, bs=BS, seq=SEQ, steps=STEPS, seed=seed, method="baseline", tag="ovh"
        )
        if base_p is None:
            (base_p, base_t) = (r["peak_bytes"], r["time_s"])
        emit(r, out)
    for label, cfg in VARIANTS:
        for seed in SEEDS:
            try:
                r = run_config(
                    model_id=MODEL,
                    bs=BS,
                    seq=SEQ,
                    steps=STEPS,
                    seed=seed,
                    method=label,
                    stacr_cfg=cfg,
                    baseline_peak_bytes=base_p,
                    tag="ovh",
                )
            except Exception as exc:
                emit(
                    {
                        "tag": "ovh",
                        "method": label,
                        "seed": seed,
                        "status": "error",
                        "error": f"{type(exc).__name__}: {exc}"[:250],
                    },
                    out,
                )
                continue
            r["time_x"] = r["time_s"] / base_t
            r["measured_saving_pct"] = 100.0 * (base_p - r["peak_bytes"]) / base_p
            emit(r, out)
            print(
                f"  >>> {label:<22} {r['time_x']:.3f}x  mem {r['measured_saving_pct']:+.2f}%  PPL {r['val_ppl']:.4f}",
                flush=True,
            )


if __name__ == "__main__":
    main()
