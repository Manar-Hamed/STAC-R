"""Pythia."""

from __future__ import annotations
import torch
import sys
from pathlib import Path
from .common import RESULTS, emit, run_config

SEEDS = [42, 123, 456]
(MODEL, BS, SEQ, STEPS) = ("EleutherAI/pythia-1.4b", 4, 1024, 300)
CFG = {
    "rank": 8,
    "residual_budget": 0.005,
    "residual_bits": 8,
    "surface": "mlp",
    "min_features": 256,
    "chunk_rows": 512,
}
ROWS = [
    ("baseline", {}),
    ("stacr_r8", dict(stacr_cfg=CFG)),
    ("turbo_fast", dict(stacr_cfg=dict(CFG, turbo_after_step=2))),
    ("ckpt_full", dict(ckpt_every=1)),
    ("ckpt_k2", dict(ckpt_every=2)),
    (
        "ckpt_k2_plus_stacr",
        dict(ckpt_every=2, ckpt_offset=0, stacr_cfg=CFG, stacr_block_pred=lambda i: i % 2 == 1),
    ),
]


def main():
    out = RESULTS / "a_pythia.jsonl"
    bp = bt = None
    for method, extra in ROWS:
        for seed in SEEDS:
            try:
                r = run_config(
                    model_id=MODEL,
                    bs=BS,
                    seq=SEQ,
                    steps=STEPS,
                    seed=seed,
                    method=method,
                    baseline_peak_bytes=bp,
                    tag="pythia",
                    **extra,
                )
            except Exception as exc:
                emit(
                    {
                        "tag": "pythia",
                        "method": method,
                        "seed": seed,
                        "status": "error",
                        "error": f"{type(exc).__name__}: {exc}"[:250],
                    },
                    out,
                )
                continue
            if method == "baseline" and bp is None:
                (bp, bt) = (r["peak_bytes"], r["time_s"])
            if bt:
                r["time_x"] = r["time_s"] / bt
                r["measured_saving_pct"] = 100.0 * (bp - r["peak_bytes"]) / bp
            emit(r, out)
            print(
                f"  >>> {method:<20} {r.get('time_x', 1.0):.3f}x  mem {r.get('measured_saving_pct', 0.0):+.2f}%  PPL {r['val_ppl']:.4f}",
                flush=True,
            )


if __name__ == "__main__":
    main()
