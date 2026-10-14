"""Training 2000."""

from __future__ import annotations
import torch
import sys
from pathlib import Path
from .common import RESULTS, emit, run_config

SEEDS = [42, 123, 456]
STEPS = 2000
CFG8 = {
    "rank": 8,
    "residual_budget": 0.005,
    "residual_bits": 8,
    "surface": "mlp",
    "min_features": 256,
    "chunk_rows": 512,
}
TURBO = dict(CFG8, turbo_after_step=2)
ROWS = [
    ("baseline", {}),
    ("stacr_r8", dict(stacr_cfg=CFG8)),
    ("turbo_fast", dict(stacr_cfg=TURBO)),
    ("ckpt_full", dict(ckpt_every=1)),
    ("ckpt_k2", dict(ckpt_every=2)),
    (
        "ckpt_k2_plus_stacr",
        dict(ckpt_every=2, ckpt_offset=0, stacr_cfg=CFG8, stacr_block_pred=lambda i: i % 2 == 1),
    ),
]


def main():
    out = RESULTS / "q1_longhorizon.jsonl"
    base = None
    for method, extra in ROWS:
        for seed in SEEDS:
            try:
                r = run_config(
                    model_id="gpt2-medium",
                    bs=8,
                    seq=1024,
                    steps=STEPS,
                    seed=seed,
                    method=method,
                    eval_batches=32,
                    measure_from=12,
                    baseline_peak_bytes=base,
                    tag="lh_gpt2m",
                    **extra,
                )
            except Exception as exc:
                emit(
                    {
                        "tag": "lh_gpt2m",
                        "method": method,
                        "seed": seed,
                        "status": "error",
                        "error": f"{type(exc).__name__}: {exc}"[:300],
                    },
                    out,
                )
                continue
            if method == "baseline" and base is None:
                base = r["peak_bytes"]
            emit(r, out)


if __name__ == "__main__":
    main()
