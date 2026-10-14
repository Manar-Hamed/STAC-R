"""Training 2000 composition."""

from __future__ import annotations
import torch
import sys
from pathlib import Path
from .common import RESULTS, emit, run_config

SEEDS = [42, 123, 456]
STEPS = 2000
BASE_PEAK = 25515734016
CFG8 = {
    "rank": 8,
    "residual_budget": 0.005,
    "residual_bits": 8,
    "surface": "mlp",
    "min_features": 256,
    "chunk_rows": 512,
}


def main():
    out = RESULTS / "q1_longhorizon.jsonl"
    for seed in SEEDS:
        try:
            r = run_config(
                model_id="gpt2-medium",
                bs=8,
                seq=1024,
                steps=STEPS,
                seed=seed,
                method="ckpt_k2_plus_stacr",
                eval_batches=32,
                measure_from=12,
                baseline_peak_bytes=BASE_PEAK,
                tag="lh_gpt2m",
                ckpt_every=2,
                ckpt_offset=0,
                stacr_cfg=CFG8,
                stacr_block_pred=lambda i: i % 2 == 1,
            )
        except Exception as exc:
            emit(
                {
                    "tag": "lh_gpt2m",
                    "method": "ckpt_k2_plus_stacr",
                    "seed": seed,
                    "status": "error",
                    "error": f"{type(exc).__name__}: {exc}"[:300],
                },
                out,
            )
            continue
        emit(r, out)
        print(
            f"  >>> seed {seed}: {r['peak_GiB']:.4f} GiB  {r['time_s']:.1f}s  PPL {r['val_ppl']:.4f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
