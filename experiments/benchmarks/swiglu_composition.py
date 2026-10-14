"""Swiglu composition."""

from __future__ import annotations
import torch
import sys
from pathlib import Path
from .common import RESULTS, emit, run_config

CFG = {
    "rank": 8,
    "residual_budget": 0.005,
    "residual_bits": 8,
    "surface": "mlp",
    "min_features": 256,
    "chunk_rows": 512,
}
CASES = [("llama", "meta-llama/Llama-3.2-1B", 1, 4096), ("qwen25", "Qwen/Qwen2.5-1.5B", 2, 1024)]
ROWS = [
    ("baseline_no_ckpt", dict(ckpt_every=0)),
    ("ckpt_all", dict(ckpt_every=1)),
    ("ckpt_all_plus_stacr_same", dict(ckpt_every=1, stacr_cfg=CFG)),
    ("ckpt_k2", dict(ckpt_every=2)),
    (
        "ckpt_k2_plus_stacr_disjoint",
        dict(ckpt_every=2, ckpt_offset=0, stacr_cfg=CFG, stacr_block_pred=lambda i: i % 2 == 1),
    ),
    (
        "ckpt_k2_plus_stacr_same",
        dict(ckpt_every=2, ckpt_offset=0, stacr_cfg=CFG, stacr_block_pred=lambda i: i % 2 == 0),
    ),
]


def main():
    out = RESULTS / "r2_swiglu_fix.jsonl"
    for tagm, model_id, bs, seq in CASES:
        ref = None
        for label, extra in ROWS:
            try:
                r = run_config(
                    model_id=model_id,
                    bs=bs,
                    seq=seq,
                    steps=24,
                    seed=42,
                    method=label,
                    amp=True,
                    eight_bit=True,
                    dtype=torch.bfloat16,
                    eval_batches=2,
                    measure_from=6,
                    tag=f"fix/{tagm}",
                    **extra,
                )
            except Exception as exc:
                emit(
                    {
                        "tag": f"fix/{tagm}",
                        "method": label,
                        "status": "OOM_or_error",
                        "error": f"{type(exc).__name__}: {exc}"[:200],
                    },
                    out,
                )
                continue
            if ref is None:
                ref = r["peak_bytes"]
            r["vs_no_ckpt_pct"] = 100.0 * (ref - r["peak_bytes"]) / ref
            emit(r, out)
            print(
                f"  >>> {tagm}/{label}: {r['peak_GiB']:.3f} GiB ({r['vs_no_ckpt_pct']:+.2f}% vs uncheckpointed)",
                flush=True,
            )


if __name__ == "__main__":
    main()
