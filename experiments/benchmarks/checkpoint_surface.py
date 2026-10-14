"""Checkpoint surface."""

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
    "basis_dtype": "float16",
    "freeze_after_warmup": True,
    "warmup_steps": 4,
}
CASES = [
    ("llama_paper_shape", "meta-llama/Llama-3.2-1B", 1, 4096, None),
    ("qwen_paper_shape", "Qwen/Qwen2.5-1.5B", 2, 1024, None),
]
CKPT_MODES = [("no_ckpt", 0), ("ckpt_all", 1), ("ckpt_k2", 2)]


def main():
    out = RESULTS / "r1_ckpt_baseline.jsonl"
    for label, model_id, bs, seq, _ in CASES:
        for mode, every in CKPT_MODES:
            tag = f"{label}/{mode}"
            try:
                base = run_config(
                    model_id=model_id,
                    bs=bs,
                    seq=seq,
                    steps=24,
                    seed=42,
                    method="baseline",
                    ckpt_every=every,
                    amp=True,
                    eight_bit=True,
                    dtype=torch.bfloat16,
                    eval_batches=2,
                    measure_from=6,
                    tag=tag,
                )
            except Exception as exc:
                emit(
                    {
                        "tag": tag,
                        "method": "baseline",
                        "status": "OOM_or_error",
                        "error": f"{type(exc).__name__}: {exc}"[:200],
                    },
                    out,
                )
                continue
            emit(base, out)
            bp = base["peak_bytes"]
            try:
                r = run_config(
                    model_id=model_id,
                    bs=bs,
                    seq=seq,
                    steps=24,
                    seed=42,
                    method="stacr_r8",
                    stacr_cfg=dict(CFG),
                    ckpt_every=every,
                    amp=True,
                    eight_bit=True,
                    dtype=torch.bfloat16,
                    eval_batches=2,
                    measure_from=6,
                    baseline_peak_bytes=bp,
                    tag=tag,
                )
            except Exception as exc:
                emit(
                    {
                        "tag": tag,
                        "method": "stacr_r8",
                        "status": "OOM_or_error",
                        "error": f"{type(exc).__name__}: {exc}"[:200],
                    },
                    out,
                )
                continue
            r["measured_saving_pct"] = 100.0 * (bp - r["peak_bytes"]) / bp
            r["baseline_peak_GiB"] = base["peak_GiB"]
            r["ckpt_mode"] = mode
            emit(r, out)
            print(
                f"  >>> {tag}: baseline {base['peak_GiB']:.3f} -> stacr {r['peak_GiB']:.3f}  = {r['measured_saving_pct']:+.2f}% ({('SAVING' if r['measured_saving_pct'] > 0 else 'INCREASE')})",
                flush=True,
            )


if __name__ == "__main__":
    main()
