"""Capacity search."""

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
PROBES = [
    ("gpt2-large", 1024, torch.float32, 4, 10),
    ("gpt2-xl", 1024, torch.float32, 1, 6),
    ("gpt2-xl", 1024, torch.bfloat16, 2, 12),
    ("Qwen/Qwen2.5-1.5B", 1024, torch.bfloat16, 2, 12),
]


def largest_fitting(model_id, seq, dtype, lo, hi, method, cfg, out, tag):
    best = None
    for bs in range(lo, hi + 1):
        try:
            r = run_config(
                model_id=model_id,
                bs=bs,
                seq=seq,
                steps=12,
                seed=42,
                method=method,
                stacr_cfg=cfg,
                amp=True,
                eight_bit=True,
                dtype=dtype,
                eval_batches=2,
                measure_from=5,
                tag=tag,
            )
            r["fits"] = True
            emit(r, out)
            best = r
        except Exception as exc:
            emit(
                {
                    "tag": tag,
                    "method": method,
                    "bs": bs,
                    "seq": seq,
                    "fits": False,
                    "status": type(exc).__name__,
                    "error": str(exc)[:160],
                },
                out,
            )
            break
    return best


def main():
    out = RESULTS / "q5_capacity.jsonl"
    for model_id, seq, dtype, lo, hi in PROBES:
        dt = {torch.float32: "fp32", torch.bfloat16: "bf16"}[dtype]
        tag = f"cap/{model_id.split('/')[-1]}/{dt}"
        b = largest_fitting(model_id, seq, dtype, lo, hi, "baseline", None, out, tag)
        s = largest_fitting(model_id, seq, dtype, lo, hi, "stacr_r8", CFG, out, tag)
        emit(
            {
                "tag": tag,
                "summary": True,
                "max_batch_baseline": b["bs"] if b else None,
                "max_batch_stacr": s["bs"] if s else None,
                "peak_baseline_GiB": b["peak_GiB"] if b else None,
                "peak_stacr_GiB": s["peak_GiB"] if s else None,
            },
            out,
        )


if __name__ == "__main__":
    main()
