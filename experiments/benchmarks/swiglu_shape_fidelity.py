"""Swiglu shape fidelity."""

from __future__ import annotations
import torch
import sys
from pathlib import Path
from .common import RESULTS, emit, gradient_fidelity

BASE = {"residual_bits": 8, "surface": "mlp", "min_features": 256, "chunk_rows": 512}
MODELS = [
    ("llama32_1b", "meta-llama/Llama-3.2-1B", 2, 1024),
    ("qwen25_1.5b", "Qwen/Qwen2.5-1.5B", 2, 1024),
]


def main():
    out = RESULTS / "a_swiglu_fid_papershape.jsonl"
    for tag, mid, bs, seq in MODELS:
        for rank in (8,):
            for budget in (0.005, 0.0):
                t = f"{tag}/b{bs}_s{seq}/r{rank}/res{budget}"
                try:
                    r = gradient_fidelity(
                        model_id=mid,
                        bs=bs,
                        seq=seq,
                        seed=42,
                        steps=6,
                        stacr_cfg=dict(BASE, rank=rank, residual_budget=budget),
                        tag=t,
                    )
                except Exception as exc:
                    emit(
                        {
                            "tag": t,
                            "status": "error",
                            "error": f"{type(exc).__name__}: {exc}"[:250],
                        },
                        out,
                    )
                    print(f"  {t:<40} ERROR {type(exc).__name__}", flush=True)
                    continue
                emit(r, out)
                print(
                    f"  {t:<40} full={r['mean_cosine_full_model']:.6f} local={r['mean_cosine_local_wrapped']:.6f}",
                    flush=True,
                )


if __name__ == "__main__":
    main()
