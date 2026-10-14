"""Fidelity horizon."""

from __future__ import annotations
import torch
import sys
from pathlib import Path
from .common import RESULTS, emit, gradient_fidelity

BASE = {
    "residual_bits": 8,
    "surface": "mlp",
    "min_features": 256,
    "chunk_rows": 512,
    "rank": 8,
    "residual_budget": 0.005,
}
MODELS = [
    ("qwen25_1.5b", "Qwen/Qwen2.5-1.5B"),
    ("llama32_1b", "meta-llama/Llama-3.2-1B"),
    ("gpt2m", "gpt2-medium"),
]
STEPS = [6, 24, 96, 240]


def main():
    out = RESULTS / "a_fid_steps.jsonl"
    for tag, mid in MODELS:
        for steps in STEPS:
            t = f"{tag}/b2_s1024/r8/steps{steps}"
            try:
                r = gradient_fidelity(
                    model_id=mid, bs=2, seq=1024, seed=42, steps=steps, stacr_cfg=dict(BASE), tag=t
                )
            except Exception as exc:
                emit(
                    {"tag": t, "status": "error", "error": f"{type(exc).__name__}: {exc}"[:250]},
                    out,
                )
                print(f"  {t:<36} ERROR {type(exc).__name__}", flush=True)
                continue
            emit(r, out)
            print(
                f"  {t:<36} full={r['mean_cosine_full_model']:.6f} local={r['mean_cosine_local_wrapped']:.6f}",
                flush=True,
            )


if __name__ == "__main__":
    main()
