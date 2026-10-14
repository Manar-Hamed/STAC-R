"""Swiglu fidelity."""

from __future__ import annotations
import torch
import sys
from pathlib import Path
from .common import RESULTS, emit, gradient_fidelity

BASE = {"residual_bits": 8, "surface": "mlp", "min_features": 256, "chunk_rows": 512}
MODELS = [
    ("llama32_1b", "meta-llama/Llama-3.2-1B", 1, 512),
    ("qwen25_1.5b", "Qwen/Qwen2.5-1.5B", 1, 512),
    ("qwen3_1.7b", "Qwen/Qwen3-1.7B", 1, 512),
    ("gpt2m_ref", "gpt2-medium", 2, 1024),
]


def main():
    import os

    out = RESULTS / os.environ.get("SWIGLU_FID_OUT", "q4_swiglu_fidelity.jsonl")
    print(f"[swiglu_fidelity] writing to {out.name}", flush=True)
    for tag, mid, bs, seq in MODELS:
        for rank in [8, 32, 128]:
            for budget in [0.005, 0.0]:
                try:
                    emit(
                        gradient_fidelity(
                            model_id=mid,
                            bs=bs,
                            seq=seq,
                            seed=42,
                            steps=6,
                            stacr_cfg=dict(BASE, rank=rank, residual_budget=budget),
                            tag=f"{tag}/r{rank}/b{budget}",
                        ),
                        out,
                    )
                except Exception as exc:
                    emit(
                        {
                            "tag": f"{tag}/r{rank}/b{budget}",
                            "status": "error",
                            "error": f"{type(exc).__name__}: {exc}"[:300],
                        },
                        out,
                    )


if __name__ == "__main__":
    main()
