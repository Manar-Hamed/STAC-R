"""Large model fidelity."""

from __future__ import annotations
import torch
import sys
from pathlib import Path
from .single_model_fidelity import run as single_run
from .common import RESULTS, emit

MODELS = [
    ("qwen25_7b", "Qwen/Qwen2.5-7B", 1, 512),
    ("llama31_8b", "meta-llama/Llama-3.1-8B", 1, 512),
    ("pythia14b", "EleutherAI/pythia-1.4b", 2, 1024),
]


def main():
    out = RESULTS / "a_scale_fidelity.jsonl"
    for tag, mid, bs, seq in MODELS:
        for rank in [8, 32]:
            for budget in [0.005, 0.0]:
                t = f"{tag}/r{rank}/b{budget}"
                try:
                    r = single_run(mid, bs, seq, rank, budget, t, steps=3)
                    emit(r, out)
                    print(
                        f"  {t:<28} full={r['mean_cosine_full_model']:.4f} local={r['mean_cosine_local_wrapped']:.4f} unwrapped={r['worst_unwrapped_cosine']:.5f}",
                        flush=True,
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


if __name__ == "__main__":
    main()
