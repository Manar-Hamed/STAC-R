"""Basis fidelity."""

from __future__ import annotations
import torch
import sys
from pathlib import Path
from .common import RESULTS, emit, gradient_fidelity

MODEL = "gpt2-medium"
BASE = {"rank": 8, "residual_bits": 8, "surface": "mlp", "min_features": 256, "chunk_rows": 512}
KINDS = ["stacr", "lowrank", "untracked", "oasis", "prac"]


def main():
    out = RESULTS / "e3_fidelity_fixed.jsonl"
    for kind in KINDS:
        try:
            r = gradient_fidelity(
                model_id=MODEL,
                bs=2,
                seq=1024,
                seed=42,
                steps=12,
                stacr_cfg=dict(BASE),
                comparator=kind,
                tag="cmp_fid_fixed",
            )
        except Exception as exc:
            emit(
                {
                    "tag": "cmp_fid_fixed",
                    "comparator": kind,
                    "status": "error",
                    "error": f"{type(exc).__name__}: {exc}"[:300],
                },
                out,
            )
            continue
        emit(r, out)
        print(
            f"  {kind:<10} full={r['mean_cosine_full_model']:.6f} local={r['mean_cosine_local_wrapped']:.6f} worstUnwrapped={r['worst_unwrapped_cosine']:.9f} lossgap={r['mean_loss_gap']:.2e}",
            flush=True,
        )


if __name__ == "__main__":
    main()
