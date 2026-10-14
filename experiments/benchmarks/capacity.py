"""Capacity."""

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
PROBES = [("gpt2-large", [8, 10, 12, 14, 16], 1024), ("gpt2-xl", [2, 3, 4, 5, 6, 7], 1024)]


def main():
    out = RESULTS / "e8_capacity.jsonl"
    for model_id, batches, seq in PROBES:
        for method, cfg in [("baseline", None), ("stacr_r8", CFG)]:
            for bs in batches:
                try:
                    r = run_config(
                        model_id=model_id,
                        bs=bs,
                        seq=seq,
                        steps=14,
                        seed=42,
                        method=method,
                        stacr_cfg=cfg,
                        eval_batches=2,
                        measure_from=6,
                        tag=f"cap/{model_id}",
                    )
                    r["fits"] = True
                    emit(r, out)
                except Exception as exc:
                    emit(
                        {
                            "tag": f"cap/{model_id}",
                            "method": method,
                            "bs": bs,
                            "seq": seq,
                            "fits": False,
                            "status": type(exc).__name__,
                            "error": str(exc)[:200],
                        },
                        out,
                    )
                    break


if __name__ == "__main__":
    main()
