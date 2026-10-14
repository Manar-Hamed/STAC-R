"""Principal random rank."""

from __future__ import annotations
import torch
import sys
from pathlib import Path
from .common import RESULTS, emit, run_config, gradient_fidelity

SEEDS = [42, 123, 456]
(MODEL, BS, SEQ, STEPS) = ("gpt2-medium", 8, 1024, 300)
RANK = 307
BASE = {"rank": RANK, "residual_bits": 8, "surface": "mlp", "min_features": 256, "chunk_rows": 512}
KINDS = ["prac", "stacr"]


def main():
    out = RESULTS / "e3b_paperrank.jsonl"
    base_peak = None
    for seed in SEEDS:
        r = run_config(
            model_id=MODEL,
            bs=BS,
            seq=SEQ,
            steps=STEPS,
            seed=seed,
            method="baseline",
            tag="paperrank",
        )
        if base_peak is None:
            base_peak = r["peak_bytes"]
        emit(r, out)
    for kind in KINDS:
        budget = 0.005 if kind == "stacr" else 0.0
        cfg = dict(BASE, residual_budget=budget)
        for seed in SEEDS:
            try:
                r = run_config(
                    model_id=MODEL,
                    bs=BS,
                    seq=SEQ,
                    steps=STEPS,
                    seed=seed,
                    method=f"pr_{kind}",
                    stacr_cfg=dict(cfg),
                    comparator=kind,
                    baseline_peak_bytes=base_peak,
                    tag="paperrank",
                )
            except Exception as exc:
                emit(
                    {
                        "tag": "paperrank",
                        "method": f"pr_{kind}",
                        "rank": RANK,
                        "seed": seed,
                        "status": "error",
                        "error": f"{type(exc).__name__}: {exc}"[:300],
                    },
                    out,
                )
                continue
            r["measured_saving_pct"] = 100.0 * (base_peak - r["peak_bytes"]) / base_peak
            r["rank_used"] = RANK
            emit(r, out)
    fid = RESULTS / "e3b_paperrank_fidelity.jsonl"
    for kind in KINDS:
        budget = 0.005 if kind == "stacr" else 0.0
        try:
            f = gradient_fidelity(
                model_id=MODEL,
                bs=2,
                seq=1024,
                seed=42,
                steps=12,
                stacr_cfg=dict(BASE, residual_budget=budget),
                comparator=kind,
                tag=f"paperrank_fid/{kind}",
            )
            f["rank_used"] = RANK
            emit(f, fid)
        except Exception as exc:
            emit(
                {
                    "tag": f"paperrank_fid/{kind}",
                    "rank": RANK,
                    "status": "error",
                    "error": f"{type(exc).__name__}: {exc}"[:300],
                },
                fid,
            )


if __name__ == "__main__":
    main()
