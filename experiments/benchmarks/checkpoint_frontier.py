"""Checkpoint frontier."""

from __future__ import annotations
import torch
import sys
from pathlib import Path
from .common import RESULTS, emit, run_config

SEEDS = [42, 123, 456]
CFG8 = {
    "rank": 8,
    "residual_budget": 0.005,
    "residual_bits": 8,
    "surface": "mlp",
    "min_features": 256,
    "chunk_rows": 512,
}
TURBO = dict(CFG8, turbo_after_step=2)
COHORTS = [
    dict(name="gpt2m", model_id="gpt2-medium", bs=8, seq=1024, steps=400),
    dict(name="gpt2l", model_id="gpt2-large", bs=4, seq=1024, steps=300),
]


def rows(cohort):
    return [
        ("baseline", {}),
        ("ckpt_full", dict(ckpt_every=1)),
        ("ckpt_k2", dict(ckpt_every=2)),
        ("ckpt_k4", dict(ckpt_every=4)),
        ("stacr_r8", dict(stacr_cfg=CFG8)),
        ("turbo_fast", dict(stacr_cfg=TURBO)),
        (
            "ckpt_k2_plus_stacr",
            dict(
                ckpt_every=2, ckpt_offset=0, stacr_cfg=CFG8, stacr_block_pred=lambda i: i % 2 == 1
            ),
        ),
        (
            "ckpt_k2_stacr_same_blocks",
            dict(
                ckpt_every=2, ckpt_offset=0, stacr_cfg=CFG8, stacr_block_pred=lambda i: i % 2 == 0
            ),
        ),
    ]


def main():
    out = RESULTS / "e1_frontier.jsonl"
    for coh in COHORTS:
        print(
            f"\n=== cohort {coh['name']}  {coh['model_id']} b={coh['bs']} s={coh['seq']} steps={coh['steps']} ===",
            flush=True,
        )
        base_peak = None
        for method, extra in rows(coh):
            for seed in SEEDS:
                try:
                    rec = run_config(
                        model_id=coh["model_id"],
                        bs=coh["bs"],
                        seq=coh["seq"],
                        steps=coh["steps"],
                        seed=seed,
                        method=method,
                        tag=coh["name"],
                        baseline_peak_bytes=base_peak,
                        **extra,
                    )
                except torch.cuda.OutOfMemoryError as exc:
                    print(f"  OOM {method} seed={seed}: {exc}", flush=True)
                    emit({"tag": coh["name"], "method": method, "seed": seed, "status": "OOM"}, out)
                    continue
                except Exception as exc:
                    print(f"  FAIL {method} seed={seed}: {type(exc).__name__}: {exc}", flush=True)
                    emit(
                        {
                            "tag": coh["name"],
                            "method": method,
                            "seed": seed,
                            "status": "error",
                            "error": f"{type(exc).__name__}: {exc}",
                        },
                        out,
                    )
                    continue
                if method == "baseline" and base_peak is None:
                    base_peak = rec["peak_bytes"]
                emit(rec, out)


if __name__ == "__main__":
    import torch

    main()
