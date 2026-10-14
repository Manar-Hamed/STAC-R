"""Equal budget frontier."""

from __future__ import annotations
import torch
import sys
from pathlib import Path
from .common import RESULTS, emit, gib, run_config

SEEDS = [42, 123, 456]
(MODEL, BS, SEQ, STEPS) = ("gpt2-medium", 8, 1024, 300)
CFG = {
    "rank": 8,
    "residual_budget": 0.005,
    "residual_bits": 8,
    "surface": "mlp",
    "min_features": 256,
    "chunk_rows": 512,
}
ROWS = [
    ("baseline", {}),
    ("ckpt_k6", dict(ckpt_every=6)),
    ("ckpt_k8", dict(ckpt_every=8)),
    ("ckpt_k12", dict(ckpt_every=12)),
    ("ckpt_k16", dict(ckpt_every=16)),
    ("stacr_r8", dict(stacr_cfg=CFG)),
    ("turbo_fast", dict(stacr_cfg=dict(CFG, turbo_after_step=2))),
]


def main():
    out = RESULTS / "r6_equalpareto.jsonl"
    base_peaks: list[float] = []
    base_times: list[float] = []
    bp = bt = None
    for method, extra in ROWS:
        for seed in SEEDS:
            try:
                r = run_config(
                    model_id=MODEL,
                    bs=BS,
                    seq=SEQ,
                    steps=STEPS,
                    seed=seed,
                    method=method,
                    baseline_peak_bytes=bp,
                    tag="pareto",
                    **extra,
                )
            except Exception as exc:
                emit(
                    {
                        "tag": "pareto",
                        "method": method,
                        "seed": seed,
                        "status": "error",
                        "error": f"{type(exc).__name__}: {exc}"[:250],
                    },
                    out,
                )
                continue
            if method == "baseline":
                base_peaks.append(r["peak_bytes"])
                base_times.append(r["time_s"])
                if bp is None:
                    bp = r["peak_bytes"]
            if bt:
                r["time_x"] = r["time_s"] / bt
                r["measured_saving_pct"] = 100.0 * (bp - r["peak_bytes"]) / bp
            emit(r, out)
            print(
                f"  >>> {method:<14} {r.get('time_x', 1.0):.3f}x  mem {r.get('measured_saving_pct', 0.0):+.2f}%  PPL {r['val_ppl']:.4f}",
                flush=True,
            )
        if method == "baseline" and base_times:
            bp = sum(base_peaks) / len(base_peaks)
            bt = sum(base_times) / len(base_times)
            print(
                f"  baseline mean over {len(base_times)} seeds: {gib(bp):.4f} GiB / {bt:.3f} s",
                flush=True,
            )


if __name__ == "__main__":
    main()
