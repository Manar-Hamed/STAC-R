"""Surface expansion."""

from __future__ import annotations
import torch
import sys
from pathlib import Path
from .common import RESULTS, emit, run_config

SEEDS = [42, 123, 456]
BASE = {
    "rank": 8,
    "residual_budget": 0.005,
    "residual_bits": 8,
    "min_features": 256,
    "chunk_rows": 512,
}


def cfg(surface, turbo=True):
    c = dict(BASE, surface=surface)
    if turbo:
        c["turbo_after_step"] = 2
    return c


def complement(k: int):
    return lambda i, _k=k: i % _k != 0


COHORT = dict(name="sx_gpt2m", model_id="gpt2-medium", bs=8, seq=1024, steps=400)
ARMS = [
    ("baseline", {}),
    ("turbo_mlp", dict(stacr_cfg=cfg("mlp"))),
    ("turbo_mlpattn", dict(stacr_cfg=cfg("mlp+attention"))),
    ("turbo_all", dict(stacr_cfg=cfg("all"))),
    ("stacr_mlpattn", dict(stacr_cfg=cfg("mlp+attention", turbo=False))),
    ("ckpt_k4", dict(ckpt_every=4, ckpt_offset=0)),
    (
        "ckpt_k4_plus_turbo_mlp",
        dict(ckpt_every=4, ckpt_offset=0, stacr_cfg=cfg("mlp"), stacr_block_pred=complement(4)),
    ),
    (
        "ckpt_k4_plus_turbo_mlpattn",
        dict(
            ckpt_every=4,
            ckpt_offset=0,
            stacr_cfg=cfg("mlp+attention"),
            stacr_block_pred=complement(4),
        ),
    ),
    ("ckpt_k2", dict(ckpt_every=2, ckpt_offset=0)),
    (
        "ckpt_k2_plus_turbo_mlpattn",
        dict(
            ckpt_every=2,
            ckpt_offset=0,
            stacr_cfg=cfg("mlp+attention"),
            stacr_block_pred=complement(2),
        ),
    ),
]


def main():
    out = RESULTS / "x2_surface_expansion.jsonl"
    coh = COHORT
    print(
        f"=== cohort {coh['name']} {coh['model_id']} b={coh['bs']} s={coh['seq']} steps={coh['steps']}; {len(ARMS)} arms x {len(SEEDS)} seeds ===",
        flush=True,
    )
    base_peak = None
    for method, extra in ARMS:
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
            except torch.cuda.OutOfMemoryError:
                emit({"tag": coh["name"], "method": method, "seed": seed, "status": "OOM"}, out)
                print(f"  OOM {method} seed={seed}", flush=True)
                torch.cuda.empty_cache()
                continue
            except Exception as exc:
                emit(
                    {
                        "tag": coh["name"],
                        "method": method,
                        "seed": seed,
                        "status": "error",
                        "error": f"{type(exc).__name__}: {str(exc)[:200]}",
                    },
                    out,
                )
                print(f"  FAIL {method} seed={seed}: {type(exc).__name__}: {exc}", flush=True)
                torch.cuda.empty_cache()
                continue
            if method == "baseline" and base_peak is None:
                base_peak = rec["peak_bytes"]
                print(f"  baseline peak = {rec['peak_GiB']:.3f} GiB", flush=True)
            emit(rec, out)
            sav = 100.0 * (base_peak - rec["peak_bytes"]) / base_peak if base_peak else 0.0
            print(
                f"  {method:30s} seed={seed} peak={rec['peak_GiB']:7.3f} GiB ({sav:+6.2f}%) t={rec['time_s']:6.1f}s ppl={rec['val_ppl']:.4f} ckpt={rec.get('n_checkpointed_blocks')} wrapped={rec.get('n_wrapped_layers')}",
                flush=True,
            )


if __name__ == "__main__":
    main()
