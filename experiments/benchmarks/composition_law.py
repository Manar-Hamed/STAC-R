"""Composition law."""

from __future__ import annotations
import torch
import sys
from pathlib import Path
from .common import RESULTS, emit, run_config

SEEDS = [42, 123, 456]
(MODEL, BS, SEQ, STEPS) = ("gpt2-medium", 8, 1024, 300)
K = 4
BASE = {
    "rank": 8,
    "residual_budget": 0.005,
    "residual_bits": 8,
    "surface": "mlp",
    "min_features": 256,
    "chunk_rows": 512,
}
KINDS = ["stacr", "lowrank", "oasis", "prac", "untracked"]


def complement(k: int):
    return lambda i, _k=k: i % _k != 0


def arms():
    out = [("baseline", {}), (f"ckpt_k{K}", dict(ckpt_every=K, ckpt_offset=0))]
    for kind in KINDS:
        out.append((f"{kind}_alone", dict(stacr_cfg=dict(BASE), comparator=kind)))
        out.append(
            (
                f"ckpt_k{K}_plus_{kind}",
                dict(
                    ckpt_every=K,
                    ckpt_offset=0,
                    stacr_cfg=dict(BASE),
                    comparator=kind,
                    stacr_block_pred=complement(K),
                ),
            )
        )
    out.append(("turbo_alone", dict(stacr_cfg=dict(BASE, turbo_after_step=2))))
    out.append(
        (
            f"ckpt_k{K}_plus_turbo",
            dict(
                ckpt_every=K,
                ckpt_offset=0,
                stacr_cfg=dict(BASE, turbo_after_step=2),
                stacr_block_pred=complement(K),
            ),
        )
    )
    return out


def main():
    out = RESULTS / "x3_composition_law.jsonl"
    A = arms()
    print(
        f"=== composition law: {MODEL} b={BS} s={SEQ} steps={STEPS}, {len(A)} arms x {len(SEEDS)} seeds ===",
        flush=True,
    )
    base_peak = None
    for method, extra in A:
        for seed in SEEDS:
            try:
                rec = run_config(
                    model_id=MODEL,
                    bs=BS,
                    seq=SEQ,
                    steps=STEPS,
                    seed=seed,
                    method=method,
                    tag="law_gpt2m",
                    baseline_peak_bytes=base_peak,
                    **extra,
                )
            except torch.cuda.OutOfMemoryError:
                emit({"tag": "law_gpt2m", "method": method, "seed": seed, "status": "OOM"}, out)
                print(f"  OOM {method} seed={seed}", flush=True)
                torch.cuda.empty_cache()
                continue
            except Exception as exc:
                emit(
                    {
                        "tag": "law_gpt2m",
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
                f"  {method:26s} seed={seed} peak={rec['peak_GiB']:7.3f} GiB ({sav:+6.2f}%) t={rec['time_s']:6.1f}s ppl={rec['val_ppl']:.4f} wrapped={rec.get('n_wrapped_layers')}",
                flush=True,
            )


if __name__ == "__main__":
    main()
