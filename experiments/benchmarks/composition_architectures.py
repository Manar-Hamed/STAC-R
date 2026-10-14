"""Composition architectures."""

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
    "surface": "mlp",
    "min_features": 256,
    "chunk_rows": 512,
}
KINDS = ["stacr", "lowrank", "oasis", "prac", "untracked"]
DENSITIES = [2, 4, 8]
MODELS = {
    "qwen25_1_5b": ("Qwen/Qwen2.5-1.5B", 28, torch.float32, 2, 200),
    "pythia_1_4b": ("EleutherAI/pythia-1.4b", 24, torch.float32, 2, 200),
}


def complement(k: int):
    return lambda i, _k=k: i % _k != 0


def arms_for(k: int):
    out = [("baseline", {}), (f"ckpt_k{k}", dict(ckpt_every=k, ckpt_offset=0))]
    for kind in KINDS:
        out.append((f"{kind}_alone", dict(stacr_cfg=dict(BASE), comparator=kind)))
        out.append(
            (
                f"ckpt_k{k}_plus_{kind}",
                dict(
                    ckpt_every=k,
                    ckpt_offset=0,
                    stacr_cfg=dict(BASE),
                    comparator=kind,
                    stacr_block_pred=complement(k),
                ),
            )
        )
    out.append(("turbo_alone", dict(stacr_cfg=dict(BASE, turbo_after_step=2))))
    out.append(
        (
            f"ckpt_k{k}_plus_turbo",
            dict(
                ckpt_every=k,
                ckpt_offset=0,
                stacr_cfg=dict(BASE, turbo_after_step=2),
                stacr_block_pred=complement(k),
            ),
        )
    )
    return out


def run_model(tag: str, out: Path, smoke: bool = False):
    (model_id, nblk, dtype, bs, steps) = MODELS[tag]
    densities = DENSITIES[:1] if smoke else DENSITIES
    seeds = SEEDS[:1] if smoke else SEEDS
    steps = 3 if smoke else steps
    mf = 1 if smoke else 20
    for k in densities:
        A = arms_for(k)
        if smoke:
            A = A[:4]
        print(
            f"\n########## X12 {tag} k={k} ({model_id}, {nblk} blocks)  {len(A)} arms x {len(seeds)} seeds ##########",
            flush=True,
        )
        base_peak = None
        for method, extra in A:
            for seed in seeds:
                try:
                    rec = run_config(
                        model_id=model_id,
                        bs=bs,
                        seq=1024,
                        steps=steps,
                        seed=seed,
                        method=method,
                        tag=f"law/{tag}/k{k}",
                        dtype=dtype,
                        amp=True,
                        eight_bit=True,
                        eval_batches=2 if smoke else 8,
                        measure_from=mf,
                        baseline_peak_bytes=base_peak,
                        **extra,
                    )
                except torch.cuda.OutOfMemoryError:
                    emit(
                        {"tag": f"law/{tag}/k{k}", "method": method, "seed": seed, "status": "OOM"},
                        out,
                    )
                    print(f"  OOM  {method} seed={seed}", flush=True)
                    torch.cuda.empty_cache()
                    continue
                except Exception as exc:
                    emit(
                        {
                            "tag": f"law/{tag}/k{k}",
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
                rec["cohort"] = f"law_{tag}_k{k}"
                rec["density_k"] = k
                if method == "baseline" and base_peak is None:
                    base_peak = rec["peak_bytes"]
                emit(rec, out)
                sav = 100.0 * (base_peak - rec["peak_bytes"]) / base_peak if base_peak else 0.0
                print(
                    f"  {method:26s} seed={seed} peak={rec['peak_GiB']:7.3f} GiB ({sav:+6.2f}%) t={rec['time_s']:6.1f}s wrapped={rec.get('n_wrapped_layers')} ckpt={rec.get('n_checkpointed_blocks')}",
                    flush=True,
                )
                torch.cuda.empty_cache()


def main():
    smoke = "--smoke" in sys.argv
    out = RESULTS / ("x12_law_arch_smoke.jsonl" if smoke else "x12_law_arch.jsonl")
    want = [a for a in sys.argv[1:] if a in MODELS] or list(MODELS)
    for tag in want:
        run_model(tag, out, smoke=smoke)
    print("X12 DONE", flush=True)


if __name__ == "__main__":
    main()
