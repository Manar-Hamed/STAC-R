"""Training horizon."""

from __future__ import annotations
import torch
import sys
from pathlib import Path
from .common import RESULTS, emit, run_config

CFG8 = {
    "rank": 8,
    "residual_budget": 0.005,
    "residual_bits": 8,
    "surface": "mlp",
    "min_features": 256,
    "chunk_rows": 512,
}
TURBO = dict(CFG8, turbo_after_step=2)
(MODEL, BS, SEQ) = ("gpt2-medium", 8, 1024)


def complement(k: int):
    return lambda i, _k=k: i % _k != 0


ARMS = [
    ("baseline", {}),
    ("turbo_fast", dict(stacr_cfg=TURBO)),
    ("ckpt_k4", dict(ckpt_every=4, ckpt_offset=0)),
    (
        "ckpt_k4_plus_turbo",
        dict(ckpt_every=4, ckpt_offset=0, stacr_cfg=TURBO, stacr_block_pred=complement(4)),
    ),
]
ARMS_10K = [ARMS[0], ARMS[3]]
ARMS_MODERN = [
    ("baseline", {}),
    ("freeze_mlp", dict(freeze_surface=True)),
    ("turbo_fast", dict(stacr_cfg=TURBO)),
    (
        "ckpt_k4_plus_turbo",
        dict(ckpt_every=4, ckpt_offset=0, stacr_cfg=TURBO, stacr_block_pred=complement(4)),
    ),
]
COHORTS = {
    "h5000": dict(steps=5000, measure_from=50, seeds=[42, 123, 456], arms=ARMS),
    "h10000": dict(steps=10000, measure_from=50, seeds=[42], arms=ARMS_10K),
    "h2000_gpt2l": dict(
        model="gpt2-large", bs=4, steps=2000, measure_from=50, seeds=[42, 123, 456], arms=ARMS
    ),
    "h2000_qwen": dict(
        model="Qwen/Qwen2.5-1.5B",
        bs=2,
        steps=2000,
        measure_from=50,
        seeds=[42, 123, 456],
        arms=ARMS_MODERN,
    ),
    "h2000_llama": dict(
        model="meta-llama/Llama-3.2-1B",
        bs=2,
        steps=2000,
        measure_from=50,
        seeds=[42, 123, 456],
        arms=ARMS_MODERN,
    ),
    "h2000_owt": dict(
        steps=2000, measure_from=50, seeds=[42, 123, 456], arms=ARMS_10K, dataset="openwebtext"
    ),
    "h2000_gpt2xl": dict(
        model="gpt2-xl",
        bs=2,
        steps=2000,
        measure_from=50,
        dtype=torch.bfloat16,
        seeds=[42, 123, 456],
        arms=ARMS,
    ),
    "h10000_more": dict(steps=10000, measure_from=50, seeds=[123, 456], arms=ARMS_10K),
    "h20000_owt": dict(
        steps=20000, measure_from=100, seeds=[42], arms=ARMS_10K, dataset="openwebtext"
    ),
    "h5000_owt": dict(
        steps=5000, measure_from=50, seeds=[42, 123, 456], arms=ARMS_10K, dataset="openwebtext"
    ),
    "h10000_owt": dict(
        steps=10000, measure_from=50, seeds=[42], arms=ARMS_10K, dataset="openwebtext"
    ),
}


def run_cohort(name: str, spec: dict, out: Path, smoke: bool = False):
    steps = 3 if smoke else spec["steps"]
    measure_from = 1 if smoke else spec["measure_from"]
    seeds = spec["seeds"][:1] if smoke else spec["seeds"]
    arms = spec["arms"][:2] if smoke else spec["arms"]
    model = spec.get("model", MODEL)
    bs = spec.get("bs", BS)
    dataset = spec.get("dataset", "wikitext")
    dtype = spec.get("dtype", torch.float32)
    print(
        f"\n########## X10 {name}  {model} b={bs} s={SEQ} steps={steps} data={dataset} dtype={dtype}  {len(arms)} arms x {len(seeds)} seeds ##########",
        flush=True,
    )
    base_peak = None
    for method, extra in arms:
        for seed in seeds:
            try:
                rec = run_config(
                    model_id=model,
                    bs=bs,
                    seq=SEQ,
                    steps=steps,
                    seed=seed,
                    method=method,
                    tag=name,
                    dtype=dtype,
                    amp=True,
                    eight_bit=True,
                    dataset=dataset,
                    eval_batches=16,
                    measure_from=measure_from,
                    baseline_peak_bytes=base_peak,
                    **extra,
                )
            except torch.cuda.OutOfMemoryError:
                emit({"tag": name, "method": method, "seed": seed, "status": "OOM"}, out)
                print(f"  OOM  {method} seed={seed}", flush=True)
                torch.cuda.empty_cache()
                continue
            except Exception as exc:
                emit(
                    {
                        "tag": name,
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
            rec["tokens_per_s"] = rec["bs"] * rec["seq"] * rec["steps_measured"] / rec["time_s"]
            rec["cohort"] = name
            if method == "baseline" and base_peak is None:
                base_peak = rec["peak_bytes"]
            emit(rec, out)
            sav = 100.0 * (base_peak - rec["peak_bytes"]) / base_peak if base_peak else 0.0
            short = (
                ""
                if rec.get("steps_ran") == steps
                else f"  !! STREAM SHORT: ran {rec.get('steps_ran')} of {steps}"
            )
            print(
                f"  {method:20s} seed={seed} peak={rec['peak_GiB']:7.3f} GiB ({sav:+6.2f}%) t={rec['time_s']:7.1f}s ppl={rec['val_ppl']:.4f} wrapped={rec.get('n_wrapped_layers')} ckpt={rec.get('n_checkpointed_blocks')}{short}",
                flush=True,
            )
            torch.cuda.empty_cache()


def main():
    smoke = "--smoke" in sys.argv
    out = RESULTS / ("x10_horizon_smoke.jsonl" if smoke else "x10_horizon.jsonl")
    want = [a for a in sys.argv[1:] if a in COHORTS] or list(COHORTS)
    for name in want:
        run_cohort(name, COHORTS[name], out, smoke=smoke)
    print("X10 DONE", flush=True)


if __name__ == "__main__":
    main()
