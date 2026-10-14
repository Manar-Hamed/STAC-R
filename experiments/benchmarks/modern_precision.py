"""Modern precision."""

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
SEEDS = [42, 123, 456]
MODELS = [
    ("qwen25_1_5b", "Qwen/Qwen2.5-1.5B", 2, 1024),
    ("llama32_1b", "meta-llama/Llama-3.2-1B", 2, 1024),
    ("qwen3_1_7b", "Qwen/Qwen3-1.7B", 2, 1024),
]


def complement(k: int):
    return lambda i, _k=k: i % _k != 0


FP32_ARMS = [
    ("baseline_nockpt", {}),
    ("baseline_ckptfull", dict(ckpt_every=1, ckpt_offset=0)),
    ("stacr_on_ckptfull", dict(ckpt_every=1, ckpt_offset=0, stacr_cfg=CFG8)),
    ("turbo_on_ckptfull", dict(ckpt_every=1, ckpt_offset=0, stacr_cfg=TURBO)),
    ("ckpt_k4", dict(ckpt_every=4, ckpt_offset=0)),
    (
        "ckpt_k4_plus_turbo",
        dict(ckpt_every=4, ckpt_offset=0, stacr_cfg=TURBO, stacr_block_pred=complement(4)),
    ),
    ("turbo_vs_nockpt", dict(stacr_cfg=TURBO)),
]
Q600_ARMS = [
    ("baseline_nockpt", {}),
    ("turbo_vs_nockpt", dict(stacr_cfg=TURBO)),
    ("stacr_r8_vs_nockpt", dict(stacr_cfg=CFG8)),
    ("ckpt_k4", dict(ckpt_every=4, ckpt_offset=0)),
    (
        "ckpt_k4_plus_turbo",
        dict(ckpt_every=4, ckpt_offset=0, stacr_cfg=TURBO, stacr_block_pred=complement(4)),
    ),
]
COHORTS = {
    "ms_fp32": dict(arms=FP32_ARMS, dtype=torch.float32, steps=40, measure_from=10, eval_batches=2),
    "ms_q600": dict(
        arms=Q600_ARMS, dtype=torch.bfloat16, steps=600, measure_from=20, eval_batches=16
    ),
    "ms_q600f": dict(
        arms=Q600_ARMS, dtype=torch.float32, steps=600, measure_from=20, eval_batches=16
    ),
}


def run_cohort(name: str, spec: dict, out: Path):
    print(
        f"\n########## cohort {name}  dtype={spec['dtype']} steps={spec['steps']}  {len(spec['arms'])} arms x {len(SEEDS)} seeds ##########",
        flush=True,
    )
    for tag, model_id, bs, seq in MODELS:
        print(f"\n=== {name}/{tag}  {model_id}  b={bs} s={seq} ===", flush=True)
        base_peak = None
        for method, extra in spec["arms"]:
            for seed in SEEDS:
                try:
                    rec = run_config(
                        model_id=model_id,
                        bs=bs,
                        seq=seq,
                        steps=spec["steps"],
                        seed=seed,
                        method=method,
                        tag=f"{name}/{tag}",
                        dtype=spec["dtype"],
                        amp=True,
                        eight_bit=True,
                        eval_batches=spec["eval_batches"],
                        measure_from=spec["measure_from"],
                        baseline_peak_bytes=base_peak,
                        **extra,
                    )
                except torch.cuda.OutOfMemoryError:
                    emit(
                        {"tag": f"{name}/{tag}", "method": method, "seed": seed, "status": "OOM"},
                        out,
                    )
                    print(f"  OOM {method} seed={seed}", flush=True)
                    torch.cuda.empty_cache()
                    continue
                except Exception as exc:
                    emit(
                        {
                            "tag": f"{name}/{tag}",
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
                if method == "baseline_nockpt" and base_peak is None:
                    base_peak = rec["peak_bytes"]
                    print(f"  uncheckpointed baseline peak = {rec['peak_GiB']:.3f} GiB", flush=True)
                emit(rec, out)
                sav = 100.0 * (base_peak - rec["peak_bytes"]) / base_peak if base_peak else 0.0
                print(
                    f"  {method:22s} seed={seed} peak={rec['peak_GiB']:7.3f} GiB ({sav:+6.2f}%) t={rec['time_s']:6.1f}s ppl={rec['val_ppl']:.4f} wrapped={rec.get('n_wrapped_layers')} ckpt={rec.get('n_checkpointed_blocks')}",
                    flush=True,
                )
                torch.cuda.empty_cache()


def main():
    out = RESULTS / "x6_modern_scope.jsonl"
    want = [a for a in sys.argv[1:] if a in COHORTS]
    if not want:
        want = list(COHORTS)
    for name in want:
        run_cohort(name, COHORTS[name], out)


if __name__ == "__main__":
    main()
