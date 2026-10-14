"""Architecture breadth."""

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
(STEPS, MEASURE_FROM) = (40, 10)


def complement(k: int):
    return lambda i, _k=k: i % _k != 0


ARMS = [
    ("baseline_nockpt", {}),
    ("turbo_vs_nockpt", dict(stacr_cfg=TURBO)),
    ("ckpt_k4", dict(ckpt_every=4, ckpt_offset=0)),
    (
        "ckpt_k4_plus_turbo",
        dict(ckpt_every=4, ckpt_offset=0, stacr_cfg=TURBO, stacr_block_pred=complement(4)),
    ),
    ("ckpt_all", dict(ckpt_every=1, ckpt_offset=0)),
    ("ckpt_all_plus_turbo", dict(ckpt_every=1, ckpt_offset=0, stacr_cfg=TURBO)),
]
TIER_FP32 = [
    ("pythia_1_4b", "EleutherAI/pythia-1.4b", "GPT-NeoX, GELU MLP, non-SwiGLU"),
    ("falcon_rw_1b", "tiiuae/falcon-rw-1b", "Falcon, transformer.h, non-SwiGLU"),
    ("gemma3_1b", "google/gemma-3-1b-pt", "GeGLU, Google"),
    ("olmo2_1b", "allenai/OLMo-2-0425-1B", "OLMo 2, AI2"),
    ("stablelm2_1_6b", "stabilityai/stablelm-2-1_6b", "StableLM 2, Stability"),
]
TIER_BF16 = [
    ("phi2_2_7b", "microsoft/phi-2", "parallel attention+MLP block, fc1/fc2"),
    ("gemma2_2b", "google/gemma-2-2b", "GeGLU at 2.6B, logit softcap"),
    ("starcoder2_3b", "bigcode/starcoder2-3b", "code domain, 3B, c_fc/c_proj"),
]
TIER_OPT = [("opt_1_3b", "facebook/opt-1.3b", "OPT, ReLU FFN, fc1/fc2 on the decoder layer")]
TIERS = {
    "fp32": dict(models=TIER_FP32, dtype=torch.float32),
    "bf16": dict(models=TIER_BF16, dtype=torch.bfloat16),
    "opt": dict(models=TIER_OPT, dtype=torch.float32),
}


def run_tier(name: str, spec: dict, out: Path, smoke: bool = False):
    models = spec["models"][:1] if smoke else spec["models"]
    arms = ARMS[:2] if smoke else ARMS
    seeds = SEEDS[:1] if smoke else SEEDS
    steps = 3 if smoke else STEPS
    measure_from = 1 if smoke else MEASURE_FROM
    print(
        f"\n########## X9 tier {name}  dtype={spec['dtype']}  {len(models)} models x {len(arms)} arms x {len(seeds)} seeds ##########",
        flush=True,
    )
    for tag, model_id, why in models:
        print(f"\n=== {name}/{tag}  {model_id}  ({why}) ===", flush=True)
        base_peak = None
        for method, extra in arms:
            for seed in seeds:
                try:
                    rec = run_config(
                        model_id=model_id,
                        bs=2,
                        seq=1024,
                        steps=steps,
                        seed=seed,
                        method=method,
                        tag=f"breadth_{name}/{tag}",
                        dtype=spec["dtype"],
                        amp=True,
                        eight_bit=True,
                        eval_batches=2,
                        measure_from=measure_from,
                        baseline_peak_bytes=base_peak,
                        **extra,
                    )
                except torch.cuda.OutOfMemoryError:
                    emit(
                        {
                            "tag": f"breadth_{name}/{tag}",
                            "method": method,
                            "seed": seed,
                            "status": "OOM",
                        },
                        out,
                    )
                    print(f"  OOM  {method} seed={seed}", flush=True)
                    torch.cuda.empty_cache()
                    continue
                except Exception as exc:
                    emit(
                        {
                            "tag": f"breadth_{name}/{tag}",
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
                rec["cohort"] = f"breadth_{name}"
                rec["family_note"] = why
                if method == "baseline_nockpt" and base_peak is None:
                    base_peak = rec["peak_bytes"]
                emit(rec, out)
                sav = 100.0 * (base_peak - rec["peak_bytes"]) / base_peak if base_peak else 0.0
                print(
                    f"  {method:20s} seed={seed} peak={rec['peak_GiB']:7.3f} GiB ({sav:+6.2f}%) t={rec['time_s']:6.1f}s tok/s={rec['tokens_per_s']:8.0f} wrapped={rec.get('n_wrapped_layers')} ckpt={rec.get('n_checkpointed_blocks')}",
                    flush=True,
                )
                torch.cuda.empty_cache()


def main():
    smoke = "--smoke" in sys.argv
    out = RESULTS / ("x9_breadth_smoke.jsonl" if smoke else "x9_breadth.jsonl")
    want = [a for a in sys.argv[1:] if a in TIERS] or list(TIERS)
    for name in want:
        run_tier(name, TIERS[name], out, smoke=smoke)
    print("X9 DONE", flush=True)


if __name__ == "__main__":
    main()
