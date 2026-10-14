"""Precision corpus."""

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


def complement(k: int):
    return lambda i, _k=k: i % _k != 0


FRONTIER_ARMS = [
    ("baseline", {}),
    ("turbo_fast", dict(stacr_cfg=TURBO)),
    ("ckpt_k4", dict(ckpt_every=4, ckpt_offset=0)),
    (
        "ckpt_k4_plus_turbo",
        dict(ckpt_every=4, ckpt_offset=0, stacr_cfg=TURBO, stacr_block_pred=complement(4)),
    ),
    ("ckpt_k2", dict(ckpt_every=2, ckpt_offset=0)),
    (
        "ckpt_k2_plus_turbo",
        dict(ckpt_every=2, ckpt_offset=0, stacr_cfg=TURBO, stacr_block_pred=complement(2)),
    ),
    ("ckpt_all", dict(ckpt_every=1, ckpt_offset=0)),
    ("ckpt_all_plus_turbo", dict(ckpt_every=1, ckpt_offset=0, stacr_cfg=TURBO)),
]
SHAPE_ARMS = [
    ("baseline_nockpt", {}),
    ("baseline_ckptfull", dict(ckpt_every=1, ckpt_offset=0)),
    ("stacr_on_ckptfull", dict(ckpt_every=1, ckpt_offset=0, stacr_cfg=CFG8)),
    ("turbo_on_ckptfull", dict(ckpt_every=1, ckpt_offset=0, stacr_cfg=TURBO)),
]
COHORTS = {
    "xl_fp32": dict(
        models=[("gpt2xl_fp32", "gpt2-xl")],
        bs=1,
        seq=1024,
        steps=120,
        measure_from=20,
        dtype=torch.float32,
        dataset="wikitext",
        arms=FRONTIER_ARMS,
        seeds=SEEDS,
        eval_batches=8,
    ),
    "owt": dict(
        models=[("gpt2m_owt", "gpt2-medium")],
        bs=8,
        seq=1024,
        steps=400,
        measure_from=8,
        dtype=torch.float32,
        dataset="openwebtext",
        arms=FRONTIER_ARMS,
        seeds=SEEDS,
        eval_batches=16,
    ),
    "shape32": dict(
        models=[("qwen25_1_5b", "Qwen/Qwen2.5-1.5B"), ("llama32_1b", "meta-llama/Llama-3.2-1B")],
        bs=3,
        seq=1024,
        steps=40,
        measure_from=10,
        dtype=torch.float32,
        dataset="wikitext",
        arms=SHAPE_ARMS,
        seeds=SEEDS,
        eval_batches=2,
        also_bs=[2],
    ),
}


def run_cohort(name: str, spec: dict, out: Path, smoke: bool = False):
    arms = spec["arms"][:3] if smoke else spec["arms"]
    seeds = spec["seeds"][:1] if smoke else spec["seeds"]
    steps = 3 if smoke else spec["steps"]
    mf = 1 if smoke else spec["measure_from"]
    batches = [spec["bs"]] + list(spec.get("also_bs", []))
    for tag, model_id in spec["models"]:
        for bs in batches:
            t = f"{name}/{tag}/b{bs}"
            print(
                f"\n########## X13 {t}  {model_id} b={bs} s={spec['seq']} steps={steps} dtype={spec['dtype']} data={spec['dataset']}  {len(arms)} arms x {len(seeds)} seeds ##########",
                flush=True,
            )
            base_peak = None
            for method, extra in arms:
                for seed in seeds:
                    try:
                        rec = run_config(
                            model_id=model_id,
                            bs=bs,
                            seq=spec["seq"],
                            steps=steps,
                            seed=seed,
                            method=method,
                            tag=t,
                            dtype=spec["dtype"],
                            amp=True,
                            eight_bit=True,
                            dataset=spec["dataset"],
                            eval_batches=2 if smoke else spec["eval_batches"],
                            measure_from=mf,
                            baseline_peak_bytes=base_peak,
                            **extra,
                        )
                    except torch.cuda.OutOfMemoryError:
                        emit({"tag": t, "method": method, "seed": seed, "status": "OOM"}, out)
                        print(f"  OOM  {method} seed={seed}", flush=True)
                        torch.cuda.empty_cache()
                        continue
                    except Exception as exc:
                        emit(
                            {
                                "tag": t,
                                "method": method,
                                "seed": seed,
                                "status": "error",
                                "error": f"{type(exc).__name__}: {str(exc)[:200]}",
                            },
                            out,
                        )
                        print(
                            f"  FAIL {method} seed={seed}: {type(exc).__name__}: {exc}", flush=True
                        )
                        torch.cuda.empty_cache()
                        continue
                    rec["tokens_per_s"] = (
                        rec["bs"] * rec["seq"] * rec["steps_measured"] / rec["time_s"]
                    )
                    rec["cohort"] = name
                    if method in ("baseline", "baseline_nockpt") and base_peak is None:
                        base_peak = rec["peak_bytes"]
                    emit(rec, out)
                    sav = 100.0 * (base_peak - rec["peak_bytes"]) / base_peak if base_peak else 0.0
                    print(
                        f"  {method:22s} seed={seed} peak={rec['peak_GiB']:7.3f} GiB ({sav:+6.2f}%) t={rec['time_s']:6.1f}s ppl={rec['val_ppl']:.4f} wrapped={rec.get('n_wrapped_layers')} ckpt={rec.get('n_checkpointed_blocks')}",
                        flush=True,
                    )
                    torch.cuda.empty_cache()


def main():
    smoke = "--smoke" in sys.argv
    out = RESULTS / ("x13_extra_smoke.jsonl" if smoke else "x13_extra.jsonl")
    want = [a for a in sys.argv[1:] if a in COHORTS] or list(COHORTS)
    for name in want:
        run_cohort(name, COHORTS[name], out, smoke=smoke)
    print("X13 DONE", flush=True)


if __name__ == "__main__":
    main()
