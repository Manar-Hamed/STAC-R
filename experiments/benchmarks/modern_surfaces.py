"""Modern surfaces."""

from __future__ import annotations
import torch
import sys
from pathlib import Path
from .common import RESULTS, emit, run_config

BASE = {
    "rank": 8,
    "residual_budget": 0.005,
    "residual_bits": 8,
    "min_features": 256,
    "chunk_rows": 512,
}
SEEDS = [42, 123, 456]
(STEPS, MEASURE_FROM) = (40, 10)
MODELS = [
    ("qwen25_1_5b", "Qwen/Qwen2.5-1.5B"),
    ("llama32_1b", "meta-llama/Llama-3.2-1B"),
    ("qwen3_1_7b", "Qwen/Qwen3-1.7B"),
]


def turbo(surface: str):
    return dict(BASE, surface=surface, turbo_after_step=2)


def complement(k: int):
    return lambda i, _k=k: i % _k != 0


ARMS = [
    ("baseline_nockpt", {}),
    ("turbo_mlp", dict(stacr_cfg=turbo("mlp"))),
    ("turbo_mlpattn", dict(stacr_cfg=turbo("mlp+attention"))),
    ("turbo_all", dict(stacr_cfg=turbo("all"))),
    ("ckpt_k4", dict(ckpt_every=4, ckpt_offset=0)),
    (
        "ckpt_k4_plus_mlp",
        dict(ckpt_every=4, ckpt_offset=0, stacr_cfg=turbo("mlp"), stacr_block_pred=complement(4)),
    ),
    (
        "ckpt_k4_plus_mlpattn",
        dict(
            ckpt_every=4,
            ckpt_offset=0,
            stacr_cfg=turbo("mlp+attention"),
            stacr_block_pred=complement(4),
        ),
    ),
]
QUALITY_ARMS = [
    ("baseline", {}),
    ("turbo_mlp", dict(stacr_cfg=turbo("mlp"))),
    ("turbo_mlpattn", dict(stacr_cfg=turbo("mlp+attention"))),
    (
        "ckpt_k4_plus_mlp",
        dict(ckpt_every=4, ckpt_offset=0, stacr_cfg=turbo("mlp"), stacr_block_pred=complement(4)),
    ),
    (
        "ckpt_k4_plus_mlpattn",
        dict(
            ckpt_every=4,
            ckpt_offset=0,
            stacr_cfg=turbo("mlp+attention"),
            stacr_block_pred=complement(4),
        ),
    ),
]


def quality(out: Path, smoke: bool = False):
    arms = QUALITY_ARMS[:2] if smoke else QUALITY_ARMS
    seeds = SEEDS[:1] if smoke else SEEDS
    steps = 3 if smoke else 2000
    mf = 1 if smoke else 50
    print(
        f"\n########## X14-Q surface widening, quality cost  gpt2-medium b=8 s=1024 {steps} steps  {len(arms)} arms x {len(seeds)} seeds ##########",
        flush=True,
    )
    (base_peak, base_ppl) = (None, None)
    for method, extra in arms:
        for seed in seeds:
            try:
                rec = run_config(
                    model_id="gpt2-medium",
                    bs=8,
                    seq=1024,
                    steps=steps,
                    seed=seed,
                    method=method,
                    tag="surf_q/gpt2m",
                    dtype=torch.float32,
                    amp=True,
                    eight_bit=True,
                    eval_batches=2 if smoke else 16,
                    measure_from=mf,
                    baseline_peak_bytes=base_peak,
                    **extra,
                )
            except torch.cuda.OutOfMemoryError:
                emit({"tag": "surf_q/gpt2m", "method": method, "seed": seed, "status": "OOM"}, out)
                print(f"  OOM  {method} seed={seed}", flush=True)
                torch.cuda.empty_cache()
                continue
            except Exception as exc:
                emit(
                    {
                        "tag": "surf_q/gpt2m",
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
            rec["cohort"] = "surface_widen_quality"
            if method == "baseline" and base_peak is None:
                (base_peak, base_ppl) = (rec["peak_bytes"], rec["val_ppl"])
            emit(rec, out)
            sav = 100.0 * (base_peak - rec["peak_bytes"]) / base_peak if base_peak else 0.0
            dq = 100.0 * (rec["val_ppl"] - base_ppl) / base_ppl if base_ppl else 0.0
            print(
                f"  {method:22s} seed={seed} peak={rec['peak_GiB']:7.3f} GiB ({sav:+6.2f}%) t={rec['time_s']:7.1f}s ppl={rec['val_ppl']:.4f} ({dq:+6.2f}%) wrapped={rec.get('n_wrapped_layers')}",
                flush=True,
            )
            torch.cuda.empty_cache()


LONGSEQ_MODELS = [("qwen25_1_5b", "Qwen/Qwen2.5-1.5B"), ("llama32_1b", "meta-llama/Llama-3.2-1B")]
LONGSEQ_ARMS = [
    ("baseline", {}),
    ("turbo_mlp", dict(stacr_cfg=turbo("mlp"))),
    ("turbo_mlpattn", dict(stacr_cfg=turbo("mlp+attention"))),
    ("ckpt_k4", dict(ckpt_every=4, ckpt_offset=0)),
    (
        "ckpt_k4_plus_mlp",
        dict(ckpt_every=4, ckpt_offset=0, stacr_cfg=turbo("mlp"), stacr_block_pred=complement(4)),
    ),
    (
        "ckpt_k4_plus_mlpattn",
        dict(
            ckpt_every=4,
            ckpt_offset=0,
            stacr_cfg=turbo("mlp+attention"),
            stacr_block_pred=complement(4),
        ),
    ),
]


def longseq(out: Path, smoke: bool = False):
    models = LONGSEQ_MODELS[:1] if smoke else LONGSEQ_MODELS
    arms = LONGSEQ_ARMS[:3] if smoke else LONGSEQ_ARMS
    seeds = SEEDS[:1] if smoke else SEEDS
    steps = 3 if smoke else 60
    mf = 1 if smoke else 15
    seq = 1024 if smoke else 4096
    for tag, model_id in models:
        print(
            f"\n########## X14-L widening at long context  {tag}  b=1 s={seq} bf16  {len(arms)} arms x {len(seeds)} seeds ##########",
            flush=True,
        )
        base_peak = None
        for method, extra in arms:
            for seed in seeds:
                try:
                    rec = run_config(
                        model_id=model_id,
                        bs=1,
                        seq=seq,
                        steps=steps,
                        seed=seed,
                        method=method,
                        tag=f"surf_ls/{tag}/s{seq}",
                        dtype=torch.bfloat16,
                        amp=True,
                        eight_bit=True,
                        eval_batches=2,
                        measure_from=mf,
                        baseline_peak_bytes=base_peak,
                        **extra,
                    )
                except torch.cuda.OutOfMemoryError:
                    emit(
                        {
                            "tag": f"surf_ls/{tag}/s{seq}",
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
                            "tag": f"surf_ls/{tag}/s{seq}",
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
                rec["cohort"] = "surface_longseq"
                if method == "baseline" and base_peak is None:
                    base_peak = rec["peak_bytes"]
                emit(rec, out)
                sav = 100.0 * (base_peak - rec["peak_bytes"]) / base_peak if base_peak else 0.0
                print(
                    f"  {method:22s} seed={seed} peak={rec['peak_GiB']:7.3f} GiB ({sav:+6.2f}%) t={rec['time_s']:6.1f}s tok/s={rec['tokens_per_s']:8.0f} wrapped={rec.get('n_wrapped_layers')} ckpt={rec.get('n_checkpointed_blocks')}",
                    flush=True,
                )
                torch.cuda.empty_cache()


def main():
    smoke = "--smoke" in sys.argv
    out = RESULTS / ("x14_surface_modern_smoke.jsonl" if smoke else "x14_surface_modern.jsonl")
    if "quality" in sys.argv:
        quality(out, smoke)
        print("X14 DONE", flush=True)
        return
    if "longseq" in sys.argv:
        longseq(out, smoke)
        print("X14 DONE", flush=True)
        return
    models = MODELS[:1] if smoke else MODELS
    arms = ARMS[:3] if smoke else ARMS
    seeds = SEEDS[:1] if smoke else SEEDS
    steps = 3 if smoke else STEPS
    mf = 1 if smoke else MEASURE_FROM
    for tag, model_id in models:
        print(
            f"\n########## X14 {tag}  {model_id}  b=2 s=1024 fp32  {len(arms)} arms x {len(seeds)} seeds ##########",
            flush=True,
        )
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
                        tag=f"surf/{tag}",
                        dtype=torch.float32,
                        amp=True,
                        eight_bit=True,
                        eval_batches=2 if smoke else 8,
                        measure_from=mf,
                        baseline_peak_bytes=base_peak,
                        **extra,
                    )
                except torch.cuda.OutOfMemoryError:
                    emit(
                        {"tag": f"surf/{tag}", "method": method, "seed": seed, "status": "OOM"}, out
                    )
                    print(f"  OOM  {method} seed={seed}", flush=True)
                    torch.cuda.empty_cache()
                    continue
                except Exception as exc:
                    emit(
                        {
                            "tag": f"surf/{tag}",
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
                rec["cohort"] = "surface_modern"
                if method == "baseline_nockpt" and base_peak is None:
                    base_peak = rec["peak_bytes"]
                emit(rec, out)
                sav = 100.0 * (base_peak - rec["peak_bytes"]) / base_peak if base_peak else 0.0
                print(
                    f"  {method:22s} seed={seed} peak={rec['peak_GiB']:7.3f} GiB ({sav:+6.2f}%) t={rec['time_s']:6.1f}s tok/s={rec['tokens_per_s']:8.0f} wrapped={rec.get('n_wrapped_layers')} ckpt={rec.get('n_checkpointed_blocks')}",
                    flush=True,
                )
                torch.cuda.empty_cache()
    print("X14 DONE", flush=True)


if __name__ == "__main__":
    main()
