"""Freeze offload controls."""

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


FREEZE_MODELS = [
    ("qwen25_1_5b", "Qwen/Qwen2.5-1.5B"),
    ("llama32_1b", "meta-llama/Llama-3.2-1B"),
    ("qwen3_1_7b", "Qwen/Qwen3-1.7B"),
]
FREEZE_ARMS = [
    ("baseline_nockpt", {}),
    ("freeze_mlp", dict(freeze_surface=True)),
    ("turbo_vs_nockpt", dict(stacr_cfg=TURBO)),
    ("stacr_r8_vs_nockpt", dict(stacr_cfg=CFG8)),
]
OFFLOAD_ARMS = [
    ("baseline", {}),
    ("offload_all", dict(offload_every=1, offload_offset=0)),
    ("offload_k4", dict(offload_every=4, offload_offset=0)),
    ("turbo_fast", dict(stacr_cfg=TURBO)),
    ("ckpt_k4", dict(ckpt_every=4, ckpt_offset=0)),
    (
        "ckpt_k4_plus_turbo",
        dict(ckpt_every=4, ckpt_offset=0, stacr_cfg=TURBO, stacr_block_pred=complement(4)),
    ),
    (
        "offload_k4_plus_turbo",
        dict(offload_every=4, offload_offset=0, stacr_cfg=TURBO, stacr_block_pred=complement(4)),
    ),
    ("offload_all_plus_turbo", dict(offload_every=1, offload_offset=0, stacr_cfg=TURBO)),
]


def cell(
    model_id,
    tag,
    method,
    extra,
    bs,
    seq,
    steps,
    measure_from,
    dtype,
    eval_batches,
    seed,
    base_peak,
    out,
):
    try:
        rec = run_config(
            model_id=model_id,
            bs=bs,
            seq=seq,
            steps=steps,
            seed=seed,
            method=method,
            tag=tag,
            dtype=dtype,
            amp=True,
            eight_bit=True,
            eval_batches=eval_batches,
            measure_from=measure_from,
            baseline_peak_bytes=base_peak,
            **extra,
        )
    except torch.cuda.OutOfMemoryError:
        emit({"tag": tag, "method": method, "seed": seed, "status": "OOM"}, out)
        print(f"  OOM  {method} seed={seed}", flush=True)
        torch.cuda.empty_cache()
        return None
    except Exception as exc:
        emit(
            {
                "tag": tag,
                "method": method,
                "seed": seed,
                "status": "error",
                "error": f"{type(exc).__name__}: {str(exc)[:200]}",
            },
            out,
        )
        print(f"  FAIL {method} seed={seed}: {type(exc).__name__}: {exc}", flush=True)
        torch.cuda.empty_cache()
        return None
    rec["tokens_per_s"] = rec["bs"] * rec["seq"] * rec["steps_measured"] / rec["time_s"]
    emit(rec, out)
    torch.cuda.empty_cache()
    return rec


def freeze(out: Path, smoke: bool = False, dtype=torch.bfloat16, tagsuf: str = ""):
    models = FREEZE_MODELS[:1] if smoke else FREEZE_MODELS
    arms = FREEZE_ARMS[:2] if smoke else FREEZE_ARMS
    seeds = SEEDS[:1] if smoke else SEEDS
    steps = 3 if smoke else 600
    mf = 1 if smoke else 20
    dname = str(dtype).replace("torch.", "")
    print(
        f"\n########## X11-A freeze control  {len(models)} models x {len(arms)} arms x {len(seeds)} seeds, {steps} steps, {dname} ##########",
        flush=True,
    )
    for tag, model_id in models:
        print(f"\n=== freeze{tagsuf}/{tag} ===", flush=True)
        (base_peak, base_ppl) = (None, None)
        for method, extra in arms:
            for seed in seeds:
                rec = cell(
                    model_id,
                    f"ctl_freeze{tagsuf}/{tag}",
                    method,
                    extra,
                    2,
                    1024,
                    steps,
                    mf,
                    dtype,
                    2 if smoke else 16,
                    seed,
                    base_peak,
                    out,
                )
                if rec is None:
                    continue
                if method == "baseline_nockpt":
                    if base_peak is None:
                        (base_peak, base_ppl) = (rec["peak_bytes"], rec["val_ppl"])
                dq = 100.0 * (rec["val_ppl"] - base_ppl) / base_ppl if base_ppl else 0.0
                print(
                    f"  {method:22s} seed={seed} peak={rec['peak_GiB']:7.3f} GiB ppl={rec['val_ppl']:.4f} ({dq:+6.2f}%) frozen={rec.get('n_frozen_params')} wrapped={rec.get('n_wrapped_layers')}",
                    flush=True,
                )


def offload(out: Path, smoke: bool = False):
    arms = OFFLOAD_ARMS[:3] if smoke else OFFLOAD_ARMS
    seeds = SEEDS[:1] if smoke else SEEDS
    steps = 3 if smoke else 120
    mf = 1 if smoke else 20
    print(
        f"\n########## X11-B offload comparator  GPT-2 Medium b=8 s=1024, {len(arms)} arms x {len(seeds)} seeds, {steps} steps ##########",
        flush=True,
    )
    base_peak = None
    for method, extra in arms:
        for seed in seeds:
            rec = cell(
                "gpt2-medium",
                "ctl_offload",
                method,
                extra,
                8,
                1024,
                steps,
                mf,
                torch.float32,
                2 if smoke else 16,
                seed,
                base_peak,
                out,
            )
            if rec is None:
                continue
            if method == "baseline" and base_peak is None:
                base_peak = rec["peak_bytes"]
            sav = 100.0 * (base_peak - rec["peak_bytes"]) / base_peak if base_peak else 0.0
            print(
                f"  {method:24s} seed={seed} peak={rec['peak_GiB']:7.3f} GiB ({sav:+6.2f}%) t={rec['time_s']:6.1f}s tok/s={rec['tokens_per_s']:8.0f} offloaded={rec.get('n_offloaded_blocks')} wrapped={rec.get('n_wrapped_layers')} ckpt={rec.get('n_checkpointed_blocks')}",
                flush=True,
            )


def main():
    smoke = "--smoke" in sys.argv
    out = RESULTS / ("x11_controls_smoke.jsonl" if smoke else "x11_controls.jsonl")
    want = [a for a in sys.argv[1:] if a in ("freeze", "offload", "freeze32")] or [
        "offload",
        "freeze",
    ]
    if "offload" in want:
        offload(out, smoke)
    if "freeze" in want:
        freeze(out, smoke)
    if "freeze32" in want:
        freeze(out, smoke, torch.float32, "32")
    print("X11 DONE", flush=True)


if __name__ == "__main__":
    main()
