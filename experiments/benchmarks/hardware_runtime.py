"""Hardware runtime."""

from __future__ import annotations
import torch
import os
import sys
from pathlib import Path
from .common import RESULTS, emit, run_config

HW = os.environ.get("STACR_HWTAG", "a100")
SEEDS = [42, 123, 456]
CFG8 = {
    "rank": 8,
    "residual_budget": 0.005,
    "residual_bits": 8,
    "surface": "mlp",
    "min_features": 256,
    "chunk_rows": 512,
}
TURBO = dict(CFG8, turbo_after_step=2)
PAPERCFG = dict(
    CFG8, update_interval=100, basis_dtype="float16", freeze_after_warmup=True, warmup_steps=20
)


def gpu_gib():
    return torch.cuda.get_device_properties(0).total_memory / 1024**3


def phase1_runtime(out):
    rows = [
        ("baseline", {}),
        ("stacr_r8", dict(stacr_cfg=CFG8)),
        ("stacr_paper_cfg", dict(stacr_cfg=PAPERCFG)),
        ("turbo_fast", dict(stacr_cfg=TURBO)),
        ("ckpt_full", dict(ckpt_every=1)),
        ("ckpt_k2", dict(ckpt_every=2)),
        (
            "ckpt_k2_plus_stacr",
            dict(
                ckpt_every=2, ckpt_offset=0, stacr_cfg=CFG8, stacr_block_pred=lambda i: i % 2 == 1
            ),
        ),
    ]
    base_p = base_t = None
    for method, extra in rows:
        for seed in SEEDS:
            try:
                r = run_config(
                    model_id="gpt2-medium",
                    bs=8,
                    seq=1024,
                    steps=400,
                    seed=seed,
                    method=method,
                    baseline_peak_bytes=base_p,
                    tag=f"{HW}_gpt2m",
                    **extra,
                )
            except Exception as exc:
                emit(
                    {
                        "tag": f"{HW}_gpt2m",
                        "method": method,
                        "seed": seed,
                        "status": "error",
                        "error": f"{type(exc).__name__}: {exc}"[:250],
                    },
                    out,
                )
                continue
            if method == "baseline" and base_p is None:
                (base_p, base_t) = (r["peak_bytes"], r["time_s"])
            if base_t:
                r["time_x"] = r["time_s"] / base_t
                r["measured_saving_pct"] = 100.0 * (base_p - r["peak_bytes"]) / base_p
            emit(r, out)
            print(
                f"  >>> {method:<20} {r.get('time_x', 1.0):.3f}x  mem {r.get('measured_saving_pct', 0.0):+.2f}%  PPL {r['val_ppl']:.4f}",
                flush=True,
            )


def phase2_scale(out):
    cap = gpu_gib()
    if cap >= 70:
        models = [
            ("qwen25_7b", "Qwen/Qwen2.5-7B", [(1, 1024), (2, 1024), (4, 1024)]),
            ("llama31_8b", "meta-llama/Llama-3.1-8B", [(1, 1024), (2, 1024)]),
        ]
    else:
        models = [
            ("qwen25_3b", "Qwen/Qwen2.5-3B", [(2, 1024), (4, 1024)]),
            ("llama32_3b", "meta-llama/Llama-3.2-3B", [(2, 1024), (4, 1024)]),
        ]
    for tagm, mid, shapes in models:
        for bs, seq in shapes:
            tag = f"{HW}_scale/{tagm}/b{bs}_s{seq}"
            try:
                base = run_config(
                    model_id=mid,
                    bs=bs,
                    seq=seq,
                    steps=20,
                    seed=42,
                    method="baseline",
                    amp=True,
                    eight_bit=True,
                    dtype=torch.bfloat16,
                    eval_batches=2,
                    measure_from=5,
                    tag=tag,
                )
            except Exception as exc:
                emit(
                    {
                        "tag": tag,
                        "method": "baseline",
                        "status": "OOM_or_error",
                        "error": f"{type(exc).__name__}: {exc}"[:200],
                    },
                    out,
                )
                continue
            emit(base, out)
            bp = base["peak_bytes"]
            for surf, cfg in [
                ("mlp", dict(CFG8)),
                ("mlp_attn", dict(CFG8, surface="mlp+attention")),
            ]:
                try:
                    r = run_config(
                        model_id=mid,
                        bs=bs,
                        seq=seq,
                        steps=20,
                        seed=42,
                        method=f"stacr_{surf}_r8",
                        stacr_cfg=cfg,
                        amp=True,
                        eight_bit=True,
                        dtype=torch.bfloat16,
                        eval_batches=2,
                        measure_from=5,
                        baseline_peak_bytes=bp,
                        tag=tag,
                    )
                except Exception as exc:
                    emit(
                        {
                            "tag": tag,
                            "method": f"stacr_{surf}_r8",
                            "status": "OOM_or_error",
                            "error": f"{type(exc).__name__}: {exc}"[:200],
                        },
                        out,
                    )
                    continue
                r["measured_saving_pct"] = 100.0 * (bp - r["peak_bytes"]) / bp
                r["baseline_peak_GiB"] = base["peak_GiB"]
                emit(r, out)
                print(f"  >>> {tag}/{surf}: {r['measured_saving_pct']:+.2f}%", flush=True)


def main():
    out = RESULTS / f"{HW}.jsonl"
    cap = gpu_gib()
    name = torch.cuda.get_device_properties(0).name
    print(f"[{HW}] device={name}  capacity={cap:.1f} GiB", flush=True)
    emit(
        {
            "tag": f"{HW}_env",
            "device": name,
            "capacity_GiB": cap,
            "torch": torch.__version__,
            "transformers": __import__("transformers").__version__,
        },
        out,
    )
    phase1_runtime(out)
    phase2_scale(out)


if __name__ == "__main__":
    main()
