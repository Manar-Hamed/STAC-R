"""Modern composition."""

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
(STEPS, MEASURE_FROM) = (40, 10)
SEEDS = [42, 123, 456]
MODELS = [
    ("qwen25_1_5b", "Qwen/Qwen2.5-1.5B", 2, 1024),
    ("llama32_1b", "meta-llama/Llama-3.2-1B", 2, 1024),
    ("qwen3_1_7b", "Qwen/Qwen3-1.7B", 2, 1024),
]


def complement(k: int):
    return lambda i, _k=k: i % _k != 0


def arms():
    out = [
        ("baseline_nockpt", {}),
        ("stacr_r8_vs_nockpt", dict(stacr_cfg=CFG8)),
        ("turbo_vs_nockpt", dict(stacr_cfg=TURBO)),
        ("baseline_ckptfull", dict(ckpt_every=1, ckpt_offset=0)),
        ("stacr_on_ckptfull", dict(ckpt_every=1, ckpt_offset=0, stacr_cfg=CFG8)),
    ]
    for k in (4, 2):
        out.append((f"ckpt_k{k}", dict(ckpt_every=k, ckpt_offset=0)))
        out.append(
            (
                f"ckpt_k{k}_plus_turbo",
                dict(ckpt_every=k, ckpt_offset=0, stacr_cfg=TURBO, stacr_block_pred=complement(k)),
            )
        )
        out.append(
            (
                f"ckpt_k{k}_plus_stacr",
                dict(ckpt_every=k, ckpt_offset=0, stacr_cfg=CFG8, stacr_block_pred=complement(k)),
            )
        )
    return out


def main():
    out = RESULTS / "x5_modern_composed.jsonl"
    for tag, model_id, bs, seq in MODELS:
        print(f"\n=== {tag}  {model_id}  b={bs} s={seq} ===", flush=True)
        base_peak = None
        for method, extra in arms():
            for seed in SEEDS:
                try:
                    rec = run_config(
                        model_id=model_id,
                        bs=bs,
                        seq=seq,
                        steps=STEPS,
                        seed=seed,
                        method=method,
                        tag=tag,
                        dtype=torch.bfloat16,
                        amp=True,
                        eight_bit=True,
                        eval_batches=2,
                        measure_from=MEASURE_FROM,
                        baseline_peak_bytes=base_peak,
                        **extra,
                    )
                except torch.cuda.OutOfMemoryError:
                    emit({"tag": tag, "method": method, "seed": seed, "status": "OOM"}, out)
                    print(f"  OOM {method} seed={seed}", flush=True)
                    torch.cuda.empty_cache()
                    continue
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
                    print(f"  FAIL {method} seed={seed}: {type(exc).__name__}", flush=True)
                    torch.cuda.empty_cache()
                    continue
                rec["tokens_per_s"] = rec["bs"] * rec["seq"] * rec["steps_measured"] / rec["time_s"]
                if method == "baseline_nockpt" and base_peak is None:
                    base_peak = rec["peak_bytes"]
                    print(f"  uncheckpointed baseline peak = {rec['peak_GiB']:.3f} GiB", flush=True)
                emit(rec, out)
                print(
                    f"  {method:24s} seed={seed} peak={rec['peak_GiB']:7.3f} GiB t={rec['time_s']:6.1f}s wrapped={rec.get('n_wrapped_layers')} ckpt={rec.get('n_checkpointed_blocks')}",
                    flush=True,
                )
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
