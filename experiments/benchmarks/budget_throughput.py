"""Budget throughput."""

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


def complement(k: int):
    return lambda i, _k=k: i % _k != 0


CONFIGS = [
    ("baseline", {}),
    ("turbo_fast", dict(stacr_cfg=TURBO)),
    ("stacr_r8", dict(stacr_cfg=CFG8)),
    ("ckpt_k4", dict(ckpt_every=4, ckpt_offset=0)),
    ("ckpt_k2", dict(ckpt_every=2, ckpt_offset=0)),
    ("ckpt_full", dict(ckpt_every=1, ckpt_offset=0)),
    (
        "ckpt_k4_plus_turbo",
        dict(ckpt_every=4, ckpt_offset=0, stacr_cfg=TURBO, stacr_block_pred=complement(4)),
    ),
    (
        "ckpt_k2_plus_turbo",
        dict(ckpt_every=2, ckpt_offset=0, stacr_cfg=TURBO, stacr_block_pred=complement(2)),
    ),
]
SWEEPS = [
    ("gpt2l", "gpt2-large", 1024, torch.float32, [2, 4, 6, 8, 10, 12, 14]),
    ("qwen15", "Qwen/Qwen2.5-1.5B", 1024, torch.bfloat16, [2, 4, 6, 8, 10, 12]),
]
SEQ_SWEEPS = [("gpt2l_seq", "gpt2-large", 2, torch.float32, [1024, 2048, 4096, 8192])]


def one(tag, model_id, bs, seq, dtype, method, extra, out, base_peak):
    try:
        rec = run_config(
            model_id=model_id,
            bs=bs,
            seq=seq,
            steps=STEPS,
            seed=42,
            method=method,
            tag=tag,
            dtype=dtype,
            amp=True,
            eight_bit=True,
            eval_batches=2,
            measure_from=MEASURE_FROM,
            baseline_peak_bytes=base_peak,
            **extra,
        )
    except torch.cuda.OutOfMemoryError:
        emit(
            {"tag": tag, "method": method, "bs": bs, "seq": seq, "fits": False, "status": "OOM"},
            out,
        )
        print(f"  {method:22s} bs={bs:<3d} seq={seq:<5d} OOM", flush=True)
        torch.cuda.empty_cache()
        return None
    except Exception as exc:
        emit(
            {
                "tag": tag,
                "method": method,
                "bs": bs,
                "seq": seq,
                "fits": False,
                "status": "error",
                "error": f"{type(exc).__name__}: {str(exc)[:160]}",
            },
            out,
        )
        print(f"  {method:22s} bs={bs:<3d} seq={seq:<5d} FAIL {type(exc).__name__}", flush=True)
        torch.cuda.empty_cache()
        return None
    rec["fits"] = True
    rec["tokens_per_s"] = rec["bs"] * rec["seq"] * rec["steps_measured"] / rec["time_s"]
    emit(rec, out)
    print(
        f"  {method:22s} bs={bs:<3d} seq={seq:<5d} peak={rec['peak_GiB']:7.3f} GiB {rec['tokens_per_s']:9.0f} tok/s",
        flush=True,
    )
    torch.cuda.empty_cache()
    return rec


def main():
    out = RESULTS / "x4_budget_throughput.jsonl"
    for tag, model_id, seq, dtype, batches in SWEEPS:
        print(f"\n=== batch sweep {tag} {model_id} seq={seq} ===", flush=True)
        for method, extra in CONFIGS:
            base_peak = None
            for bs in batches:
                rec = one(tag, model_id, bs, seq, dtype, method, extra, out, base_peak)
                if rec is None:
                    break
                if method == "baseline" and base_peak is None:
                    base_peak = rec["peak_bytes"]
    for tag, model_id, bs, dtype, seqs in SEQ_SWEEPS:
        print(f"\n=== sequence sweep {tag} {model_id} bs={bs} ===", flush=True)
        for method, extra in CONFIGS:
            for seq in seqs:
                if one(tag, model_id, bs, seq, dtype, method, extra, out, None) is None:
                    break


if __name__ == "__main__":
    main()
