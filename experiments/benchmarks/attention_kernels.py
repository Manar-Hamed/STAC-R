"""Attention kernels."""

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
SEEDS = [42, 123]
STEPS = 25
MEASURE_FROM = 8
COHORTS = {
    "small": dict(model_id="Qwen/Qwen2.5-1.5B", seq=4096, bs=1),
    "big": dict(model_id="Qwen/Qwen2.5-7B", seq=4096, bs=1),
}
IMPLS = ["sdpa", "flash_attention_2", "eager"]


def complement(k: int):
    return lambda i, _k=k: i % _k != 0


ARMS = [
    ("baseline_nockpt", {}),
    ("turbo_vs_nockpt", dict(stacr_cfg=TURBO)),
    (
        "ckpt_k4_plus_turbo",
        dict(ckpt_every=4, ckpt_offset=0, stacr_cfg=TURBO, stacr_block_pred=complement(4)),
    ),
]


def one(model_id, tag, method, extra, bs, seq, seed, attn, base_peak, out, steps=STEPS):
    try:
        rec = run_config(
            model_id=model_id,
            bs=bs,
            seq=seq,
            steps=steps,
            seed=seed,
            method=method,
            tag=tag,
            dtype=torch.bfloat16,
            amp=True,
            eight_bit=True,
            eval_batches=2,
            measure_from=MEASURE_FROM,
            baseline_peak_bytes=base_peak,
            attn_impl=attn,
            **extra,
        )
    except torch.cuda.OutOfMemoryError:
        emit(
            {
                "tag": tag,
                "method": method,
                "seed": seed,
                "bs": bs,
                "seq": seq,
                "attn_requested": attn,
                "status": "OOM",
            },
            out,
        )
        print(f"  OOM  {method:20s} attn={attn}", flush=True)
        torch.cuda.empty_cache()
        return None
    except Exception as exc:
        emit(
            {
                "tag": tag,
                "method": method,
                "seed": seed,
                "bs": bs,
                "seq": seq,
                "attn_requested": attn,
                "status": "error",
                "error": f"{type(exc).__name__}: {str(exc)[:200]}",
            },
            out,
        )
        print(f"  FAIL {method:20s} attn={attn}: {type(exc).__name__}: {exc}", flush=True)
        torch.cuda.empty_cache()
        return None
    rec["attn_requested"] = attn
    rec["tokens_per_s"] = rec["bs"] * rec["seq"] * rec["steps_measured"] / rec["time_s"]
    emit(rec, out)
    torch.cuda.empty_cache()
    return rec


def run_cohort(name: str, spec: dict, out: Path):
    print(
        f"\n########## X15  {name}  {spec['model_id']}  b={spec['bs']} s={spec['seq']} ##########",
        flush=True,
    )
    for attn in IMPLS:
        t = f"attn/{name}/{attn}"
        print(f"\n=== {t} ===", flush=True)
        base_peak = None
        base_gib = None
        for method, extra in ARMS:
            for seed in SEEDS:
                rec = one(
                    spec["model_id"],
                    t,
                    method,
                    extra,
                    spec["bs"],
                    spec["seq"],
                    seed,
                    attn,
                    base_peak,
                    out,
                )
                if rec is None:
                    continue
                if method == "baseline_nockpt" and base_peak is None:
                    (base_peak, base_gib) = (rec["peak_bytes"], rec["peak_GiB"])
                mib = (base_peak - rec["peak_bytes"]) / 2**20 if base_peak else 0.0
                pct = 100.0 * (base_peak - rec["peak_bytes"]) / base_peak if base_peak else 0.0
                print(
                    f"  {method:20s} seed={seed} saved={mib:8.1f} MiB ({pct:+6.2f}%) peak={rec['peak_GiB']:7.3f} base={(base_gib if base_gib else 0):7.3f} resolved={rec.get('attn_impl')} tok/s={rec['tokens_per_s']:8.0f}",
                    flush=True,
                )


def main():
    out = RESULTS / "x15_attnimpl.jsonl"
    if "--smoke" in sys.argv:
        global MEASURE_FROM, SEEDS, IMPLS
        (MEASURE_FROM, SEEDS) = (1, [42])
        IMPLS = ["sdpa", "eager"]
        ok = True
        for attn in IMPLS:
            for method, extra in ARMS:
                if (
                    one("gpt2", "attn_smoke", method, extra, 1, 128, 42, attn, None, out, steps=4)
                    is None
                ):
                    ok = False
        print("SMOKE OK" if ok else "SMOKE FAILED", flush=True)
        sys.exit(0 if ok else 1)
    want = [a for a in sys.argv[1:] if a in COHORTS] or list(COHORTS)
    for name in want:
        run_cohort(name, COHORTS[name], out)


if __name__ == "__main__":
    main()
