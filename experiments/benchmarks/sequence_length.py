"""Sequence length."""

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
STEPS = 40
MEASURE_FROM = 10
MODELS = [("qwen25_1_5b", "Qwen/Qwen2.5-1.5B"), ("llama32_1b", "meta-llama/Llama-3.2-1B")]
SWEEP_LENGTHS = [1024, 2048, 4096, 8192]
PROBE_LENGTHS = [1024, 2048, 3072, 4096, 6144, 8192, 12288, 16384]


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
]


def one(
    model_id: str,
    tag: str,
    method: str,
    extra: dict,
    bs: int,
    seq: int,
    seed: int,
    base_peak,
    out: Path,
):
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
        emit(
            {"tag": tag, "method": method, "seed": seed, "bs": bs, "seq": seq, "status": "OOM"}, out
        )
        print(f"  OOM  {method:20s} s={seq}", flush=True)
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
                "status": "error",
                "error": f"{type(exc).__name__}: {str(exc)[:200]}",
            },
            out,
        )
        print(f"  FAIL {method:20s} s={seq}: {type(exc).__name__}: {exc}", flush=True)
        torch.cuda.empty_cache()
        return None
    rec["tokens_per_s"] = rec["bs"] * rec["seq"] * rec["steps_measured"] / rec["time_s"]
    emit(rec, out)
    torch.cuda.empty_cache()
    return rec


def sweep(out: Path):
    print("\n########## X7-A  fixed-s sweep, b=1, bf16 ##########", flush=True)
    for tag, model_id in MODELS:
        for seq in SWEEP_LENGTHS:
            t = f"sl_sweep/{tag}/s{seq}"
            print(f"\n=== {t} ===", flush=True)
            base_peak = None
            for method, extra in ARMS:
                for seed in SEEDS:
                    rec = one(model_id, t, method, extra, 1, seq, seed, base_peak, out)
                    if rec is None:
                        continue
                    if method == "baseline_nockpt" and base_peak is None:
                        base_peak = rec["peak_bytes"]
                    sav = 100.0 * (base_peak - rec["peak_bytes"]) / base_peak if base_peak else 0.0
                    print(
                        f"  {method:20s} seed={seed} peak={rec['peak_GiB']:7.3f} GiB ({sav:+6.2f}%) t={rec['time_s']:6.1f}s tok/s={rec['tokens_per_s']:8.0f} wrapped={rec.get('n_wrapped_layers')} ckpt={rec.get('n_checkpointed_blocks')}",
                        flush=True,
                    )


def maxlen(out: Path):
    print(
        "\n########## X7-B  longest trainable sequence, b=1, bf16, one 32 GiB card ##########",
        flush=True,
    )
    for tag, model_id in MODELS:
        print(f"\n=== {tag} ===", flush=True)
        for method, extra in ARMS:
            best = None
            for seq in PROBE_LENGTHS:
                rec = one(model_id, f"sl_max/{tag}", method, extra, 1, seq, 42, None, out)
                if rec is None:
                    break
                best = (seq, rec["peak_GiB"], rec["tokens_per_s"])
            if best:
                print(
                    f"  {method:20s} longest s={best[0]:6d}  peak={best[1]:7.3f} GiB  tok/s={best[2]:8.0f}",
                    flush=True,
                )
            else:
                print(f"  {method:20s} did not train at s={PROBE_LENGTHS[0]}", flush=True)


PROBE_BATCHES = [1, 2, 3, 4, 6, 8]


def maxbatch(out: Path, seq: int = 4096):
    print(
        f"\n########## X7-C largest trainable batch at s={seq}, bf16, one 32 GiB card ##########",
        flush=True,
    )
    for tag, model_id in MODELS:
        print(f"\n=== {tag}  s={seq} ===", flush=True)
        for method, extra in ARMS:
            best = None
            for bs in PROBE_BATCHES:
                rec = one(model_id, f"sl_maxb/{tag}/s{seq}", method, extra, bs, seq, 42, None, out)
                if rec is None:
                    break
                best = (bs, rec["peak_GiB"], rec["tokens_per_s"])
            if best:
                print(
                    f"  {method:20s} largest b={best[0]:3d}  peak={best[1]:7.3f} GiB  tok/s={best[2]:8.0f}",
                    flush=True,
                )
            else:
                print(f"  {method:20s} did not train at b={PROBE_BATCHES[0]}", flush=True)


def main():
    out = RESULTS / "x7_seqlen.jsonl"
    want = [a for a in sys.argv[1:] if a in ("sweep", "maxlen", "maxbatch")] or ["sweep", "maxlen"]
    only = [int(a) for a in sys.argv[1:] if a.isdigit()]
    if only:
        global SWEEP_LENGTHS, PROBE_LENGTHS
        SWEEP_LENGTHS = [s for s in SWEEP_LENGTHS if s in only] or only
        PROBE_LENGTHS = [s for s in PROBE_LENGTHS if s in only] or only
        print(f"restricted to lengths {only}", flush=True)
    if "sweep" in want:
        sweep(out)
    if "maxlen" in want:
        maxlen(out)
    if "maxbatch" in want:
        maxbatch(out, seq=only[0] if only else 4096)


if __name__ == "__main__":
    main()
