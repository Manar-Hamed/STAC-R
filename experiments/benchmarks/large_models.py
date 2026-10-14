"""Large models."""

from __future__ import annotations
import torch
import os
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
HEADLINE = ("qwen25_7b", "Qwen/Qwen2.5-7B")
LADDER_MODELS = [
    HEADLINE,
    ("llama31_8b", "meta-llama/Llama-3.1-8B"),
    ("mistral7b_v3", "mistralai/Mistral-7B-v0.3"),
]
PERSISTENT_GIB = {"qwen25_7b": 45.6, "llama31_8b": 48.2, "mistral7b_v3": 43.5}
_only_model = os.environ.get("STACR_MODEL", "").strip()
if _only_model:
    LADDER_MODELS = [m for m in LADDER_MODELS if m[0] == _only_model]
    if not LADDER_MODELS:
        raise SystemExit(f"STACR_MODEL={_only_model!r} matches no model in LADDER_MODELS")
LONGCTX_LENGTHS = [1024, 2048, 4096, 8192]
LADDER_SEQ = 4096
PROBE_LENGTHS = [2048, 4096, 8192, 12288, 16384, 24576, 32768]
PROBE_BATCHES = [1, 2, 3, 4, 6, 8]


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
_only_arm = os.environ.get("STACR_ARM")
if _only_arm:
    ARMS = [a for a in ARMS if a[0] == _only_arm]
    if not ARMS:
        raise SystemExit(f"STACR_ARM={_only_arm!r} matches no arm in this file")


def one(model_id, tag, method, extra, bs, seq, seed, base_peak, out, steps=STEPS):
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
            **extra,
        )
    except torch.cuda.OutOfMemoryError:
        emit(
            {"tag": tag, "method": method, "seed": seed, "bs": bs, "seq": seq, "status": "OOM"}, out
        )
        print(f"  OOM  {method:20s} b={bs} s={seq}", flush=True)
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
        print(f"  FAIL {method:20s} b={bs} s={seq}: {type(exc).__name__}: {exc}", flush=True)
        torch.cuda.empty_cache()
        return None
    rec["tokens_per_s"] = rec["bs"] * rec["seq"] * rec["steps_measured"] / rec["time_s"]
    emit(rec, out)
    torch.cuda.empty_cache()
    return rec


def _report(method, seed, rec, base_peak, persistent=None):
    sav = 100.0 * (base_peak - rec["peak_bytes"]) / base_peak if base_peak else 0.0
    extra = ""
    if persistent and rec["peak_GiB"] > persistent:
        head = 100.0 * (rec["peak_GiB"] - persistent) / rec["peak_GiB"]
        extra = f" surf<={head:4.1f}%"
    print(
        f"  {method:20s} seed={seed} peak={rec['peak_GiB']:7.3f} GiB ({sav:+6.2f}%){extra} t={rec['time_s']:6.1f}s tok/s={rec['tokens_per_s']:8.0f} wrapped={rec.get('n_wrapped_layers')} ckpt={rec.get('n_checkpointed_blocks')} attn={rec.get('attn_impl')} ppl={rec.get('val_ppl')}",
        flush=True,
    )


def longctx(out: Path):
    (tag_, model_id) = HEADLINE
    print(f"\n########## X14-A  {model_id}  b=1, bf16, s sweep ##########", flush=True)
    for seq in LONGCTX_LENGTHS:
        t = f"s7b_ctx/{tag_}/s{seq}"
        print(f"\n=== {t} ===", flush=True)
        base_peak = None
        for method, extra in ARMS:
            for seed in SEEDS:
                rec = one(model_id, t, method, extra, 1, seq, seed, base_peak, out)
                if rec is None:
                    continue
                if method == "baseline_nockpt" and base_peak is None:
                    base_peak = rec["peak_bytes"]
                _report(method, seed, rec, base_peak, PERSISTENT_GIB.get(tag_))


def ladder(out: Path):
    print(f"\n########## X14-B  7B/8B ladder, b=1, s={LADDER_SEQ}, bf16 ##########", flush=True)
    for tag_, model_id in LADDER_MODELS:
        t = f"s7b_ladder/{tag_}/s{LADDER_SEQ}"
        print(f"\n=== {t} ===", flush=True)
        base_peak = None
        for method, extra in ARMS:
            for seed in SEEDS:
                rec = one(model_id, t, method, extra, 1, LADDER_SEQ, seed, base_peak, out)
                if rec is None:
                    continue
                if method == "baseline_nockpt" and base_peak is None:
                    base_peak = rec["peak_bytes"]
                _report(method, seed, rec, base_peak, PERSISTENT_GIB.get(tag_))


BIGBATCH_SHAPES = [(2, 2048), (4, 1024)]


def bigbatch(out: Path):
    (tag_, model_id) = HEADLINE
    print(f"\n########## X14-E  tokens per step, {model_id}, bf16 ##########", flush=True)
    for bs, seq in BIGBATCH_SHAPES:
        t = f"s7b_tok/{tag_}/b{bs}s{seq}"
        print(f"\n=== {t}  ({bs * seq} tokens/step) ===", flush=True)
        base_peak = None
        for method, extra in ARMS:
            for seed in SEEDS:
                rec = one(model_id, t, method, extra, bs, seq, seed, base_peak, out)
                if rec is None:
                    continue
                if method == "baseline_nockpt" and base_peak is None:
                    base_peak = rec["peak_bytes"]
                _report(method, seed, rec, base_peak, PERSISTENT_GIB.get(tag_))


def maxlen(out: Path):
    (tag_, model_id) = HEADLINE
    print(
        f"\n########## X14-C  longest trainable s, {model_id}, one 80 GiB card ##########",
        flush=True,
    )
    for method, extra in ARMS:
        best = None
        for seq in PROBE_LENGTHS:
            rec = one(model_id, f"s7b_max/{tag_}", method, extra, 1, seq, 42, None, out)
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


def maxbatch(out: Path, seq: int = 4096):
    (tag_, model_id) = HEADLINE
    print(
        f"\n########## X14-D  largest trainable b at s={seq}, {model_id}, 80 GiB ##########",
        flush=True,
    )
    for method, extra in ARMS:
        best = None
        for bs in PROBE_BATCHES:
            rec = one(model_id, f"s7b_maxb/{tag_}/s{seq}", method, extra, bs, seq, 42, None, out)
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


def smoke(out: Path):
    print("=== X14 smoke: gpt2, b=1, s=128, 4 steps ===", flush=True)
    ok = True
    for method, extra in ARMS:
        rec = one("gpt2", "s7b_smoke", method, extra, 1, 128, 42, None, out, steps=4)
        if rec is None:
            ok = False
            continue
        print(
            f"  smoke {method:20s} ok wrapped={rec['n_wrapped_layers']} ckpt={rec['n_checkpointed_blocks']} attn={rec.get('attn_impl')}",
            flush=True,
        )
    print("SMOKE OK" if ok else "SMOKE FAILED", flush=True)
    return 0 if ok else 1


def main():
    out = RESULTS / "x14_scale7b.jsonl"
    if "--smoke" in sys.argv:
        global MEASURE_FROM
        MEASURE_FROM = 1
        sys.exit(smoke(out))
    want = [
        a for a in sys.argv[1:] if a in ("longctx", "ladder", "bigbatch", "maxlen", "maxbatch")
    ] or ["longctx", "ladder"]
    only = [int(a) for a in sys.argv[1:] if a.isdigit()]
    if only:
        global LONGCTX_LENGTHS, PROBE_LENGTHS
        LONGCTX_LENGTHS = [s for s in LONGCTX_LENGTHS if s in only] or only
        PROBE_LENGTHS = [s for s in PROBE_LENGTHS if s in only] or only
        print(f"restricted to lengths {only}", flush=True)
    if "longctx" in want:
        longctx(out)
    if "ladder" in want:
        ladder(out)
    if "bigbatch" in want:
        bigbatch(out)
    if "maxlen" in want:
        maxlen(out)
    if "maxbatch" in want:
        maxbatch(out, seq=only[0] if only else 4096)


if __name__ == "__main__":
    main()
