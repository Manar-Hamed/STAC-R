"""Bert sst2."""

from __future__ import annotations
import torch
import sys
import time
from pathlib import Path
from .common import (
    RESULTS,
    emit,
    load_model,
    build_optimizer,
    amp_ctx,
    free_all,
    set_seed,
    gib,
    sync,
    apply_selective_checkpointing,
)
from stacr.surgery import apply_stacr, set_step

SEEDS = [42, 123, 456]
CFG = {
    "turbo_after_step": None,
    "rank": 8,
    "residual_budget": 0.005,
    "residual_bits": 8,
    "surface": "mlp",
    "min_features": 256,
    "chunk_rows": 512,
}


def sst2(tok, split, bs, n, dev, seq=128):
    from datasets import load_dataset

    ds = load_dataset("nyu-mll/glue", "sst2", split=split)
    out = []
    for i in range(min(n, len(ds) // bs)):
        rows = ds[i * bs : (i + 1) * bs]
        enc = tok(
            rows["sentence"],
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=seq,
        )
        out.append(
            {
                "input_ids": enc["input_ids"].to(dev),
                "attention_mask": enc["attention_mask"].to(dev),
                "labels": torch.tensor(rows["label"], device=dev),
            }
        )
    return out


def run(method, cfg, seed, bs=32, steps=400, ckpt=0):
    from transformers import AutoTokenizer

    free_all()
    set_seed(seed)
    tok = AutoTokenizer.from_pretrained("bert-base-uncased")
    m = load_model("bert-base-uncased", "cuda")
    if ckpt:
        apply_selective_checkpointing(m, ckpt)
    layers = apply_stacr(m, {"turbo_after_step": None, **cfg}) if cfg else []
    opt = build_optimizer(m, 2e-05, True)
    tr = sst2(tok, "train", bs, steps, "cuda")
    va = sst2(tok, "validation", bs, 27, "cuda")
    sync()
    t0 = None
    for i, b in enumerate(tr):
        if layers:
            set_step(layers, i)
        if i == min(8, steps - 1):
            sync()
            torch.cuda.reset_peak_memory_stats()
            t0 = time.perf_counter()
        opt.zero_grad(set_to_none=True)
        with amp_ctx(True):
            loss = m(**b).loss
        loss.backward()
        opt.step()
    sync()
    elapsed = time.perf_counter() - t0
    m.eval()
    ok = tot = 0
    with torch.no_grad():
        for b in va:
            with amp_ctx(True):
                pred = m(**b).logits.argmax(-1)
            ok += int((pred == b["labels"]).sum())
            tot += b["labels"].numel()
    rec = {
        "tag": "bert_sst2",
        "method": method,
        "seed": seed,
        "peak_GiB": gib(torch.cuda.max_memory_allocated()),
        "peak_bytes": int(torch.cuda.max_memory_allocated()),
        "time_s": elapsed,
        "accuracy": 100.0 * ok / max(tot, 1),
        "n_wrapped_layers": len(layers),
    }
    del m, opt, layers, tr, va
    free_all()
    return rec


def main():
    out = RESULTS / "b_bert_sst2.jsonl"
    bp = bt = None
    rows = [
        ("baseline", None, 0),
        ("stacr_r8", CFG, 0),
        ("turbo_fast", dict(CFG, turbo_after_step=2), 0),
        ("ckpt_full", None, 1),
    ]
    for method, cfg, ck in rows:
        for seed in SEEDS:
            try:
                r = run(method, cfg, seed, ckpt=ck)
            except Exception as exc:
                emit(
                    {
                        "tag": "bert_sst2",
                        "method": method,
                        "seed": seed,
                        "status": "error",
                        "error": f"{type(exc).__name__}: {exc}"[:250],
                    },
                    out,
                )
                continue
            if method == "baseline" and bp is None:
                (bp, bt) = (r["peak_bytes"], r["time_s"])
            if bt:
                r["time_x"] = r["time_s"] / bt
                r["measured_saving_pct"] = 100.0 * (bp - r["peak_bytes"]) / bp
            emit(r, out)
            print(
                f"  >>> {method:<14} s{seed} mem {r.get('measured_saving_pct', 0):+.2f}% acc {r['accuracy']:.2f} time {r.get('time_x', 1.0):.2f}x",
                flush=True,
            )


if __name__ == "__main__":
    main()
