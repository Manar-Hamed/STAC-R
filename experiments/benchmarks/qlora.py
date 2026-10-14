"""Qlora."""

from __future__ import annotations
import torch
import sys
import time
from pathlib import Path
from .common import (
    RESULTS,
    emit,
    build_optimizer,
    amp_ctx,
    free_all,
    get_batches,
    gib,
    set_seed,
    sync,
)
from stacr.surgery import set_step
from .lora import _apply_mlp_composition_control

SEEDS = [42, 123, 456]
CFG = {
    "rank": 8,
    "residual_budget": 0.005,
    "residual_bits": 8,
    "surface": "mlp",
    "min_features": 256,
    "chunk_rows": 512,
}
SHAPES = [("bert_b32_s128", 32, 128), ("bert_b32_s512", 32, 512), ("bert_b64_s512", 64, 512)]
MODEL = "bert-base-uncased"


def build(kind: str):
    from transformers import AutoModelForSequenceClassification

    if kind == "qlora":
        from transformers import BitsAndBytesConfig

        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float32,
        )
        m = AutoModelForSequenceClassification.from_pretrained(
            MODEL, num_labels=2, quantization_config=bnb, device_map={"": 0}
        )
    else:
        m = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=2)
        m = m.to("cuda", dtype=torch.float32)
    if kind in ("lora", "qlora"):
        from peft import LoraConfig, get_peft_model

        m = get_peft_model(
            m, LoraConfig(r=8, lora_alpha=16, target_modules=["query", "value"], lora_dropout=0.0)
        )
    return m


def run(kind: str, use_stacr: bool, bs: int, seq: int, steps: int, seed: int, tag: str):
    free_all()
    set_seed(seed)
    model = build(kind)
    layers, compression = _apply_mlp_composition_control(
        model, CFG, enabled=use_stacr, adapter=kind in ("lora", "qlora")
    )
    opt = build_optimizer(model, 5e-05, True)
    batches = get_batches(MODEL, "train", bs, seq, steps, "wikitext", "cuda")
    for b in batches:
        b["labels"] = torch.randint(0, 2, (bs,), device="cuda")
    sync()
    t0 = None
    for i, b in enumerate(batches):
        if layers:
            set_step(layers, i)
        if i == min(6, steps - 1):
            sync()
            torch.cuda.reset_peak_memory_stats()
            t0 = time.perf_counter()
        opt.zero_grad(set_to_none=True)
        with amp_ctx(True):
            loss = model(**b).loss
        loss.backward()
        opt.step()
    sync()
    rec = {
        "tag": tag,
        "method": f"{kind}{('+stacr' if use_stacr else '')}",
        "model": MODEL,
        "bs": bs,
        "seq": seq,
        "steps": len(batches),
        "seed": seed,
        "quant": kind == "qlora",
        "peak_GiB": gib(torch.cuda.max_memory_allocated()),
        "peak_bytes": int(torch.cuda.max_memory_allocated()),
        "time_s": time.perf_counter() - t0,
        "final_loss": float(loss.detach()),
        **compression,
        "quality_metric_valid": False,
        "label_protocol": "random_binary",
        "adapter_rank": 8 if kind in ("lora", "qlora") else None,
    }
    del model, opt, layers, batches
    free_all()
    return rec


def main():
    import os

    out = RESULTS / os.environ.get("QLORA_OUT", "e5b_qlora.jsonl")
    print(f"[qlora] writing to {out.name}", flush=True)
    arms = [("full", False), ("lora", False), ("lora", True), ("qlora", False), ("qlora", True)]
    for tag, bs, seq in SHAPES:
        for kind, use_stacr in arms:
            for seed in SEEDS:
                try:
                    emit(run(kind, use_stacr, bs, seq, 40, seed, tag), out)
                except Exception as exc:
                    emit(
                        {
                            "tag": tag,
                            "method": f"{kind}{('+stacr' if use_stacr else '')}",
                            "seed": seed,
                            "status": "error",
                            "error": f"{type(exc).__name__}: {exc}"[:300],
                        },
                        out,
                    )
                    break
        print(f"  [{tag}] done", flush=True)


if __name__ == "__main__":
    main()
