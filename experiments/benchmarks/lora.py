"""Lora."""

from __future__ import annotations
import torch
import sys
from pathlib import Path
from .common import (
    RESULTS,
    emit,
    run_config,
    load_model,
    get_batches,
    build_optimizer,
    amp_ctx,
    free_all,
    set_seed,
    gib,
    sync,
)
from stacr.surgery import apply_stacr, select_target_names, set_step
import time, math

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
SHAPES = [
    ("bert_b32_s128", "bert-base-uncased", 32, 128),
    ("bert_b32_s512", "bert-base-uncased", 32, 512),
    ("bert_b64_s512", "bert-base-uncased", 64, 512),
    ("gpt2m_b8_s1024", "gpt2-medium", 8, 1024),
]


def _apply_mlp_composition_control(model, config, *, enabled, adapter):
    """Measure the requested trainable MLP surface, including an empty adapter control."""
    selection = {"surface": config["surface"], "min_features": config["min_features"]}
    eligible = select_target_names(model, **selection)
    total = select_target_names(model, **selection, trainable_only=False)
    resolved = {"turbo_after_step": None, **config}
    if enabled and adapter:
        # Attention-only adapters freeze the MLP. This negative-composition
        # control deliberately has no eligible MLP activation payload.
        resolved["allow_empty"] = True
    layers = apply_stacr(model, resolved) if enabled else []
    return layers, {
        "n_wrapped_layers": len(layers),
        "n_eligible_mlp_layers": len(eligible),
        "n_frozen_mlp_layers": len(total) - len(eligible),
        "eligible_mlp_layers": eligible,
        "empty_compression_control": bool(enabled and adapter and not eligible),
        "frozen_weight_compression": "bypassed",
        "stacr_cfg": resolved if enabled else None,
        "runtime_semantics": {
            "selection": "trainable_mlp_inputs",
            "frozen_mlp_payloads_stored": False,
            "timing_memory_protocol": "Frozen-layer payload allocation is bypassed; timing and memory differ from measurements that stored those payloads.",
        },
    }


def run_lora(model_id, bs, seq, steps, seed, use_lora, use_stacr, qlora=False, tag=""):
    if qlora:
        raise ValueError("Use the qlora experiment for NF4 quantized models")
    from peft import LoraConfig, get_peft_model

    free_all()
    set_seed(seed)
    dtype = torch.float32
    model = load_model(model_id, "cuda", dtype=dtype)
    if use_lora:
        targets = ["query", "value"] if "bert" in model_id else ["c_attn"]
        model = get_peft_model(
            model, LoraConfig(r=8, lora_alpha=16, target_modules=targets, lora_dropout=0.0)
        )
    layers, compression = _apply_mlp_composition_control(
        model, CFG, enabled=use_stacr, adapter=use_lora
    )
    opt = build_optimizer(model, 5e-05, True)
    is_bert = "bert" in model_id
    batches = get_batches(
        "bert-base-uncased" if is_bert else model_id, "train", bs, seq, steps, "wikitext", "cuda"
    )
    if is_bert:
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
        "method": f"{('lora' if use_lora else 'full')}{('+stacr' if use_stacr else '')}",
        "model": model_id,
        "bs": bs,
        "seq": seq,
        "steps": len(batches),
        "seed": seed,
        "peak_GiB": gib(torch.cuda.max_memory_allocated()),
        "peak_bytes": int(torch.cuda.max_memory_allocated()),
        "time_s": time.perf_counter() - t0,
        "final_loss": float(loss.detach()),
        **compression,
        "quality_metric_valid": not is_bert,
        "label_protocol": "random_binary" if is_bert else "causal_lm",
        "adapter_rank": 8 if use_lora else None,
    }
    del model, opt, layers, batches
    free_all()
    return rec


def main():
    out = RESULTS / "e5_lora.jsonl"
    for tag, mid, bs, seq in SHAPES:
        for use_lora, use_stacr in [(False, False), (True, False), (True, True), (False, True)]:
            for seed in SEEDS:
                try:
                    r = run_lora(mid, bs, seq, 40, seed, use_lora, use_stacr, tag=tag)
                except Exception as exc:
                    emit(
                        {
                            "tag": tag,
                            "method": f"lora={use_lora},stacr={use_stacr}",
                            "seed": seed,
                            "status": "error",
                            "error": f"{type(exc).__name__}: {exc}"[:300],
                        },
                        out,
                    )
                    continue
                emit(r, out)


if __name__ == "__main__":
    main()
