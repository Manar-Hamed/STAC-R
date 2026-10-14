"""Paired OpenWebText memory transfer with optional disjoint evaluation.

The protocol uses disabled dropout, AdamW with a constant learning rate,
and one shuffled pass over document-truncated, concatenated token blocks. Its
reported perplexity is the last training minibatch, not validation perplexity.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import time
from pathlib import Path


def _loaders(args, tokenizer, seed):
    import torch
    from datasets import load_dataset
    from torch.utils.data import DataLoader

    errors = []
    for dataset in dict.fromkeys([args.dataset, args.fallback_dataset]):
        if not dataset:
            continue
        try:
            raw = load_dataset(dataset, split=args.split)
            break
        except Exception as exc:
            errors.append(f"{dataset}: {type(exc).__name__}: {exc}")
    else:
        raise RuntimeError("Dataset loading failed: " + " | ".join(errors))
    if "text" not in raw.column_names:
        raise ValueError("OpenWebText data must contain a text column")
    tokenized = raw.map(
        lambda batch: tokenizer(batch["text"], truncation=True, max_length=args.max_length),
        batched=True,
        remove_columns=raw.column_names,
    )

    def group(examples):
        ids = sum(examples["input_ids"], [])
        count = len(ids) // args.max_length * args.max_length
        chunks = [ids[i : i + args.max_length] for i in range(0, count, args.max_length)]
        return {
            "input_ids": chunks,
            "attention_mask": [[1] * args.max_length for _ in chunks],
            "labels": [chunk[:] for chunk in chunks],
        }

    grouped = tokenized.map(group, batched=True)
    grouped.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    count = len(grouped)
    if args.validation_blocks >= count or count - args.validation_blocks < args.batch_size:
        raise ValueError(
            "The selected corpus does not contain a complete training batch after reserving validation blocks"
        )
    split = count - args.validation_blocks
    train = grouped.select(range(split)) if args.validation_blocks else grouped
    validation = grouped.select(range(split, count)) if args.validation_blocks else None
    generator = torch.Generator().manual_seed(seed) if args.data_order == "paired" else None
    loader = DataLoader(
        train,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0,
        generator=generator,
    )
    val_loader = (
        DataLoader(validation, batch_size=args.batch_size, shuffle=False, num_workers=0)
        if validation is not None
        else None
    )
    return (
        loader,
        val_loader,
        {
            "dataset_used": dataset,
            "training_blocks": split,
            "validation_blocks": args.validation_blocks,
        },
    )


def _perplexity(loss):
    return math.exp(loss) if loss is not None and math.isfinite(loss) and loss < 700 else None


def _run_one(args, method, seed):
    import torch
    from transformers import AutoTokenizer, GPT2LMHeadModel

    started = time.perf_counter()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; use --device cpu for a small functional check")
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    loader, val_loader, data = _loaders(args, tokenizer, seed)
    model = GPT2LMHeadModel.from_pretrained(args.model).to(device)
    model.eval()  # Disable dropout for this protocol.
    if method == "full":
        from .training import (
            ConvergenceConfig,
            get_compression_targets_gpt2,
            replace_linear_with_stacr,
        )

        cfg = ConvergenceConfig(
            task="wikitext103",
            model=args.model,
            method="stacr",
            rank=args.rank,
            batch_size=args.batch_size,
            max_length=args.max_length,
            seed=seed,
            quiet=True,
            stacr_warmup_steps=0,
            stacr_residual_mode="block_sparse",
            stacr_residual_budget=0.005,
            stacr_residual_quantize_bits=4,
            stacr_quantize_coefficients=True,
            stacr_quantize_residuals=True,
            stacr_basis_dtype=args.basis_dtype,
            stacr_freeze_basis_after_warmup=True,
        )
        replace_linear_with_stacr(model, cfg, get_compression_targets_gpt2(model))
        model.to(device)
        # Newly installed compression modules train; pretrained dropout stays off.
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scaler = torch.amp.GradScaler(device.type, enabled=args.amp == "fp16")
    dtype = torch.float16 if args.amp == "fp16" else torch.bfloat16
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    training_start = time.perf_counter()
    last_loss = None
    steps = 0
    for batch in loader:
        if steps >= args.steps:
            break
        optimizer.zero_grad(set_to_none=True)
        inputs = {key: value.to(device) for key, value in batch.items()}
        with torch.autocast(device_type=device.type, dtype=dtype, enabled=args.amp != "off"):
            loss = model(**inputs).loss
        if not torch.isfinite(loss):
            raise FloatingPointError("Training loss is non-finite")
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        last_loss = float(loss.detach())
        steps += 1
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    training_time = time.perf_counter() - training_start
    result = {
        "method": method,
        "seed": seed,
        "status": "success" if steps == args.steps else "incomplete",
        "steps_requested": args.steps,
        "steps_ran": steps,
        "last_train_minibatch_loss": last_loss,
        "last_train_minibatch_ppl": _perplexity(last_loss),
        "training_time_s": training_time,
        "total_time_s": time.perf_counter() - started,
        "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30
        if device.type == "cuda"
        else None,
        "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 2**30
        if device.type == "cuda"
        else None,
        "hardware": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        **data,
    }
    if val_loader is not None:
        model.eval()
        loss_sum, tokens = 0.0, 0
        with torch.no_grad():
            for batch in val_loader:
                inputs = {key: value.to(device) for key, value in batch.items()}
                with torch.autocast(
                    device_type=device.type, dtype=dtype, enabled=args.amp != "off"
                ):
                    loss = model(**inputs).loss
                count = inputs["labels"][:, 1:].numel()
                loss_sum += float(loss) * count
                tokens += count
        val_loss = loss_sum / tokens
        result.update(
            heldout_loss=val_loss, heldout_ppl=_perplexity(val_loss), heldout_tokens=tokens
        )
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="Skylion007/openwebtext")
    parser.add_argument("--fallback-dataset", default="stas/openwebtext-10k")
    parser.add_argument("--split", default="train[:8192]")
    parser.add_argument("--tokenizer", default="gpt2")
    parser.add_argument("--model", default="gpt2-medium")
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--steps", "--max-steps", type=int, default=500)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456])
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--amp", choices=["off", "fp16", "bf16"], default="bf16")
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--basis-dtype", choices=["float16", "float32", "auto"], default="float16")
    parser.add_argument(
        "--methods", choices=["baseline", "full"], nargs="+", default=["baseline", "full"]
    )
    parser.add_argument(
        "--data-order",
        choices=["paired", "global-rng"],
        default="paired",
        help="paired gives both methods identical minibatches; global-rng uses the shared PyTorch RNG",
    )
    parser.add_argument(
        "--validation-blocks",
        type=int,
        default=0,
        help="Reserve disjoint final token blocks; zero reports the last training minibatch metric",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, default=Path("outputs/openwebtext_transfer.json"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if (
        min(args.max_length, args.batch_size, args.steps, args.rank) < 1
        or args.validation_blocks < 0
    ):
        parser.error(
            "Shapes, steps and rank must be positive; validation-blocks cannot be negative"
        )
    config = {
        key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()
    }
    config.update(
        dropout="disabled",
        optimizer="AdamW",
        scheduler="constant",
        quality_metric="last_training_minibatch",
        validation="disjoint_final_blocks" if args.validation_blocks else "none",
    )
    if args.dry_run:
        print(json.dumps(config, indent=2))
        return 0
    rows = []
    for seed in args.seeds:
        for method in args.methods:
            try:
                row = _run_one(args, method, seed)
            except Exception as exc:
                row = {
                    "method": method,
                    "seed": seed,
                    "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            rows.append(row)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(
                json.dumps({"config": config, "rows": rows}, indent=2, allow_nan=False) + "\n",
                encoding="utf-8",
            )
            print(json.dumps(row, allow_nan=False), flush=True)
            gc.collect()
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    return int(any(row["status"] != "success" for row in rows))


if __name__ == "__main__":
    raise SystemExit(main())
