# Adapted from INSTANT, copyright (c) 2024 Trung-Hieu Tran. MIT; see LICENSE.txt.
# Upstream revision: 8a04bd43c19cc52984a49fb1b7fd1dae14fb2ffc.
"""
INSTANT compression on GPT-2 / WikiText-103 (Causal Language Modeling).

Port of the original INSTANT BERT classification pipeline to GPT-2 CLM.
Key differences from train.py:
  - AutoModelForCausalLM instead of AutoModelForSequenceClassification
  - Conv1D -> nn.Linear conversion (GPT-2 uses Conv1D, INSTANT needs nn.Linear)
  - WikiText-103 with contiguous chunking (no padding waste)
  - Perplexity evaluation instead of accuracy
  - labels = input_ids (model computes shifted CE internally)

Usage:
  # Baseline (no compression)
  python train_gpt2.py --model_name gpt2 --max-length 512 --epochs 3 --seed 42

  # INSTANT compression
  python train_gpt2.py --model_name gpt2 --max-length 512 --epochs 3 --seed 42       --compress --over_sampling 15 --var 0.95

  # GPT-2 Large, full fine-tune
  python train_gpt2.py --model_name gpt2-large --n-last-layers 0 --max-length 512       --batch-size 4 --epochs 1 --max-steps 5000 --compress --over_sampling 15
"""

import os
import sys
import json
import math
import time
import argparse
import warnings


def parse_args():
    parser = argparse.ArgumentParser(description="INSTANT on GPT-2 / WikiText-103")
    parser.add_argument(
        "--model_name",
        type=str,
        default="gpt2",
        choices=["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"],
        help="GPT-2 model variant",
    )
    parser.add_argument(
        "--n-last-layers",
        type=int,
        default=0,
        help="Number of last layers to fine-tune (0 = full fine-tune)",
    )
    parser.add_argument(
        "--num_split", type=int, default=1, help="Number of mini-batch splits per calibration batch"
    )
    parser.add_argument(
        "--calib_batches", type=int, default=5, help="Calibration batches per recalibration"
    )
    parser.add_argument("--calib_iter", type=int, default=50, help="Steps between recalibrations")
    parser.add_argument("--output_dir", type=str, default="runs-gpt2", help="Output directory root")
    parser.add_argument("--batch-size", type=int, default=4, help="Per-device batch size")
    parser.add_argument("--learning-rate", type=float, default=2e-05, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=1, help="Training epochs")
    parser.add_argument("--max-length", type=int, default=512, help="Sequence length for chunking")
    parser.add_argument(
        "--max-steps", type=int, default=-1, help="Max training steps (-1 = use epochs)"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--var", type=float, default=0.95, help="SVD energy threshold")
    parser.add_argument("--over_sampling", type=int, default=15, help="SVD oversampling rank")
    parser.add_argument("--compress", action="store_true", help="Enable INSTANT compression")
    parser.add_argument("--dataset", type=str, default="wikitext103")
    return parser.parse_args()


def main():
    args = parse_args()
    import torch
    import torch.nn as nn
    from transformers import (
        TrainingArguments,
        Trainer,
        AutoTokenizer,
        AutoModelForCausalLM,
        DataCollatorForLanguageModeling,
    )
    from .utils import set_train_seed
    from .model import (
        freeze_layers,
        add_activate_to_module,
        add_compression_tensor_to_module,
        get_svd_layers,
        convert_conv1d_to_linear,
        print_status,
    )
    from .dataset_prep import load_wikitext103
    from .callback import CustomLoggingCallback
    from .custom_op import register_filter

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_train_seed(args.seed)
    start_time = time.time()
    train_status = args.n_last_layers
    if args.compress:
        value = int(args.var * 100)
        step_suffix = f"_{args.max_steps}steps" if args.max_steps > 0 else ""
        args.output_dir = f"{args.output_dir}/{args.model_name}_last_{train_status}_layers_compress_var_{value}_over_sampling_{args.over_sampling}_calib_batch_{args.calib_batches}_calib_iter_{args.calib_iter}_L{args.max_length}_s{args.seed}{step_suffix}"
    else:
        step_suffix = f"_{args.max_steps}steps" if args.max_steps > 0 else ""
        args.output_dir = f"{args.output_dir}/{args.model_name}_last_{train_status}_layers_uncompress_L{args.max_length}_s{args.seed}{step_suffix}"
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Output: {args.output_dir}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model_name)
    model.config.use_cache = False
    model = convert_conv1d_to_linear(model)
    model = model.to(device)
    freeze_layers(model, args.n_last_layers, args.model_name)
    if args.compress:
        add_activate_to_module()
        add_compression_tensor_to_module()
        modules_compressed = get_svd_layers(model, model_name=args.model_name)
        print(f"[INSTANT] Compressing {len(modules_compressed)} layers:")
        for m in modules_compressed[:5]:
            print(f"  {m}")
        if len(modules_compressed) > 5:
            print(f"  ... and {len(modules_compressed) - 5} more")
        model = register_filter(model, modules_compressed)
        model = model.to(device)
    print("Loading WikiText-103...")
    (train_dataset, val_dataset) = load_wikitext103(tokenizer, max_length=args.max_length)
    print(f"Train: {len(train_dataset)} chunks, Val: {len(val_dataset)} chunks")
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        weight_decay=0.01,
        warmup_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        greater_is_better=False,
        run_name=f"instant_gpt2_{args.model_name}",
        report_to="none",
        logging_strategy="steps",
        logging_steps=50,
        log_level="info",
        fp16=False,
    )
    print_status(model)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        callbacks=[CustomLoggingCallback(args=args)] if args.compress else [],
    )
    trainer.train()
    eval_results = trainer.evaluate()
    eval_loss = eval_results.get("eval_loss", float("nan"))
    final_ppl = math.exp(min(eval_loss, 100))
    wall_time = time.time() - start_time
    print(f"\n=== RESULTS ===")
    print(f"Model: {args.model_name}")
    print(f"Compress: {args.compress}")
    if args.compress:
        print(f"  var={args.var}, over_sampling={args.over_sampling}")
    print(f"Seq length: {args.max_length}")
    print(f"Final eval loss: {eval_loss:.4f}")
    print(f"Final perplexity: {final_ppl:.2f}")
    print(f"Wall time: {wall_time:.1f}s")
    peak_alloc_gb = 0.0
    peak_reserved_gb = 0.0
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        peak_alloc = torch.cuda.max_memory_allocated()
        peak_reserved = torch.cuda.max_memory_reserved()
        peak_alloc_gb = peak_alloc / 1024**3
        peak_reserved_gb = peak_reserved / 1024**3
        print(f"peak_allocated_GB={peak_alloc_gb:.4f}")
        print(f"peak_reserved_GB={peak_reserved_gb:.4f}")
    peak_alloc_bytes = int(peak_alloc_gb * 1024**3) if peak_alloc_gb else 0
    peak_reserved_bytes = int(peak_reserved_gb * 1024**3) if peak_reserved_gb else 0
    method_label = f"instant_ov{args.over_sampling}" if args.compress else "instant_base"
    config_payload = {
        "task": "wikitext103",
        "model": args.model_name,
        "method": method_label,
        "seed": args.seed,
        "batch_size": args.batch_size,
        "max_length": args.max_length,
        "max_steps": args.max_steps,
        "lr": args.learning_rate,
        "epochs": args.epochs,
        "compress": args.compress,
        "var": args.var if args.compress else None,
        "over_sampling": args.over_sampling if args.compress else None,
        "n_last_layers": args.n_last_layers,
    }
    config_path = os.path.join(args.output_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(config_payload, f, indent=2)
    final = {
        "status": "success",
        "task": "wikitext103",
        "model": args.model_name,
        "method": method_label,
        "seed": args.seed,
        "run_id": os.path.basename(args.output_dir),
        "results": {
            "best_perplexity": final_ppl,
            "final_perplexity": final_ppl,
            "best_accuracy": 0.0,
            "eval_loss": eval_loss,
            "total_time_s": wall_time,
            "peak_allocated_bytes": peak_alloc_bytes,
            "peak_reserved_bytes": peak_reserved_bytes,
        },
        "baseline_cfg": {},
    }
    final_path = os.path.join(args.output_dir, "final.json")
    with open(final_path, "w") as f:
        json.dump(final, f, indent=2)
    print(f"Saved: {final_path}")


if __name__ == "__main__":
    main()
