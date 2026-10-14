# Adapted from INSTANT, copyright (c) 2024 Trung-Hieu Tran. MIT; see LICENSE.txt.
# Upstream revision: 8a04bd43c19cc52984a49fb1b7fd1dae14fb2ffc.
import os
import argparse
import json
import time
import warnings


def parse_args():
    parser = argparse.ArgumentParser(description="Train a model on a specified GLUE dataset")
    parser.add_argument(
        "--dataset",
        type=str,
        default="mrpc",
        choices=["cola", "sst2", "mrpc", "qqp", "mnli", "qnli", "rte", "wnli"],
        help="GLUE dataset to train on (e.g., mrpc, cola, sst2)",
    )
    parser.add_argument(
        "--n-last-layers", type=int, default=1, help="Number of last layers to finetune"
    )
    parser.add_argument("--num_split", type=int, default=1, help="Number of mini batch per batch")
    parser.add_argument(
        "--calib_batches", type=int, default=5, help="Number of calibration batches"
    )
    parser.add_argument(
        "--calib_iter", type=int, default=50, help="Number of iteration per calibration"
    )
    parser.add_argument(
        "--model_name", type=str, default="bert-base-uncased", help="Pretrained model name"
    )
    parser.add_argument("--output_dir", type=str, default="runs-main", help="output dir")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for training")
    parser.add_argument(
        "--learning-rate", type=float, default=2e-05, help="Learning rate for optimizer"
    )
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--max-length", type=int, default=512, help="Tokenizer max sequence length")
    parser.add_argument("--max-steps", type=int, default=-1, help="Optional training-step limit")
    parser.add_argument("--seed", type=int, default=42, help="Seed of model")
    parser.add_argument("--var", type=float, default=0.95, help="Energy threshold of SVD")
    parser.add_argument("--over_sampling", type=int, default=7, help="Number of over sampling rank")
    parser.add_argument("--compress", action="store_true", help="Compress training or not")
    return parser.parse_args()


def main():
    args = parse_args()
    import torch
    import torch.nn as nn
    from transformers import (
        TrainingArguments,
        Trainer,
        AutoTokenizer,
        AutoModelForSequenceClassification,
    )
    from datasets import load_dataset
    from .utils import set_train_seed
    from .model import (
        freeze_bert_layers,
        add_activate_to_module,
        add_compression_tensor_to_module,
        get_svd_layers,
    )
    from .dataset_prep import (
        dataset_fields,
        dataset_to_num_labels,
        dataset_best_metrics,
        compute_metrics_with_args,
    )
    from .callback import CustomLoggingCallback
    from .custom_op import register_filter

    started = time.perf_counter()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_train_seed(args.seed)
    train_status = args.n_last_layers
    if args.compress:
        value = int(args.var * 100)
        args.output_dir = f"{args.output_dir}/{args.dataset}/{args.model_name}_last_{train_status}_layers_compress_var_{value}_over_sampling_{args.over_sampling}_calib_batch_{args.calib_batches}_calib_iter_{args.calib_iter}"
    else:
        args.output_dir = f"{args.output_dir}/{args.dataset}/{args.model_name}_last_{train_status}_layers_uncompress"
    os.makedirs(args.output_dir, exist_ok=True)
    try:
        dataset = load_dataset("glue", args.dataset)
    except Exception as e:
        print(f"Failed to load dataset {args.dataset}: {e}")
        exit(1)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    def tokenize_function(examples):
        fields = dataset_fields[args.dataset]
        if len(fields) == 2:
            return tokenizer(
                examples[fields[0]],
                examples[fields[1]],
                padding="max_length",
                truncation=True,
                max_length=args.max_length,
            )
        elif len(fields) == 1:
            return tokenizer(
                examples[fields[0]],
                padding="max_length",
                truncation=True,
                max_length=args.max_length,
            )
        else:
            raise ValueError(f"Dataset {args.dataset} has an unexpected structure")

    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    cols = ["input_ids", "attention_mask", "label"]
    if "token_type_ids" in tokenized_dataset["train"].column_names:
        cols.insert(2, "token_type_ids")
    tokenized_dataset.set_format("torch", columns=cols)
    num_labels = dataset_to_num_labels[args.dataset]
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name, num_labels=num_labels
    ).to(device)
    freeze_bert_layers(model, args.n_last_layers, args.model_name)
    if args.compress:
        add_activate_to_module()
        add_compression_tensor_to_module()
        modules_compressed = get_svd_layers(model)
        model = register_filter(model, modules_compressed)
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
        metric_for_best_model=dataset_best_metrics.get(args.dataset, "accuracy"),
        run_name=f"bert_{args.dataset}_run",
        report_to="none",
        logging_strategy="steps",
        logging_steps=100,
        log_level="info",
    )
    for name, param in model.named_parameters():
        status = "❄️ Frozen" if not param.requires_grad else "🔥 Unfrozen"
        print(f"{name}: {status}")
    print(model)
    if args.dataset == "mnli":
        eval_split = "validation_matched"
    else:
        eval_split = "validation"
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset[eval_split],
        compute_metrics=compute_metrics_with_args(args=args),
        callbacks=[CustomLoggingCallback(args=args)],
    )
    trainer.train()
    metrics = trainer.evaluate()
    result = {
        "status": "success",
        "task": args.dataset,
        "model": args.model_name,
        "seed": args.seed,
        "config": vars(args),
        "results": dict(metrics),
        "upstream_commit": "8a04bd43c19cc52984a49fb1b7fd1dae14fb2ffc",
    }
    result["results"]["total_time_s"] = time.perf_counter() - started
    result["results"]["peak_allocated_bytes"] = (
        int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else None
    )
    result["results"]["peak_reserved_bytes"] = (
        int(torch.cuda.max_memory_reserved()) if torch.cuda.is_available() else None
    )
    with open(os.path.join(args.output_dir, "final.json"), "w", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        peak_alloc = torch.cuda.max_memory_allocated()
        peak_reserved = torch.cuda.max_memory_reserved()
        print(f"peak_allocated_bytes={peak_alloc}")
        print(f"peak_reserved_bytes={peak_reserved}")
        print(f"peak_allocated_GB={peak_alloc / 1024**3:.4f}")
        print(f"peak_reserved_GB={peak_reserved / 1024**3:.4f}")


if __name__ == "__main__":
    main()
