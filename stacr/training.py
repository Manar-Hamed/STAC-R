"""Deterministic training and same-batch gradient comparisons.

Specialized measurement grids live in ``experiments.benchmarks``. This runner is
also a small reference integration for a user's own PyTorch training pipeline.
"""

from __future__ import annotations

import copy
import importlib
import math
import time
from pathlib import Path
from typing import Any

import torch

from .surgery import apply_stacr, set_step
from .utils import device_from_config, set_seed, write_csv

STACR_METHODS = {
    "stacr",
    "turbofast",
    "stacr_galore",
    "turbofast_galore",
    "subtrack_stacr",
    "stacr_checkpointing",
    "turbofast_checkpointing",
}
METHODS = STACR_METHODS | {"baseline", "checkpointing", "galore", "subtrack", "freeze"}
CORE_KEYS = (
    "rank",
    "residual_budget",
    "residual_bits",
    "oja_lr",
    "update_interval",
    "warmup_steps",
    "freeze_after_warmup",
    "turbo_after_step",
    "chunk_rows",
    "basis_dtype",
    "residual_mode",
    "residual_block_size",
    "residual_sample_ratio",
    "surface",
    "min_features",
    "extra_names",
    "target_names",
    "target_pattern",
    "exclude_names",
    "trainable_only",
    "allow_empty",
)
RUN_KEYS = {
    "task",
    "method",
    "engine",
    "compression",
    "model_name",
    "dataset",
    "device",
    "seed",
    "steps",
    "batch_size",
    "seq_len",
    "max_examples",
    "max_eval_examples",
    "fidelity_steps",
    "features",
    "hidden",
    "lr",
    "weight_decay",
    "optimizer",
    "use_8bit_optimizer",
    "gradient_accumulation_steps",
    "measure_from",
    "amp",
    "model_dtype",
    "lr_warmup_steps",
    "lr_schedule",
    "max_grad_norm",
    "checkpoint_path",
    "checkpoint_every",
    "checkpoint_offset",
    "offload_every",
    "offload_offset",
    "compression_blocks",
    "attention_surface",
    "freeze_basis_after_warmup",
    "model_revision",
    "dataset_revision",
    "attn_implementation",
    "local_files_only",
    "train_file",
    "validation_file",
    "galore_rank",
    "galore_update_proj_gap",
    "galore_scale",
    "galore_proj_type",
    "galore_target_modules",
    "subtrack_rank",
    "subtrack_scale",
    "subtrack_update_interval",
    "subtrack_step_size",
    "subtrack_target_modules",
    "subtrack_proj_type",
    "subtrack_adaptive_optimizer",
    "subtrack_recovery_scaling",
    "subtrack_norm_growth_limit",
}


def validate_config(config: dict) -> dict:
    cfg = copy.deepcopy(config)
    unknown = cfg.keys() - RUN_KEYS - set(CORE_KEYS)
    if unknown:
        raise ValueError(f"Unknown training configuration fields: {sorted(unknown)}")
    cfg.setdefault("task", "synthetic")
    cfg.setdefault("method", "turbofast" if cfg["task"] == "synthetic" else "baseline")
    cfg.setdefault("steps", 5 if cfg["task"] == "synthetic" else 500)
    cfg.setdefault("batch_size", 8 if cfg["task"] == "synthetic" else 2)
    cfg.setdefault("seed", 42)
    cfg.setdefault("gradient_accumulation_steps", 1)
    cfg.setdefault("measure_from", 0)
    cfg.setdefault("engine", "projection")
    if cfg["engine"] not in ("projection", "adaptive"):
        raise ValueError("engine must be projection or adaptive")
    if cfg["method"] not in METHODS:
        raise ValueError(f"Unknown method {cfg['method']!r}; choose from {sorted(METHODS)}")
    if cfg["task"] not in ("synthetic", "language_modeling", "sequence_classification"):
        raise ValueError(f"Unknown task {cfg['task']!r}")
    for name in ("steps", "batch_size", "gradient_accumulation_steps"):
        if int(cfg[name]) < 1:
            raise ValueError(f"{name} must be positive")
    if not 0 <= int(cfg["measure_from"]) < int(cfg["steps"]):
        raise ValueError("measure_from must be in [0, steps)")
    if cfg["task"] != "synthetic" and not cfg.get("model_name"):
        raise ValueError("model_name is required for Hugging Face training")
    if cfg["method"].startswith("turbofast"):
        cfg.setdefault("turbo_after_step", 50)
    elif cfg["method"] in STACR_METHODS:
        cfg.setdefault("turbo_after_step", None)
    if "galore" in cfg["method"]:
        cfg.setdefault("optimizer", "galore")
        if cfg["optimizer"] != "galore":
            raise ValueError("A GaLore method requires optimizer=galore")
    if "subtrack" in cfg["method"]:
        cfg.setdefault("optimizer", "subtrack")
        if cfg["optimizer"] != "subtrack":
            raise ValueError("A SubTrack method requires optimizer=subtrack")
    if "checkpointing" in cfg["method"]:
        cfg.setdefault("checkpoint_every", 2 if cfg["method"] in STACR_METHODS else 1)
    return cfg


def _stacr_config(config: dict) -> dict:
    cfg = {k: config[k] for k in CORE_KEYS if k in config}
    if config.get("engine", "projection") == "projection":
        cfg.update(config.get("compression", {}))
    if config.get("attention_surface"):
        cfg["surface"] = "mlp+attention"
    if "freeze_basis_after_warmup" in config:
        cfg["freeze_after_warmup"] = config["freeze_basis_after_warmup"]
    return cfg


def configure_model(model, config: dict):
    """Apply compositions in place before constructing an optimizer or DDP wrapper."""
    from .composition import (
        block_target_names,
        checkpoint_blocks,
        offload_blocks,
        transformer_blocks,
    )
    from .surgery import select_target_names

    cfg = validate_config(config)
    core = _stacr_config(cfg)
    selection_keys = {
        "surface",
        "min_features",
        "extra_names",
        "target_names",
        "target_pattern",
        "exclude_names",
        "trainable_only",
        "allow_empty",
    }
    if cfg["task"] == "synthetic":
        core.setdefault("surface", "all")
        core.setdefault("min_features", 1)
    if cfg["engine"] == "adaptive":
        adaptive = {k: v for k, v in core.items() if k in selection_keys}
        adaptive.update(
            rank=int(cfg.get("rank", 8)), residual_budget=float(cfg.get("residual_budget", 0.005))
        )
        for key in (
            "basis_dtype",
            "update_interval",
            "warmup_steps",
            "residual_block_size",
            "residual_mode",
        ):
            if key in cfg:
                adaptive[key] = cfg[key]
        for source, destination in (
            ("oja_lr", "oja_learning_rate"),
            ("chunk_rows", "chunk_size"),
            ("freeze_after_warmup", "freeze_basis_after_warmup"),
        ):
            if source in core:
                adaptive[destination] = core[source]
        if "residual_bits" in cfg:
            bits = cfg["residual_bits"]
            if bits not in (0, 4, 8):
                raise ValueError("residual_bits must be 0, 4, or 8")
            adaptive["quantize_residuals"] = bits != 0
            if bits:
                adaptive["residual_quantize_bits"] = bits
        if "residual_sample_ratio" in cfg:
            raise ValueError(
                "The adaptive engine uses compression.streaming_topk_sample_ratio for threshold sampling"
            )
        if cfg["method"].startswith("turbofast"):
            adaptive.update(
                post_warmup_residual_mode="none",
                residual_warmup_steps=int(cfg["turbo_after_step"]),
                refresh_trigger="residual_energy",
                cache_basis_per_optimizer_step=True,
                oja_update_per_optimizer_step=True,
            )
        adaptive.update(cfg.get("compression", {}))
        core = adaptive
    compression_blocks = cfg.get("compression_blocks", "complement")
    if compression_blocks not in ("complement", "all"):
        raise ValueError("compression_blocks must be 'complement' or 'all'")
    ckpt = int(cfg.get("checkpoint_every", 0))
    offload = int(cfg.get("offload_every", 0))
    if ckpt < 0 or offload < 0:
        raise ValueError("checkpoint_every and offload_every must be nonnegative")
    checkpointed = []
    offloaded = []
    if ckpt or offload:
        blocks = transformer_blocks(model)
        offset = int(cfg.get("checkpoint_offset", 0))
        if ckpt:
            if not 0 <= offset < ckpt:
                raise ValueError("checkpoint_offset must be in [0, checkpoint_every)")
            checkpointed = [i for i in range(len(blocks)) if i % ckpt == offset]
            if cfg["method"] in STACR_METHODS and compression_blocks == "complement":
                selection = {
                    k: v for k, v in core.items() if k in selection_keys and k != "allow_empty"
                }
                names = block_target_names(
                    model, blocks, lambda i: i not in checkpointed, **selection
                )
                if not names:
                    raise ValueError(
                        "No uncheckpointed STAC-R surface remains; use selective checkpointing"
                    )
                for key in selection_keys:
                    core.pop(key, None)
                core["surface"], core["target_names"] = "none", names
        if offload:
            offset = int(cfg.get("offload_offset", 0))
            if not 0 <= offset < offload:
                raise ValueError("offload_offset must be in [0, offload_every)")
            offloaded = [i for i in range(len(blocks)) if i % offload == offset]
    if cfg["method"] == "freeze":
        names = select_target_names(
            model, **{k: v for k, v in core.items() if k in selection_keys and k != "allow_empty"}
        )
        if not names:
            raise ValueError("No modules selected for the freeze control")
        for name in names:
            model.get_submodule(name).requires_grad_(False)
    if cfg["method"] in STACR_METHODS and cfg["engine"] == "adaptive":
        from .adaptive import apply_adaptive_stacr

        layers = apply_adaptive_stacr(model, core)
    else:
        layers = apply_stacr(model, core) if cfg["method"] in STACR_METHODS else []
    if checkpointed:
        checkpoint_blocks(blocks, checkpointed)
    if offloaded:
        offload_blocks(blocks, offloaded)
    return layers, {
        "checkpointed_blocks": checkpointed,
        "offloaded_blocks": offloaded,
        "wrapped_names": [layer.source_name for layer in layers],
    }


def _build_optimizer(model, config):
    lr = float(config.get("lr", 5e-5))
    decay = float(config.get("weight_decay", 0.0))
    name = str(
        config.get("optimizer", "adamw8bit" if config.get("use_8bit_optimizer") else "adamw")
    ).lower()
    parameters = [p for p in model.parameters() if p.requires_grad]
    if not parameters:
        raise ValueError("The model has no trainable parameters")
    if name == "adamw":
        return torch.optim.AdamW(parameters, lr=lr, weight_decay=decay), "AdamW"
    if name == "adamw8bit":
        try:
            import bitsandbytes as bnb
        except ImportError as exc:
            raise RuntimeError("8-bit AdamW requires bitsandbytes; install stacr[peft]") from exc
        if not all(p.is_cuda for p in parameters):
            raise ValueError("The 8-bit optimizer experiment requires CUDA parameters")
        return bnb.optim.AdamW8bit(parameters, lr=lr, weight_decay=decay), "AdamW8bit"
    if name == "galore":
        from comparators.galore_adapter import create_galore_optimizer

        return create_galore_optimizer(
            model,
            lr=lr,
            rank=int(config.get("galore_rank", 128)),
            update_proj_gap=int(config.get("galore_update_proj_gap", 200)),
            scale=float(config.get("galore_scale", 0.25)),
            proj_type=config.get("galore_proj_type", "std"),
            weight_decay=decay,
            target_modules=config.get("galore_target_modules"),
        ), "GaLoreAdamW"
    if name == "subtrack":
        from comparators.subtrack_adapter import create_subtrack_optimizer

        return create_subtrack_optimizer(model, config), "SubTrack++"
    if ":" in name:
        # A user-supplied optimizer factory receives the model and resolved config.
        module, attribute = str(config["optimizer"]).split(":", 1)
        optimizer = getattr(importlib.import_module(module), attribute)(model, config)
        if not isinstance(optimizer, torch.optim.Optimizer):
            raise TypeError("Optimizer factory must return torch.optim.Optimizer")
        return optimizer, str(config["optimizer"])
    raise ValueError(f"Unknown optimizer {name!r}")


def _amp_dtype(config, device):
    mode = str(config.get("amp", "none")).lower()
    if mode in ("none", "off", "false"):
        return None
    if mode in ("bf16", "bfloat16"):
        if device.type == "cuda" and not torch.cuda.is_bf16_supported():
            raise ValueError("bf16 was requested but this CUDA device does not support it")
        return torch.bfloat16
    if mode in ("fp16", "float16"):
        if device.type != "cuda":
            raise ValueError("fp16 training requires CUDA; use bf16 or none on CPU")
        return torch.float16
    raise ValueError("amp must be none, bf16 or fp16")


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.no_grad()
def _evaluate(model, loader, device, task, amp_dtype):
    was_training = model.training
    model.eval()
    total_loss = total_tokens = correct = seen = tp = fp = fn = 0
    try:
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.autocast(
                device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None
            ):
                out = model(**batch)
            if task == "sequence_classification":
                pred, labels = out.logits.argmax(-1), batch["labels"]
                correct += int((pred == labels).sum())
                seen += labels.numel()
                tp += int(((pred == 1) & (labels == 1)).sum())
                fp += int(((pred == 1) & (labels != 1)).sum())
                fn += int(((pred != 1) & (labels == 1)).sum())
            else:
                ntok = int((batch["labels"][..., 1:] != -100).sum())
                total_loss += float(out.loss) * ntok
                total_tokens += ntok
    finally:
        model.train(was_training)
    if task == "sequence_classification":
        if not seen:
            raise ValueError("Validation dataset is empty")
        return {
            "accuracy": 100 * correct / seen,
            "f1": 100 * 2 * tp / max(1, 2 * tp + fp + fn),
            "ppl": None,
        }
    if not total_tokens:
        raise ValueError("Validation dataset has no nonmasked causal targets")
    nll = total_loss / total_tokens
    return {"accuracy": None, "ppl": math.exp(nll) if nll < 709 else float("inf"), "nll": nll}


def train_model(model, train_batches, config: dict, *, validation_batches=None) -> dict:
    """Train any model returning ``.loss`` on dictionary batches.

    CPU batches are moved one at a time, so dataset length does not inflate GPU
    memory. ``steps`` counts optimizer updates; accumulation divides each loss.
    """
    cfg = validate_config(config)
    device = device_from_config(cfg.get("device", "auto"))
    if not len(train_batches):
        raise ValueError("Training dataset is empty")
    model.to(device).train()
    layers, composition = configure_model(model, cfg)
    optimizer, optimizer_name = _build_optimizer(model, cfg)
    dtype = _amp_dtype(cfg, device)
    scaler = torch.amp.GradScaler(device.type, enabled=dtype is torch.float16)
    steps = int(cfg["steps"])
    accum = int(cfg["gradient_accumulation_steps"])
    measure_from = int(cfg["measure_from"])
    warmup = int(cfg.get("lr_warmup_steps", 0))
    if not 0 <= warmup <= steps:
        raise ValueError("lr_warmup_steps must be between zero and steps")
    if cfg.get("lr_schedule", "constant") not in ("constant", "linear", "cosine"):
        raise ValueError("lr_schedule must be constant, linear or cosine")
    initial_lr = [g["lr"] for g in optimizer.param_groups]
    losses = []
    training_tokens = 0
    measured_updates = 0
    start = None
    peak = reserved = None
    for step in range(steps):
        if step == measure_from:
            _sync(device)
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            start = time.perf_counter()
        fraction = min(1.0, (step + 1) / max(1, warmup))
        if step >= warmup and cfg.get("lr_schedule", "constant") != "constant":
            progress = (step - warmup) / max(1, steps - warmup - 1)
            fraction = (
                1 - progress
                if cfg["lr_schedule"] == "linear"
                else (1 + math.cos(math.pi * progress)) / 2
            )
        for group, lr in zip(optimizer.param_groups, initial_lr):
            group["lr"] = lr * fraction
        optimizer.zero_grad(set_to_none=True)
        loss_sum = 0.0
        for micro in range(accum):
            set_step(layers, step, microbatch_index=micro, accum_steps=accum)
            batch = train_batches[(step * accum + micro) % len(train_batches)]
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.autocast(device_type=device.type, dtype=dtype, enabled=dtype is not None):
                loss = model(**batch).loss
            if not torch.isfinite(loss).all():
                raise FloatingPointError(f"Nonfinite loss at optimizer step {step}")
            scaler.scale(loss / accum).backward()
            loss_sum += float(loss.detach()) / accum
            if step >= measure_from and "input_ids" in batch:
                training_tokens += int(
                    batch.get("attention_mask", torch.ones_like(batch["input_ids"])).sum()
                )
        if cfg.get("max_grad_norm") is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["max_grad_norm"]))
        scaler.step(optimizer)
        scaler.update()
        losses.append(loss_sum)
        if step >= measure_from:
            measured_updates += 1
    _sync(device)
    elapsed = time.perf_counter() - start
    if device.type == "cuda":
        peak = torch.cuda.max_memory_allocated(device) / 1024**3
        reserved = torch.cuda.max_memory_reserved(device) / 1024**3
    result = {
        "task": cfg["task"],
        "model": cfg.get("model_name", "synthetic"),
        "method": cfg["method"],
        "engine": cfg["engine"],
        "optimizer": optimizer_name,
        "steps": steps,
        "steps_measured": measured_updates,
        "batch_size": int(cfg["batch_size"]),
        "gradient_accumulation_steps": accum,
        "seq_len": cfg.get("seq_len"),
        "seed": int(cfg["seed"]),
        "wrapped_layers": len(layers),
        "train_loss": losses[-1],
        "peak_gib": peak,
        "reserved_gib": reserved,
        "seconds": elapsed,
        "tokens_per_second": training_tokens / elapsed if training_tokens else None,
        "amp": cfg.get("amp", "none"),
        "device": str(device),
        "torch_version": torch.__version__,
        **composition,
    }
    if validation_batches is not None:
        quality = _evaluate(model, validation_batches, device, cfg["task"], dtype)
        result.update({f"val_{k}": v for k, v in quality.items()})
    if cfg.get("checkpoint_path"):
        checkpoint = Path(cfg["checkpoint_path"])
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict(),
                "step": steps,
                "config": cfg,
                "rng_state": torch.get_rng_state(),
                "cuda_rng_state": torch.cuda.get_rng_state_all() if device.type == "cuda" else [],
            },
            checkpoint,
        )
    return result


def run_synthetic(config):
    from types import SimpleNamespace

    cfg = validate_config(config)
    set_seed(cfg["seed"])
    features, hidden = int(cfg.get("features", 64)), int(cfg.get("hidden", 32))

    class Autoencoder(torch.nn.Sequential):
        def forward(self, inputs, labels):
            logits = super().forward(inputs)
            return SimpleNamespace(loss=(logits - labels).square().mean(), logits=logits)

    model = Autoencoder(
        torch.nn.Linear(features, hidden), torch.nn.GELU(), torch.nn.Linear(hidden, features)
    )
    generator = torch.Generator().manual_seed(int(cfg["seed"]) + 1)
    batches = []
    for _ in range(int(cfg["steps"]) * int(cfg["gradient_accumulation_steps"])):
        x = torch.randn(int(cfg["batch_size"]), features, generator=generator)
        batches.append({"inputs": x, "labels": x.clone()})
    return train_model(model, batches, cfg)


def _hf_inputs(config):
    from torch.utils.data import DataLoader
    from .datasets import build_examples
    from .models import load_model_and_tokenizer

    dtype_name = str(config.get("model_dtype", "float32"))
    if dtype_name not in ("float32", "float16", "bfloat16"):
        raise ValueError("model_dtype must be float32, float16 or bfloat16")
    model, tokenizer = load_model_and_tokenizer(
        config["model_name"],
        config["task"],
        revision=config.get("model_revision"),
        attn_implementation=config.get("attn_implementation"),
        dtype=getattr(torch, dtype_name),
        local_files_only=bool(config.get("local_files_only", False)),
    )
    train = build_examples(config, tokenizer)
    validation = build_examples(config, tokenizer, validation=True)
    # Materialize CPU batches once; repeated epochs retain the same ordered protocol.
    bs = int(config["batch_size"])
    return (
        model,
        list(DataLoader(train, batch_size=bs)),
        list(DataLoader(validation, batch_size=bs)),
    )


def run_hf(config):
    cfg = validate_config(config)
    set_seed(cfg["seed"])
    model, train, validation = _hf_inputs(cfg)
    return train_model(model, train, cfg, validation_batches=validation)


def gradient_fidelity(config):
    """Compare same-state gradients with dropout off and paired RNG.

    A forward with dropout disabled fits/refreshes bases before each measurement. The two
    measured passes do not mutate compression state or update model parameters.
    """
    from .diagnostics import compression_diagnostics

    cfg = validate_config(config)
    set_seed(cfg["seed"])
    reference, batches, _ = _hf_inputs(cfg)
    test = copy.deepcopy(reference)
    device = device_from_config(cfg.get("device", "auto"))
    reference.to(device).train()
    test.to(device).train()
    # Wrappers preserve Parameter identities but may introduce module nesting.
    other = dict(test.named_parameters())
    layers, _ = configure_model(test, {**cfg, "method": "stacr"})
    dtype = _amp_dtype(cfg, device)
    steps = int(cfg.get("fidelity_steps", cfg["steps"]))
    if steps < 1:
        raise ValueError("fidelity_steps must be positive")
    wrapped = {layer.source_name + ".weight" for layer in layers}
    full, local = [], []
    worst_unwrapped = 1.0
    for step in range(steps):
        batch = {k: v.to(device) for k, v in batches[step % len(batches)].items()}
        set_step(layers, step)
        # Fitting can use randomized SVD; finish it before the paired replay so
        # codec randomness cannot shift stochastic operations in that comparison.
        modes = {module: module.training for module in test.modules()}
        try:
            test.eval()
            for layer in layers:
                layer.training = True
            with torch.autocast(device_type=device.type, dtype=dtype, enabled=dtype is not None):
                fitted_output = test(**batch)
            del fitted_output
        finally:
            for module, training in modes.items():
                module.training = training
        for model in (reference, test):
            model.zero_grad(set_to_none=True)
            set_seed(int(cfg["seed"]) * 100003 + step + 1)
            with compression_diagnostics(model):
                with torch.autocast(
                    device_type=device.type, dtype=dtype, enabled=dtype is not None
                ):
                    loss = model(**batch).loss
                loss.backward()
        dot = ref_norm = test_norm = 0.0
        for name, parameter in reference.named_parameters():
            gr, gt = parameter.grad, other[name].grad
            if gr is None or gt is None:
                if (gr is None) != (gt is None):
                    raise RuntimeError(f"Gradient presence differs at {name}")
                continue
            a, b = gr.double(), gt.double()
            d, na, nb = float((a * b).sum()), float(a.square().sum()), float(b.square().sum())
            cosine = d / math.sqrt(na * nb) if na * nb > 0 else (1.0 if na == nb else 0.0)
            dot += d
            ref_norm += na
            test_norm += nb
            if name in wrapped:
                local.append(cosine)
            else:
                worst_unwrapped = min(worst_unwrapped, cosine)
        full.append(
            dot / math.sqrt(ref_norm * test_norm)
            if ref_norm * test_norm
            else (1.0 if ref_norm == test_norm else 0.0)
        )
    return {
        "model": cfg["model_name"],
        "steps": steps,
        "wrapped_layers": len(layers),
        "mean_cosine": sum(full) / len(full),
        "worst_cosine": min(full),
        "local_mean_cosine": sum(local) / len(local) if local else None,
        "local_worst_cosine": min(local) if local else None,
        "worst_unwrapped_cosine": worst_unwrapped,
        "protocol": "eval_same_batch_frozen_compression_state",
        "basis_fit": "preceding_dropout_disabled_forward",
    }


def run_config(config: dict[str, Any]) -> dict[str, Any]:
    cfg = validate_config(config)
    return run_synthetic(cfg) if cfg["task"] == "synthetic" else run_hf(cfg)


def run_many(configs, output_csv):
    rows = [run_config(cfg) for cfg in configs]
    write_csv(output_csv, rows)
    return rows
