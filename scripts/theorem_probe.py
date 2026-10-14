#!/usr/bin/env python3
"""
Post-hoc theorem probe on saved checkpoints.

Supports two modes:

1. Projection-only mode on a single checkpoint:
   - computes EV_A, q, p, kappa_G, observed projection cosine, and the
     projection lower bound sqrt(1 - p)

2. Paired baseline-vs-STAC-R mode:
   - loads both checkpoints on the same batch
   - computes the projection-path quantities from the baseline checkpoint
   - computes the actual full-method gradient error against the STAC-R checkpoint
   - reports s, cos_full, and the normalized cross-term xi

Paired checkpoints can have different trained weights. Their measured difference
includes training trajectory effects; it does not isolate compression at one
fixed model state. Use a same-state mechanism replay for that attribution.

Usage:
    # Single-checkpoint projection probe
    python scripts/theorem_probe.py \
        --checkpoint outputs/sst2/.../checkpoint/model.pt \
        --task sst2 --model bert-base-uncased --rank 8

    # Paired baseline-vs-STAC-R probe
    python scripts/theorem_probe.py \
        --baseline-checkpoint outputs/sst2/.../baseline/checkpoint/model.pt \
        --stacr-checkpoint outputs/sst2/.../stacr/checkpoint/model.pt \
        --task sst2 --model bert-base-uncased --rank 8
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


# Ensure project root is importable when running `python scripts/...`
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))


def _build_loader(task: str, model_name: str, max_length: int, batch_size: int):
    if task == "wikitext103":
        from experiments.protocols.training import ConvergenceConfig, setup_wikitext_data

        cfg = ConvergenceConfig(
            task="wikitext103",
            model=model_name,
            batch_size=batch_size,
            max_length=max_length,
            seed=42,
            quiet=True,
            num_workers=0,
        )
        _, val_loader, _ = setup_wikitext_data(cfg)
        return val_loader

    from datasets import load_dataset
    from transformers import AutoTokenizer, DataCollatorWithPadding

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    raw = load_dataset("glue", task)

    if task == "sst2":

        def preprocess(examples):
            result = tokenizer(
                examples["sentence"],
                truncation=True,
                max_length=max_length,
                padding=False,
            )
            result["labels"] = examples["label"]
            return result
    elif task == "mrpc":

        def preprocess(examples):
            result = tokenizer(
                examples["sentence1"],
                examples["sentence2"],
                truncation=True,
                max_length=max_length,
                padding=False,
            )
            result["labels"] = examples["label"]
            return result
    else:
        raise ValueError(f"Unsupported task: {task}")

    tokenized = raw.map(preprocess, batched=True, remove_columns=raw["train"].column_names)
    split = "validation" if "validation" in tokenized else "test"
    collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
    loader = DataLoader(tokenized[split], batch_size=batch_size, shuffle=False, collate_fn=collator)
    return loader


def _target_linear_names(model: nn.Module):
    from experiments.protocols.training import get_compression_targets_bert

    model_names = {name for name, _ in model.named_modules()}
    # Auto-detect GPT-2 vs BERT by checking for characteristic layer names
    if any("mlp.c_fc" in name for name in model_names):
        from experiments.protocols.training import get_compression_targets_gpt2

        names = get_compression_targets_gpt2(model)
    else:
        names = get_compression_targets_bert(model)
    # On a STAC-R-wrapped model, the naive substring matcher can also pick the
    # nested backing Linear (e.g. `...output.dense.linear`). The probe should
    # operate at the wrapper/module boundary used by training, not double-count
    # nested implementation details.
    return [name for name in names if not name.endswith(".linear")]


def _build_stacr_min_config(rank: int, residual_budget: float, residual_mode: str):
    return SimpleNamespace(
        method="stacr",
        rank=rank,
        stacr_rank=rank,
        stacr_residual_mode=residual_mode,
        stacr_residual_budget=residual_budget,
        stacr_warmup_steps=0,
        stacr_quantize_coefficients=True,
        stacr_quantize_bits=8,
        stacr_quantize_residuals=True,
        stacr_residual_quantize_bits=4,
        stacr_basis_dtype="float16",
        stacr_freeze_basis_after_warmup=True,
        stacr_residual_threshold=0.1,
        stacr_update_trigger="hybrid",
        stacr_update_interval=100,
        stacr_energy_threshold=0.05,
        stacr_streaming_algorithm="oja",
        stacr_oja_learning_rate=0.01,
        stacr_randomized_svd_threshold=1024,
        stacr_randomized_svd_oversampling=4,
        stacr_randomized_svd_niter=2,
        stacr_adaptive_oja_lr=False,
        stacr_adaptive_oja_scale=5.0,
        stacr_adaptive_oja_max_lr=0.1,
        stacr_residual_block_size=16,
        stacr_cpu_offload=False,
        stacr_use_mgs=False,
        stacr_lazy_update=True,
        stacr_lazy_update_threshold=0.01,
        stacr_use_fused_kernel=False,
        stacr_rank_mode="fixed",
        stacr_rank_variance_threshold=0.95,
        stacr_rank_min=4,
        stacr_rank_max=64,
        stacr_chunk_size=512,
        stacr_min_features=64,
        stacr_small_layer_rank_scale=1.0,
        stacr_small_layer_residual_scale=1.0,
        stacr_post_warmup_mode="full",
        stacr_residual_warmup_steps=0,
        stacr_post_warmup_threshold=0.01,
        stacr_adaptive_layer_rank="none",
        stacr_residual_schedule="none",
        stacr_residual_start=0.05,
        stacr_residual_end=0.001,
        stacr_basis_sharing="none",
        stacr_streaming_topk=False,
        stacr_streaming_topk_sample_ratio=0.1,
        stacr_index_compression="none",
        stacr_skip_residual=False,
        stacr_skip_residual_threshold=0.05,
        stacr_buffer_reuse=False,
        quiet=True,
    )


def _load_state_dict(checkpoint_path: str):
    return torch.load(checkpoint_path, map_location="cpu", weights_only=True)


def _load_model_state(model, state_dict):
    # Adaptive layers migrate absent legacy runtime metadata in their loader.
    # All parameter and basis tensors must still be present and shape-compatible.
    model.load_state_dict(state_dict, strict=True)
    return model


def _is_gpt2_model(model_name: str) -> bool:
    return "gpt2" in model_name.lower()


def _checkpoint_reference(path):
    if path is None:
        return None
    checksum = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            checksum.update(chunk)
    return {"filename": Path(path).name, "sha256": checksum.hexdigest()}


def _finite_json(value):
    if isinstance(value, dict):
        return {k: _finite_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_finite_json(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _load_plain_model(model_name: str, checkpoint_path: str, device: torch.device):
    if _is_gpt2_model(model_name):
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(model_name)
    else:
        from transformers import AutoModelForSequenceClassification

        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    state_dict = _load_state_dict(checkpoint_path)
    _load_model_state(model, state_dict)
    return model.to(device)


def _load_stacr_model(
    model_name: str,
    checkpoint_path: str,
    device: torch.device,
    rank: int,
    residual_budget: float,
    residual_mode: str,
):
    from experiments.protocols.training import replace_linear_with_stacr

    if _is_gpt2_model(model_name):
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(model_name)
    else:
        from transformers import AutoModelForSequenceClassification

        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    targets = _target_linear_names(model)
    cfg = _build_stacr_min_config(rank, residual_budget, residual_mode)
    model = replace_linear_with_stacr(model, cfg, targets)
    state_dict = _load_state_dict(checkpoint_path)
    _load_model_state(model, state_dict)
    return model.to(device)


def _collect_linear_stats(model: nn.Module, batch: Dict[str, torch.Tensor], device: torch.device):
    from stacr.diagnostics import compression_diagnostics

    model.zero_grad(set_to_none=True)
    activations: Dict[str, torch.Tensor] = {}
    grad_outputs: Dict[str, torch.Tensor] = {}
    weight_grads: Dict[str, torch.Tensor] = {}
    hooks = []

    target_names = set(_target_linear_names(model))

    def is_target(name: str, module: nn.Module) -> bool:
        return name in target_names

    def make_fwd_hook(name: str):
        def hook(module, inp, out):
            if inp and torch.is_tensor(inp[0]):
                activations[name] = inp[0].detach().clone()

        return hook

    def make_bwd_hook(name: str):
        def hook(module, grad_input, grad_output):
            if grad_output and torch.is_tensor(grad_output[0]):
                grad_outputs[name] = grad_output[0].detach().clone()

        return hook

    target_modules = []
    for name, module in model.named_modules():
        if is_target(name, module):
            target_modules.append((name, module))
            hooks.append(module.register_forward_hook(make_fwd_hook(name)))
            hooks.append(module.register_full_backward_hook(make_bwd_hook(name)))

    try:
        inputs = {k: v.to(device) for k, v in batch.items()}
        with compression_diagnostics(model), torch.enable_grad():
            outputs = model(**inputs)
            outputs.loss.backward()

        for name, module in target_modules:
            weight_param = None
            if hasattr(module, "weight"):
                weight_param = module.weight
            elif hasattr(module, "linear") and hasattr(module.linear, "weight"):
                weight_param = module.linear.weight
            grad = getattr(weight_param, "grad", None) if weight_param is not None else None
            if grad is not None:
                # GPT-2 Conv1D stores (input, output); use (output, input) here.
                if module.__class__.__name__ == "Conv1D":
                    grad = grad.t()
                weight_grads[name] = grad.detach().clone().reshape(-1).float()
    finally:
        for hook in hooks:
            hook.remove()
        model.zero_grad(set_to_none=True)
    return activations, grad_outputs, weight_grads


def _cosine_bound(squared_relative_error):
    """The sharp bound applies only when the relative error is below one."""
    if 0 <= squared_relative_error < 1:
        return math.sqrt(1 - squared_relative_error)
    return None


def _flatten_activation_and_grad(
    A: torch.Tensor, G: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    if A.dim() == 3:
        batch, length, features = A.shape
        A = A.reshape(batch * length, features)
        G = G.reshape(batch * length, G.shape[-1])
    elif A.dim() == 2:
        pass
    else:
        raise ValueError(f"Unsupported activation rank: {A.dim()}")
    return A.float(), G.float()


def _projection_metrics(activations, grad_outputs, weight_grads, rank: int):
    internal = {}
    serializable = {}

    for name, A in activations.items():
        if name not in grad_outputs:
            continue

        G = grad_outputs[name]
        try:
            A2, G2 = _flatten_activation_and_grad(A, G)
        except ValueError:
            continue
        if A2.shape[1] < rank * 2:
            continue

        g_true = (G2.T @ A2).reshape(-1).float()
        if g_true.norm().item() < 1e-20:
            continue

        try:
            _, _, Vh = torch.linalg.svd(A2, full_matrices=False)
            Q = Vh[:rank, :].T
            A_proj = A2 @ Q @ Q.T
        except Exception:
            continue

        g_proj = (G2.T @ A_proj).reshape(-1)
        delta_proj = g_true - g_proj

        total_var = (A2**2).sum().item()
        E = A2 - A_proj
        residual_var = (E**2).sum().item()
        ev_a = 1.0 - residual_var / (total_var + 1e-30)

        g_norm_sq = (g_true**2).sum().item()
        p = ((delta_proj**2).sum() / g_norm_sq).item()
        cos_proj = (torch.dot(g_true, g_proj) / (g_true.norm() * g_proj.norm() + 1e-30)).item()

        A_proj_norm_sq = (A_proj**2).sum().item()
        E_norm_sq = (E**2).sum().item()
        if A_proj_norm_sq > 1e-20 and E_norm_sq > 1e-20:
            mu_sig_sq = ((G2.T @ A_proj) ** 2).sum().item() / A_proj_norm_sq
            mu_res_sq = ((G2.T @ E) ** 2).sum().item() / E_norm_sq
            q = mu_sig_sq / (mu_res_sq + 1e-30)
        else:
            mu_sig_sq = 0.0
            mu_res_sq = 0.0
            q = float("nan")

        g_spectral = torch.linalg.norm(G2, ord=2).item()
        a_fro = math.sqrt(total_var)
        gta_fro = g_true.norm().item()
        kappa_g = g_spectral * a_fro / (gta_fro + 1e-30)
        cos_bound = _cosine_bound(p)

        cos_q_bridge = float("nan")
        if q > 0 and not math.isnan(q):
            cos_q_bridge_sq = q * ev_a / (q * ev_a + 1.0 - ev_a)
            cos_q_bridge = math.sqrt(max(cos_q_bridge_sq, 0.0))

        internal[name] = {
            "g_true": g_true,
            "g_proj": g_proj,
            "delta_proj": delta_proj,
        }
        serializable[name] = {
            "rank": rank,
            "shape": list(A2.shape),
            "EV_A": ev_a,
            "p": p,
            "q": q,
            "mu_sig_sq": mu_sig_sq,
            "mu_res_sq": mu_res_sq,
            "kappa_G": kappa_g,
            "cos_proj_observed": cos_proj,
            "cos_proj_bound": cos_bound,
            "cos_q_bridge": cos_q_bridge,
            "grad_vector_matches_weight_grad": (
                name in weight_grads
                and torch.allclose(g_true, weight_grads[name], atol=1e-4, rtol=1e-3)
            ),
        }

    return internal, serializable


def _pair_metrics(proj_internal, proj_serializable, stacr_weight_grads):
    paired = {}
    for name, vals in proj_internal.items():
        if name not in stacr_weight_grads:
            continue
        g_true = vals["g_true"]
        g_proj = vals["g_proj"]
        g_full = stacr_weight_grads[name]
        if g_true.shape != g_full.shape:
            continue

        delta_proj = vals["delta_proj"]
        delta_full = g_true - g_full
        delta_res = g_proj - g_full
        g_norm_sq = (g_true**2).sum().item()

        s = ((delta_full**2).sum() / g_norm_sq).item()
        cos_full = (torch.dot(g_true, g_full) / (g_true.norm() * g_full.norm() + 1e-30)).item()
        cos_full_bound = _cosine_bound(s)
        xi = (torch.dot(delta_proj, delta_res) / g_norm_sq).item()

        paired[name] = {
            **proj_serializable[name],
            "s": s,
            "xi": xi,
            "cos_full_observed": cos_full,
            "cos_full_bound": cos_full_bound,
        }
    return paired


def _aggregate(results: Dict[str, Dict[str, float]]):
    if not results:
        return {"num_layers": 0}

    def avg(key: str):
        vals = [
            v[key]
            for v in results.values()
            if isinstance(v.get(key), (float, int)) and math.isfinite(v[key])
        ]
        return sum(vals) / len(vals) if vals else None

    summary = {
        "num_layers": len(results),
        "projection_bound_applicable_layers": sum(
            v.get("cos_proj_bound") is not None for v in results.values()
        ),
        "mean_EV_A": avg("EV_A"),
        "mean_q": avg("q"),
        "mean_kappa_G": avg("kappa_G"),
        "mean_p": avg("p"),
        "mean_cos_proj_observed": avg("cos_proj_observed"),
        "mean_cos_proj_bound": avg("cos_proj_bound"),
        "mean_cos_q_bridge": avg("cos_q_bridge"),
        "min_cos_proj_observed": min(v["cos_proj_observed"] for v in results.values()),
        "max_kappa_G": max(v["kappa_G"] for v in results.values()),
    }
    if any("s" in v for v in results.values()):
        summary.update(
            {
                "full_bound_applicable_layers": sum(
                    v.get("cos_full_bound") is not None for v in results.values()
                ),
                "mean_s": avg("s"),
                "mean_xi": avg("xi"),
                "mean_cos_full_observed": avg("cos_full_observed"),
                "mean_cos_full_bound": avg("cos_full_bound"),
                "min_cos_full_observed": min(
                    v["cos_full_observed"] for v in results.values() if "cos_full_observed" in v
                ),
            }
        )
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description="Theorem probe on saved checkpoints")
    parser.add_argument(
        "--checkpoint", type=str, default=None, help="Single-checkpoint projection probe path"
    )
    parser.add_argument(
        "--baseline-checkpoint", type=str, default=None, help="Baseline checkpoint for paired probe"
    )
    parser.add_argument(
        "--stacr-checkpoint", type=str, default=None, help="STAC-R checkpoint for paired probe"
    )
    parser.add_argument("--task", type=str, default="sst2", choices=["sst2", "mrpc", "wikitext103"])
    parser.add_argument("--model", type=str, default="bert-base-uncased")
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stacr-residual-budget", type=float, default=0.005)
    parser.add_argument("--stacr-residual-mode", type=str, default="block_sparse")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args(argv)
    if args.rank < 1 or args.batch_size < 1 or args.max_length < 1:
        parser.error("rank, batch-size and max-length must be positive")

    single_mode = args.checkpoint is not None
    pair_mode = args.baseline_checkpoint is not None and args.stacr_checkpoint is not None
    if single_mode and pair_mode:
        raise SystemExit(
            "Use either --checkpoint or the paired --baseline-checkpoint/--stacr-checkpoint form, not both."
        )
    if not single_mode and not pair_mode:
        raise SystemExit(
            "Provide either --checkpoint or both --baseline-checkpoint and --stacr-checkpoint."
        )

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    loader = _build_loader(args.task, args.model, args.max_length, args.batch_size)
    batch = next(iter(loader))

    start = time.time()
    if pair_mode:
        baseline_model = _load_plain_model(args.model, args.baseline_checkpoint, device)
        stacr_model = _load_stacr_model(
            args.model,
            args.stacr_checkpoint,
            device,
            args.rank,
            args.stacr_residual_budget,
            args.stacr_residual_mode,
        )

        base_acts, base_grads, base_weight_grads = _collect_linear_stats(
            baseline_model, batch, device
        )
        _, _, stacr_weight_grads = _collect_linear_stats(stacr_model, batch, device)
        proj_internal, proj_serializable = _projection_metrics(
            base_acts, base_grads, base_weight_grads, args.rank
        )
        per_layer = _pair_metrics(proj_internal, proj_serializable, stacr_weight_grads)
        mode = "paired"
    else:
        model = _load_plain_model(args.model, args.checkpoint, device)
        acts, grads, weight_grads = _collect_linear_stats(model, batch, device)
        _, per_layer = _projection_metrics(acts, grads, weight_grads, args.rank)
        mode = "single"

    elapsed = time.time() - start
    summary = _aggregate(per_layer)
    if not summary["num_layers"]:
        raise RuntimeError(
            "No layers produced valid gradient measurements; check model targets, rank and checkpoint pair"
        )
    results = {
        "mode": mode,
        "metadata": {
            "task": args.task,
            "model": args.model,
            "rank": args.rank,
            "seed": args.seed,
            "probe_time_s": elapsed,
            "checkpoint": _checkpoint_reference(args.checkpoint),
            "baseline_checkpoint": _checkpoint_reference(args.baseline_checkpoint),
            "stacr_checkpoint": _checkpoint_reference(args.stacr_checkpoint),
            "comparison_scope": "different_trained_checkpoints"
            if pair_mode
            else "same_state_projection",
            "interpretation": "Paired gradient differences include training trajectory effects; they are not same-state compression errors."
            if pair_mode
            else "Projection is evaluated at one model state.",
            "dropout": "disabled",
            "compression": "enabled with read-only basis and runtime state",
            "checkpoint_loading": "strict weights and bases; absent legacy runtime metadata migrated by layer loader",
            "batch_size": args.batch_size,
            "max_length": args.max_length,
        },
        "summary": summary,
        "per_layer": per_layer,
    }

    print(f"\n{'=' * 64}")
    print(f" Theorem Probe Summary ({summary['num_layers']} layers, mode={mode})")
    print(f"{'=' * 64}")
    for key, value in summary.items():
        print(f"  {key}: {value if value is not None else 'not applicable'}")
    print(f"{'=' * 64}")

    if args.output is None:
        if pair_mode:
            args.output = str(Path(args.stacr_checkpoint).parent / "theorem_probe.json")
        else:
            args.output = str(Path(args.checkpoint).parent / "theorem_probe.json")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(
        json.dumps(_finite_json(results), indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
