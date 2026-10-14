from __future__ import annotations
from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set
import torch
import torch.nn as nn
from stacr.protocols.baselines.amp import resolve_amp_dtype
from stacr.memory_accounting import bytes_per_elem, stacr_ckpt_stored_bytes, stacr_stored_bytes
from stacr.protocols.checkpointing import CheckpointSiteCandidate, collect_c1_site_candidates


@dataclass
class LayerDecision:
    name: str
    keep: bool
    reason: str
    in_features: int
    batch_tokens_est: int
    saved_activation_bytes: int
    stacr_stored_bytes: int
    net_bytes: int
    forced: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CheckpointSiteDecision:
    name: str
    keep: bool
    reason: str
    hidden_size: int
    batch_tokens_est: int
    saved_boundary_bytes: int
    stacr_stored_bytes: int
    basis_persistent_bytes: int
    net_bytes: int
    forced: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _is_hf_conv1d(module: nn.Module) -> bool:
    return module.__class__.__name__ == "Conv1D"


def _module_in_features(module: nn.Module) -> Optional[int]:
    if isinstance(module, nn.Linear):
        return int(module.in_features)
    if _is_hf_conv1d(module):
        return int(module.weight.shape[0])
    if hasattr(module, "in_features"):
        return int(getattr(module, "in_features"))
    if hasattr(module, "in_channels"):
        return int(getattr(module, "in_channels"))
    return None


def _resolve_planner_basis_dtype(config: Any) -> torch.dtype:
    raw = getattr(config, "stacr_basis_dtype", "auto")
    if not isinstance(raw, str):
        return torch.float32
    value = raw.lower()
    if value == "auto":
        baseline_cfg = getattr(config, "baseline_cfg", None)
        amp_mode = getattr(baseline_cfg, "amp", "none")
        device_value = getattr(config, "device", "cuda")
        try:
            device_type = torch.device(device_value).type
        except (TypeError, RuntimeError, ValueError):
            device_type = "cuda"
        (resolved_dtype, _) = resolve_amp_dtype(amp_mode, device=device_type)
        if resolved_dtype == torch.bfloat16:
            value = "bfloat16"
        elif resolved_dtype == torch.float16:
            value = "float16"
        else:
            value = "float32"
    mapping = {
        "float16": torch.float16,
        "half": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    return mapping.get(value, torch.float32)


def _resolve_planner_saved_tensor_dtype(config: Any) -> torch.dtype:
    baseline_cfg = getattr(config, "baseline_cfg", None)
    amp_mode = getattr(baseline_cfg, "amp", "none")
    device_value = getattr(config, "device", "cuda")
    try:
        device_type = torch.device(device_value).type
    except (TypeError, RuntimeError, ValueError):
        device_type = "cuda"
    (resolved_dtype, _) = resolve_amp_dtype(amp_mode, device=device_type)
    return resolved_dtype if resolved_dtype is not None else torch.float32


def _estimate_batch_tokens(config: Any) -> int:
    task = getattr(config, "task", None)
    batch_size = int(getattr(config, "batch_size", 1))
    max_length = int(getattr(config, "max_length", 1))
    if task in ("sst2", "wikitext103"):
        return batch_size * max_length
    return batch_size


def _effective_layer_rank(
    config: Any, *, in_features: int, layer_index: int, total_layers: int
) -> int:
    rank = int(getattr(config, "rank", 8))
    if in_features < 1024:
        small_scale = float(getattr(config, "stacr_small_layer_rank_scale", 1.0))
        if small_scale != 1.0:
            rank = max(4, int(rank * small_scale))
    adaptive_mode = getattr(config, "stacr_adaptive_layer_rank", "none")
    if adaptive_mode == "heuristic":
        if layer_index < total_layers // 3:
            multiplier = 1.0
        elif layer_index < 2 * total_layers // 3:
            multiplier = 0.7
        else:
            multiplier = 0.4
        rank = max(8, int(rank * multiplier))
    return min(rank, in_features)


def _effective_residual_budget(config: Any, *, in_features: int) -> float:
    residual_mode = getattr(config, "stacr_residual_mode", "topk")
    if residual_mode == "none":
        return 0.0
    budget = float(getattr(config, "stacr_residual_budget", 0.0))
    if in_features < 1024:
        budget *= float(getattr(config, "stacr_small_layer_residual_scale", 1.0))
    return budget


def _quant_bits(config: Any) -> tuple[Optional[int], Optional[int]]:
    coeff_bits = None
    if getattr(config, "stacr_quantize_coefficients", False):
        coeff_bits = int(getattr(config, "stacr_quantize_bits", 8))
    residual_bits = None
    if getattr(config, "stacr_quantize_residuals", False):
        residual_bits = int(getattr(config, "stacr_residual_quantize_bits", 8))
    return (coeff_bits, residual_bits)


def _parse_force_names(raw: Optional[str]) -> Set[str]:
    if raw is None:
        return set()
    names = [part.strip() for part in raw.split(",") if part.strip()]
    return set(names)


def _checkpointing_eliminates_decoder_mlp_saves(mode: Any) -> bool:
    if not isinstance(mode, str):
        return False
    normalized = mode.strip().lower()
    passthrough_modes = {"none", "", "attention_only", "vision_stage", "vision_block"}
    eliminating_modes = {"transformer_hf", "full"}
    if normalized in passthrough_modes:
        return False
    if normalized in eliminating_modes:
        return True
    raise ValueError(f"Unknown checkpointing mode for planner profitability logic: {mode!r}")


def plan_stacr_layers(
    model: nn.Module, config: Any, candidate_layers: Sequence[str]
) -> List[LayerDecision]:
    module_dict = dict(model.named_modules())
    candidate_set = set(candidate_layers)
    forced_names = _parse_force_names(getattr(config, "stacr_force_compress_layers", None))
    unknown_forced = sorted(forced_names - candidate_set)
    if unknown_forced:
        raise ValueError(
            "Unknown --stacr-force-compress-layers entries for current candidate set: "
            + ", ".join(unknown_forced)
        )
    batch_tokens = _estimate_batch_tokens(config)
    checkpointing = getattr(getattr(config, "baseline_cfg", None), "checkpointing", "none")
    ckpt_active = _checkpointing_eliminates_decoder_mlp_saves(checkpointing)
    basis_dtype = _resolve_planner_basis_dtype(config)
    saved_tensor_dtype = _resolve_planner_saved_tensor_dtype(config)
    (coeff_bits, residual_bits) = _quant_bits(config)
    min_features = int(getattr(config, "stacr_min_features", 0))
    decisions: List[LayerDecision] = []
    total_layers = len(candidate_layers)
    for idx, name in enumerate(candidate_layers):
        module = module_dict.get(name)
        in_features = _module_in_features(module) if module is not None else None
        if in_features is None:
            if name in forced_names:
                raise RuntimeError(
                    f"Forced planner keep target '{name}' could not be resolved to a supported module."
                )
            decisions.append(
                LayerDecision(
                    name=name,
                    keep=False,
                    reason="unsupported_module",
                    in_features=-1,
                    batch_tokens_est=batch_tokens,
                    saved_activation_bytes=0,
                    stacr_stored_bytes=0,
                    net_bytes=0,
                    forced=False,
                )
            )
            continue
        if min_features > 0 and in_features < min_features:
            if name in forced_names:
                raise RuntimeError(
                    f"Forced planner keep target '{name}' is below stacr_min_features={min_features}."
                )
            decisions.append(
                LayerDecision(
                    name=name,
                    keep=False,
                    reason="below_min_features",
                    in_features=in_features,
                    batch_tokens_est=batch_tokens,
                    saved_activation_bytes=0,
                    stacr_stored_bytes=0,
                    net_bytes=0,
                    forced=False,
                )
            )
            continue
        rank = _effective_layer_rank(
            config, in_features=in_features, layer_index=idx, total_layers=total_layers
        )
        residual_budget = _effective_residual_budget(config, in_features=in_features)
        stored = stacr_stored_bytes(
            batch_tokens,
            in_features,
            rank,
            residual_budget,
            saved_tensor_dtype,
            coeff_bits=coeff_bits,
            residual_bits=residual_bits,
            basis_dtype=basis_dtype,
            include_meta=False,
        ).total_bytes
        saved = (
            0 if ckpt_active else batch_tokens * in_features * bytes_per_elem(saved_tensor_dtype)
        )
        net = saved - stored
        forced = name in forced_names
        if forced:
            keep = True
            reason = "forced_keep"
        elif ckpt_active:
            keep = False
            reason = "checkpointing_zero_save"
        elif net > 0:
            keep = True
            reason = "positive_net"
        else:
            keep = False
            reason = "non_positive_net"
        decisions.append(
            LayerDecision(
                name=name,
                keep=keep,
                reason=reason,
                in_features=in_features,
                batch_tokens_est=batch_tokens,
                saved_activation_bytes=saved,
                stacr_stored_bytes=stored,
                net_bytes=net,
                forced=forced,
            )
        )
    return decisions


def summarize_layer_decisions(
    decisions: Iterable[LayerDecision], *, target_mode: str, planner_enforced: bool
) -> Dict[str, Any]:
    decision_list = list(decisions)
    kept = sum((1 for d in decision_list if d.keep))
    forced = sum((1 for d in decision_list if d.keep and d.forced))
    return {
        "candidate_count": len(decision_list),
        "kept_count": kept,
        "skipped_count": len(decision_list) - kept,
        "forced_count": forced,
        "target_mode": target_mode,
        "planner_enforced": bool(planner_enforced),
    }


def plan_stacr_checkpoint_sites(
    model: nn.Module, config: Any, *, candidates: Optional[Sequence[CheckpointSiteCandidate]] = None
) -> List[CheckpointSiteDecision]:
    candidate_list = (
        list(candidates) if candidates is not None else collect_c1_site_candidates(model, config)
    )
    checkpointing = getattr(getattr(config, "baseline_cfg", None), "checkpointing", "none")
    batch_tokens = _estimate_batch_tokens(config)
    saved_tensor_dtype = _resolve_planner_saved_tensor_dtype(config)
    decisions: List[CheckpointSiteDecision] = []
    for candidate in candidate_list:
        spec = candidate.spec
        hidden_size = int(spec.hidden_size)
        if checkpointing != "transformer_hf":
            decisions.append(
                CheckpointSiteDecision(
                    name=spec.site_id,
                    keep=False,
                    reason="unsupported_checkpoint_mode",
                    hidden_size=hidden_size,
                    batch_tokens_est=batch_tokens,
                    saved_boundary_bytes=0,
                    stacr_stored_bytes=0,
                    basis_persistent_bytes=0,
                    net_bytes=0,
                )
            )
            continue
        if batch_tokens * hidden_size < int(spec.min_numel):
            decisions.append(
                CheckpointSiteDecision(
                    name=spec.site_id,
                    keep=False,
                    reason="below_min_numel",
                    hidden_size=hidden_size,
                    batch_tokens_est=batch_tokens,
                    saved_boundary_bytes=0,
                    stacr_stored_bytes=0,
                    basis_persistent_bytes=0,
                    net_bytes=0,
                )
            )
            continue
        if getattr(config, "method", None) in ("stacr", "stacr-v4") and (
            not getattr(config, "stacr_ckpt_allow_inner_stacr", False)
        ):
            decisions.append(
                CheckpointSiteDecision(
                    name=spec.site_id,
                    keep=False,
                    reason="inner_stacr_conflict",
                    hidden_size=hidden_size,
                    batch_tokens_est=batch_tokens,
                    saved_boundary_bytes=0,
                    stacr_stored_bytes=0,
                    basis_persistent_bytes=0,
                    net_bytes=0,
                )
            )
            continue
        residual_bits = spec.residual_quantize_bits if spec.quantize_residuals else None
        coeff_bits = spec.quantize_bits if spec.quantize_coefficients else None
        breakdown = stacr_ckpt_stored_bytes(
            batch_tokens,
            hidden_size,
            int(spec.rank),
            residual_budget=float(spec.residual_budget if spec.residual_mode != "none" else 0.0),
            dtype=saved_tensor_dtype,
            coeff_bits=coeff_bits,
            residual_bits=residual_bits,
            basis_dtype=spec.basis_dtype,
            include_meta=False,
            include_basis=False,
        )
        raw_bytes = batch_tokens * hidden_size * bytes_per_elem(saved_tensor_dtype)
        persistent_basis_bytes = int(breakdown.persistent_cache_bytes)
        effective_stored = int(breakdown.total_bytes + persistent_basis_bytes)
        net = int(raw_bytes - effective_stored)
        keep = net > 0
        decisions.append(
            CheckpointSiteDecision(
                name=spec.site_id,
                keep=keep,
                reason="positive_boundary_net" if keep else "non_positive_boundary_net",
                hidden_size=hidden_size,
                batch_tokens_est=batch_tokens,
                saved_boundary_bytes=int(raw_bytes),
                stacr_stored_bytes=int(breakdown.total_bytes),
                basis_persistent_bytes=persistent_basis_bytes,
                net_bytes=net,
            )
        )
    return decisions


def summarize_checkpoint_site_decisions(
    decisions: Iterable[CheckpointSiteDecision],
    *,
    target: str,
    planner_mode: str,
    planner_enforced: bool,
) -> Dict[str, Any]:
    decision_list = list(decisions)
    kept = sum((1 for d in decision_list if d.keep))
    forced = sum((1 for d in decision_list if d.keep and d.forced))
    return {
        "candidate_count": len(decision_list),
        "kept_count": kept,
        "skipped_count": len(decision_list) - kept,
        "forced_count": forced,
        "target": target,
        "planner_mode": planner_mode,
        "planner_enforced": bool(planner_enforced),
    }
