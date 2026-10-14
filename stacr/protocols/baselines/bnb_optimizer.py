"""Optional bitsandbytes optimizer factory and byte estimates.

Estimates exclude quantization metadata and temporary workspace; use measured
peaks to compare complete training configurations."""

from dataclasses import dataclass
from enum import Enum
import inspect
from typing import Any, Dict, Iterator, Optional, Tuple
import torch
import torch.nn as nn


class BNBStatus(Enum):
    """Status of bitsandbytes optimizer."""

    ENABLED = "enabled"
    SKIPPED_MISSING_DEP = "skipped_missing_dep"
    SKIPPED_CPU = "skipped_cpu"
    ERROR = "error"


@dataclass
class BNBResult:
    """Result of 8-bit optimizer creation."""

    status: BNBStatus
    optimizer: Optional[torch.optim.Optimizer] = None
    reason: Optional[str] = None
    meta: Optional[Dict[str, Any]] = None

    @property
    def ok(self) -> bool:
        return self.status == BNBStatus.ENABLED and self.optimizer is not None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status.value,
            "reason": self.reason,
            "meta": self.meta or {},
            "optimizer_type": type(self.optimizer).__name__ if self.optimizer else None,
        }


def check_bnb_available() -> Tuple[bool, Optional[str]]:
    """Check if bitsandbytes is available.

    Returns:
        Tuple of (available, reason_if_not)
    """
    try:
        import bitsandbytes as bnb

        return (True, None)
    except ImportError as e:
        return (False, f"bitsandbytes not installed: {e}. Install with: pip install bitsandbytes")
    except Exception as e:
        return (False, f"bitsandbytes import error: {e}")


def try_create_8bit_optimizer(
    params: Iterator[nn.Parameter],
    lr: float = 0.0001,
    weight_decay: float = 0.01,
    optim_type: str = "adamw",
    betas: Tuple[float, float] = (0.9, 0.999),
    eps: float = 1e-08,
    percentile_clipping: int = 100,
    block_wise: bool = True,
    min_8bit_size: int = 4096,
) -> BNBResult:
    """Create an 8-bit optimizer if bitsandbytes is available.

    Args:
        params: Model parameters to optimize
        lr: Learning rate
        weight_decay: Weight decay (L2 regularization)
        optim_type: Optimizer type ("adamw", "adam", or "sgd")
        betas: Adam beta parameters
        eps: Adam epsilon
        percentile_clipping: Dynamic range clipping percentile
        block_wise: Use block-wise quantization (recommended for stability)
        min_8bit_size: Minimum tensor size to use 8-bit (smaller tensors stay fp32)

    Returns:
        BNBResult with optimizer if successful
    """
    params_list = list(params)
    if not params_list:
        return BNBResult(status=BNBStatus.ERROR, reason="No parameters provided")
    first_param = params_list[0]
    if not first_param.is_cuda:
        return BNBResult(
            status=BNBStatus.SKIPPED_CPU, reason="8-bit optimizer requires CUDA tensors"
        )
    (available, reason) = check_bnb_available()
    if not available:
        return BNBResult(status=BNBStatus.SKIPPED_MISSING_DEP, reason=reason)
    try:
        import bitsandbytes as bnb

        omitted_defaults = []
        if optim_type.lower() in ("adamw", "adam"):
            optimizer_class = (
                bnb.optim.AdamW8bit if optim_type.lower() == "adamw" else bnb.optim.Adam8bit
            )
            signature = inspect.signature(optimizer_class).parameters
            accepts_kwargs = any(
                p.kind == inspect.Parameter.VAR_KEYWORD for p in signature.values()
            )
            legacy_options = {}
            # New bitsandbytes versions always use block-wise quantization and
            # no percentile clipping. Omit only those equivalent defaults;
            # unsupported nondefault requests must fail explicitly.
            for name, value, default in (
                ("percentile_clipping", percentile_clipping, 100),
                ("block_wise", block_wise, True),
            ):
                if name in signature or accepts_kwargs:
                    legacy_options[name] = value
                elif value != default:
                    raise ValueError(
                        f"Installed bitsandbytes no longer supports {name}={value!r}; "
                        f"use {default!r} or a version supporting this option"
                    )
                else:
                    omitted_defaults.append(name)
            optimizer = optimizer_class(
                params_list,
                lr=lr,
                betas=betas,
                eps=eps,
                weight_decay=weight_decay,
                min_8bit_size=min_8bit_size,
                **legacy_options,
            )
        elif optim_type.lower() == "sgd":
            optimizer = bnb.optim.SGD8bit(
                params_list, lr=lr, weight_decay=weight_decay, momentum=0.9
            )
        else:
            return BNBResult(
                status=BNBStatus.ERROR,
                reason=f"Unknown optimizer type: {optim_type}. Use 'adamw', 'adam', or 'sgd'",
            )
        total_params = sum((p.numel() for p in params_list))
        eligible_params = sum((p.numel() for p in params_list if p.numel() >= min_8bit_size))
        ineligible_params = total_params - eligible_params
        if optim_type.lower() == "sgd":
            fp32_bytes_per_param = 4
            int8_bytes_per_param = 1
        else:
            fp32_bytes_per_param = 8
            int8_bytes_per_param = 2
        fp32_optimizer_bytes = total_params * fp32_bytes_per_param
        quantized_optimizer_bytes = (
            eligible_params * int8_bytes_per_param + ineligible_params * fp32_bytes_per_param
        )
        savings_mb = max(fp32_optimizer_bytes - quantized_optimizer_bytes, 0) / 1000000.0
        return BNBResult(
            status=BNBStatus.ENABLED,
            optimizer=optimizer,
            meta={
                "optim_type": optim_type,
                "total_params": total_params,
                "eligible_params": eligible_params,
                "ineligible_params": ineligible_params,
                "estimated_savings_mb": savings_mb,
                "block_wise": block_wise,
                "percentile_clipping": percentile_clipping,
                "compatibility_omitted_defaults": omitted_defaults,
                "min_8bit_size": min_8bit_size,
            },
        )
    except Exception as e:
        return BNBResult(status=BNBStatus.ERROR, reason=f"Failed to create 8-bit optimizer: {e}")


def estimate_optimizer_savings(model: nn.Module, *, min_8bit_size: int = 4096) -> Dict[str, float]:
    """Estimate memory savings from 8-bit AdamW.

    Args:
        model: The model to analyze
        min_8bit_size: Minimum tensor size that qualifies for 8-bit state

    Returns:
        Dict with memory estimates in MB
    """
    params = list(model.parameters())
    total_params = sum((p.numel() for p in params))
    eligible_params = sum((p.numel() for p in params if p.numel() >= min_8bit_size))
    ineligible_params = total_params - eligible_params
    fp32_size_mb = total_params * 8 / 1000000.0
    int8_size_mb = (eligible_params * 2 + ineligible_params * 8) / 1000000.0
    return {
        "total_params": total_params,
        "eligible_params": eligible_params,
        "ineligible_params": ineligible_params,
        "fp32_optimizer_mb": fp32_size_mb,
        "int8_optimizer_mb": int8_size_mb,
        "savings_mb": fp32_size_mb - int8_size_mb,
        "savings_percent": (1 - int8_size_mb / fp32_size_mb) * 100 if fp32_size_mb else 0.0,
    }
