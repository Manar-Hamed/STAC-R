"""GaLore optimizer projection adapter for the training protocols.

The result object records unavailable dependencies and incompatible settings;
the experiment runner must require an enabled result before training."""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.optim as optim


class GaLoreStatus(Enum):
    """Status of GaLore adapter."""

    ENABLED = "enabled"
    SKIPPED_MISSING_DEP = "skipped_missing_dep"
    SKIPPED_INCOMPATIBLE = "skipped_incompatible"
    ERROR = "error"


@dataclass
class GaLoreResult:
    """Result of GaLore enable attempt."""

    status: GaLoreStatus
    reason: Optional[str] = None
    meta: Optional[Dict[str, Any]] = None

    @property
    def ok(self) -> bool:
        return self.status == GaLoreStatus.ENABLED

    def to_dict(self) -> Dict[str, Any]:
        return {"status": self.status.value, "reason": self.reason, "meta": self.meta or {}}


@dataclass
class GaLoreConfig:
    """Configuration for GaLore optimizer."""

    rank: int = 128
    update_proj_gap: int = 200
    scale: float = 0.25
    proj_type: str = "std"


def check_galore_available() -> Tuple[bool, Optional[str]]:
    """Check if galore-torch is available.

    Returns:
        Tuple of (available, reason_if_not)
    """
    try:
        from galore_torch import GaLoreAdamW

        return (True, None)
    except ImportError as e:
        return (False, f"galore-torch not installed: {e}. Install with: pip install galore-torch")
    except Exception as e:
        return (False, f"galore-torch import error: {e}")


def check_pytorch_version() -> Tuple[bool, Optional[str]]:
    """Check if PyTorch version is compatible with GaLore.

    GaLore requires PyTorch >= 2.1 for register_post_accumulate_grad_hook.
    """
    version = torch.__version__
    try:
        parts = version.split(".")[:2]
        major = int(parts[0])
        minor_str = parts[1].split("+")[0].split("a")[0].split("b")[0].split("rc")[0]
        minor = int(minor_str)
    except (ValueError, IndexError):
        return (True, f"Could not parse PyTorch version '{version}', assuming compatible")
    if major < 2:
        return (False, f"GaLore requires PyTorch >= 2.1, got {version}")
    if major == 2 and minor < 1:
        return (False, f"GaLore requires PyTorch >= 2.1, got {version}")
    return (True, None)


def get_galore_version() -> Optional[str]:
    """Get galore-torch version if available."""
    try:
        import galore_torch

        return getattr(galore_torch, "__version__", "unknown")
    except ImportError:
        return None


def create_galore_optimizer(
    model: nn.Module,
    lr: float = 0.01,
    weight_decay: float = 0.0,
    config: Optional[GaLoreConfig] = None,
    target_modules: Optional[List[str]] = None,
) -> Tuple[Optional[optim.Optimizer], GaLoreResult]:
    """Create GaLore optimizer for a model.

    Args:
        model: Model to optimize
        lr: Learning rate
        weight_decay: Weight decay
        config: GaLore configuration (uses defaults if None)
        target_modules: List of module name patterns to apply GaLore to.
            If None, applies to all parameters with dim >= 2.

    Returns:
        Tuple of (optimizer or None, GaLoreResult)
    """
    (available, reason) = check_galore_available()
    if not available:
        return (None, GaLoreResult(status=GaLoreStatus.SKIPPED_MISSING_DEP, reason=reason))
    (compatible, compat_reason) = check_pytorch_version()
    if not compatible:
        return (None, GaLoreResult(status=GaLoreStatus.SKIPPED_INCOMPATIBLE, reason=compat_reason))
    if config is None:
        config = GaLoreConfig()
    try:
        from galore_torch import GaLoreAdamW

        galore_params = []
        regular_params = []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            apply_galore = False
            if target_modules is not None:
                for pattern in target_modules:
                    if pattern in name:
                        apply_galore = True
                        break
            else:
                apply_galore = len(param.shape) >= 2
            if apply_galore:
                galore_params.append(param)
            else:
                regular_params.append(param)
        param_groups = []
        if regular_params:
            param_groups.append({"params": regular_params, "weight_decay": weight_decay})
        if galore_params:
            param_groups.append(
                {
                    "params": galore_params,
                    "rank": config.rank,
                    "update_proj_gap": config.update_proj_gap,
                    "scale": config.scale,
                    "proj_type": config.proj_type,
                    "weight_decay": weight_decay,
                }
            )
        if not param_groups:
            return (
                None,
                GaLoreResult(status=GaLoreStatus.ERROR, reason="No parameters to optimize"),
            )
        optimizer = GaLoreAdamW(param_groups, lr=lr)
        return (
            optimizer,
            GaLoreResult(
                status=GaLoreStatus.ENABLED,
                reason=f"GaLore enabled (rank={config.rank}, {len(galore_params)} GaLore params, {len(regular_params)} regular)",
                meta={
                    "rank": config.rank,
                    "update_proj_gap": config.update_proj_gap,
                    "scale": config.scale,
                    "proj_type": config.proj_type,
                    "galore_param_count": len(galore_params),
                    "regular_param_count": len(regular_params),
                    "galore_version": get_galore_version(),
                },
            ),
        )
    except Exception as e:
        return (
            None,
            GaLoreResult(
                status=GaLoreStatus.ERROR,
                reason=f"Failed to create GaLore optimizer: {e}",
                meta={"error": str(e)},
            ),
        )


def try_enable_galore(
    model: nn.Module,
    lr: float = 0.01,
    weight_decay: float = 0.0,
    rank: int = 128,
    update_proj_gap: int = 200,
    scale: float = 0.25,
    proj_type: str = "std",
    target_modules: Optional[List[str]] = None,
) -> Tuple[Optional[optim.Optimizer], GaLoreResult]:
    """Try to enable GaLore optimizer for a model.

    Convenience wrapper around create_galore_optimizer with explicit parameters.

    Args:
        model: Model to optimize
        lr: Learning rate
        weight_decay: Weight decay
        rank: Low-rank projection dimension
        update_proj_gap: Update projection every N steps
        scale: Scaling factor
        proj_type: Projection type
        target_modules: Module name patterns for GaLore

    Returns:
        Tuple of (optimizer or None, GaLoreResult)
    """
    config = GaLoreConfig(
        rank=rank, update_proj_gap=update_proj_gap, scale=scale, proj_type=proj_type
    )
    return create_galore_optimizer(
        model=model, lr=lr, weight_decay=weight_decay, config=config, target_modules=target_modules
    )


def parse_galore_config_string(config_str: str) -> Optional[GaLoreConfig]:
    """Parse GaLore config from string format.

    Format: "rank=128,update_gap=200,scale=0.25"

    Returns:
        GaLoreConfig or None if parsing fails
    """
    if not config_str or config_str.lower() == "none":
        return None
    config = GaLoreConfig()
    try:
        for item in config_str.split(","):
            if "=" not in item:
                continue
            (key, value) = item.split("=", 1)
            key = key.strip().lower()
            value = value.strip()
            if key == "rank":
                config.rank = int(value)
            elif key in ("update_gap", "update_proj_gap"):
                config.update_proj_gap = int(value)
            elif key == "scale":
                config.scale = float(value)
            elif key in ("proj_type", "type"):
                config.proj_type = value
        return config
    except Exception:
        return None
