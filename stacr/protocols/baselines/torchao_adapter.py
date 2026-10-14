"""
TorchAO adapter baseline.

TorchAO provides quantization utilities for PyTorch models. This adapter
uses weight-only quantization (int8/fp8) as a lightweight baseline and
falls back gracefully when torchao is unavailable or unsupported.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional, Tuple
import torch
import torch.nn as nn


class TorchAOStatus(Enum):
    """Status of TorchAO adapter."""

    ENABLED = "enabled"
    SKIPPED_MISSING_DEP = "skipped_missing_dep"
    SKIPPED_INCOMPATIBLE = "skipped_incompatible"
    ERROR = "error"


@dataclass
class TorchAOResult:
    """Result of TorchAO enable attempt."""

    status: TorchAOStatus
    reason: Optional[str] = None
    meta: Optional[Dict[str, Any]] = None

    @property
    def ok(self) -> bool:
        return self.status == TorchAOStatus.ENABLED

    def to_dict(self) -> Dict[str, Any]:
        return {"status": self.status.value, "reason": self.reason, "meta": self.meta or {}}


TORCHAO_MODES = ["auto", "int8", "fp8"]


def check_torchao_available() -> Tuple[bool, Optional[str]]:
    """Check if torchao is importable."""
    try:
        import torchao

        return (True, None)
    except ImportError as exc:
        return (False, f"torchao not installed: {exc}")
    except Exception as exc:
        return (False, f"torchao import error: {exc}")


def get_torchao_version() -> Optional[str]:
    """Get torchao version if available."""
    try:
        import torchao

        return getattr(torchao, "__version__", "unknown")
    except ImportError:
        return None


def _resolve_fp8_dtype() -> Optional[torch.dtype]:
    """Pick a supported FP8 dtype if available."""
    for name in ("float8_e4m3fn", "float8_e4m3fnuz", "float8_e5m2"):
        dtype = getattr(torch, name, None)
        if dtype is not None:
            return dtype
    return None


def _apply_torchao_quantization(model: nn.Module, mode: str) -> Tuple[nn.Module, str]:
    """Apply torchao quantization in-place when possible.

    Returns:
        (model, applied_mode)
    """
    try:
        from torchao.quantization import quantize_
    except Exception as exc:
        raise RuntimeError(f"torchao quantization API unavailable: {exc}") from exc
    applied_mode = mode
    if mode == "auto":
        applied_mode = "fp8" if _resolve_fp8_dtype() is not None else "int8"
    if applied_mode == "int8":
        try:
            from torchao.quantization.quant_api import Int8WeightOnlyConfig

            result = quantize_(model, Int8WeightOnlyConfig())
            if isinstance(result, nn.Module):
                model = result
            return (model, applied_mode)
        except ImportError:
            pass
        except Exception:
            pass
        try:
            result = quantize_(model, weight_dtype=torch.int8)
        except TypeError:
            result = quantize_(model, dtype=torch.int8)
        if isinstance(result, nn.Module):
            model = result
        return (model, applied_mode)
    if applied_mode == "fp8":
        fp8_dtype = _resolve_fp8_dtype()
        if fp8_dtype is None:
            raise RuntimeError("FP8 dtype not available in this PyTorch build")
        try:
            from torchao.quantization.quant_api import Float8WeightOnlyConfig

            result = quantize_(model, Float8WeightOnlyConfig())
            if isinstance(result, nn.Module):
                model = result
            return (model, applied_mode)
        except ImportError:
            pass
        except Exception:
            pass
        try:
            result = quantize_(model, weight_dtype=fp8_dtype)
        except TypeError:
            result = quantize_(model, dtype=fp8_dtype)
        if isinstance(result, nn.Module):
            model = result
        return (model, applied_mode)
    raise RuntimeError(f"Unsupported TorchAO mode: {applied_mode}")


def try_enable_torchao(model: nn.Module, mode: str = "auto") -> Tuple[nn.Module, TorchAOResult]:
    """Try to enable TorchAO quantization on a model.

    Args:
        model: PyTorch model to quantize
        mode: "auto", "int8", or "fp8"

    Returns:
        (model, TorchAOResult)
    """
    if mode not in TORCHAO_MODES:
        return (
            model,
            TorchAOResult(
                status=TorchAOStatus.ERROR, reason=f"Invalid mode '{mode}'. Valid: {TORCHAO_MODES}"
            ),
        )
    if not torch.cuda.is_available():
        return (
            model,
            TorchAOResult(
                status=TorchAOStatus.SKIPPED_INCOMPATIBLE,
                reason="CUDA not available (TorchAO baseline requires GPU)",
                meta={"requested_mode": mode},
            ),
        )
    (available, reason) = check_torchao_available()
    if not available:
        return (
            model,
            TorchAOResult(
                status=TorchAOStatus.SKIPPED_MISSING_DEP,
                reason=reason,
                meta={"requested_mode": mode},
            ),
        )
    try:
        (model, applied_mode) = _apply_torchao_quantization(model, mode)
        return (
            model,
            TorchAOResult(
                status=TorchAOStatus.ENABLED,
                reason=f"TorchAO enabled ({applied_mode})",
                meta={
                    "requested_mode": mode,
                    "applied_mode": applied_mode,
                    "version": get_torchao_version(),
                },
            ),
        )
    except Exception as exc:
        if mode == "auto":
            try:
                (model, applied_mode) = _apply_torchao_quantization(model, "int8")
                return (
                    model,
                    TorchAOResult(
                        status=TorchAOStatus.ENABLED,
                        reason="TorchAO enabled (int8 fallback)",
                        meta={
                            "requested_mode": mode,
                            "applied_mode": applied_mode,
                            "version": get_torchao_version(),
                            "fallback_reason": str(exc),
                        },
                    ),
                )
            except Exception as fallback_exc:
                return (
                    model,
                    TorchAOResult(
                        status=TorchAOStatus.ERROR,
                        reason=f"TorchAO fallback failed: {fallback_exc}",
                        meta={"requested_mode": mode, "error": str(exc)},
                    ),
                )
        return (
            model,
            TorchAOResult(
                status=TorchAOStatus.ERROR,
                reason=f"TorchAO enable failed: {exc}",
                meta={"requested_mode": mode, "error": str(exc)},
            ),
        )


def create_torchao_final_json(result: TorchAOResult, output_dir: str) -> Dict[str, Any]:
    """Create final.json content for a TorchAO run."""
    return {
        "baseline": "torchao",
        "status": result.status.value,
        "reason": result.reason,
        "meta": result.meta or {},
        "output_dir": output_dir,
    }
