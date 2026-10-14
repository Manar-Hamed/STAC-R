"""
GACT adapter for T12.

GACT is a gradient-aware activation compression training library.
Paper: https://arxiv.org/abs/2206.11357
Repo: https://github.com/LiuXiaoxuanPKU/GACT-ICML

This adapter provides:
- Graceful skip if GACT not installed (requires PyTorch 1.10-1.11)
- Unified interface matching other baselines
- Level-based configuration (L1-L2)
- Controller pattern for saved_tensors hooks

Paper levels:
- L1: per-group quantization with auto-precision (4-bit target)
- L2: L1 + swapping/prefetching (4-bit target)
- CB1/CB2: checkpointing/attention variants, outside this adapter.

Released-code level names in `gact/conf.py` are slightly different from the
paper table:
- L1: fixed 4-bit
- L2: auto-precision 4-bit
- L3: auto-precision 4-bit + swap/prefetch (closest to paper-table L2)
- L1.1/L1.2: fixed 8-bit / 2-bit diagnostics
- L2.1/L2.2: auto-precision 3-bit / 2-bit diagnostics

Key insight from GACT code:
- Uses saved_tensors_default_hooks for compression
- Controller.iterate() needed for auto-precision (optional)
- Does NOT support gradient_accumulation_steps > 1
- Only quantizes 2D/3D/4D tensors; 5D+ tensors are skipped
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, Optional, Tuple, Union
import torch
import torch.nn as nn


class GACTStatus(Enum):
    """Status of GACT adapter."""

    ENABLED = "enabled"
    SKIPPED_MISSING_DEP = "skipped_missing_dep"
    SKIPPED_INCOMPATIBLE = "skipped_incompatible"
    ERROR = "error"


@dataclass
class GACTResult:
    """Result of GACT enable attempt."""

    status: GACTStatus
    reason: Optional[str] = None
    meta: Optional[Dict[str, Any]] = None

    @property
    def ok(self) -> bool:
        return self.status == GACTStatus.ENABLED

    def to_dict(self) -> Dict[str, Any]:
        return {"status": self.status.value, "reason": self.reason, "meta": self.meta or {}}


GACT_LEVELS = ["L1", "L1.1", "L1.2", "L2", "L2.1", "L2.2", "L3", "L4bit-swap"]


def check_gact_available() -> Tuple[bool, Optional[str]]:
    """Check if GACT is available and compatible.

    Returns:
        Tuple of (available, reason_if_not)
    """
    try:
        import gact

        return (True, None)
    except ImportError as e:
        return (False, f"GACT not installed: {e}")
    except Exception as e:
        return (False, f"GACT import error: {e}")


def get_gact_version() -> Optional[str]:
    """Get GACT version if available."""
    try:
        import gact

        return getattr(gact, "__version__", "unknown")
    except ImportError:
        return None


def check_pytorch_compatibility() -> Tuple[bool, Optional[str]]:
    """Check if PyTorch version is compatible with GACT.

    GACT requires PyTorch >= 1.10 for saved_tensors hooks.
    Best with 1.10-1.11, may have issues with 2.x.
    """
    version = torch.__version__
    try:
        parts = version.split(".")[:2]
        major = int(parts[0])
        minor_str = parts[1].split("+")[0].split("a")[0].split("b")[0].split("rc")[0]
        minor = int(minor_str)
    except (ValueError, IndexError):
        return (True, f"Could not parse PyTorch version '{version}', assuming compatible")
    if major < 1 or (major == 1 and minor < 10):
        return (False, f"GACT requires PyTorch >= 1.10, got {version}")
    if major >= 2:
        return (True, f"PyTorch {version} may have compatibility issues with GACT")
    return (True, None)


def try_enable_gact(
    model: nn.Module, level: str = "L1", auto_precision: bool = False
) -> Tuple[nn.Module, Union["GACTController", "MockGACTController"], GACTResult]:
    """Try to enable GACT compression on a model.

    Args:
        model: PyTorch model to wrap
        level: GACT optimization level (L1-L2)
        auto_precision: Enable adaptive precision (requires iterate() calls)

    Returns:
        Tuple of (model, controller, result):
        - model: Original model (GACT uses hooks, not wrapping)
        - controller: GACTController for managing hooks
        - result: GACTResult with status and metadata
    """
    if level not in GACT_LEVELS:
        return (
            model,
            MockGACTController(model, level),
            GACTResult(
                status=GACTStatus.ERROR, reason=f"Invalid level '{level}'. Valid: {GACT_LEVELS}"
            ),
        )
    (available, reason) = check_gact_available()
    if not available:
        return (
            model,
            MockGACTController(model, level),
            GACTResult(
                status=GACTStatus.SKIPPED_MISSING_DEP,
                reason=reason,
                meta={"requested_level": level},
            ),
        )
    if not torch.cuda.is_available():
        return (
            model,
            MockGACTController(model, level),
            GACTResult(
                status=GACTStatus.SKIPPED_INCOMPATIBLE,
                reason="CUDA not available (GACT requires GPU)",
                meta={"requested_level": level},
            ),
        )
    (compatible, compat_reason) = check_pytorch_compatibility()
    if not compatible:
        return (
            model,
            MockGACTController(model, level),
            GACTResult(
                status=GACTStatus.SKIPPED_INCOMPATIBLE,
                reason=compat_reason,
                meta={"requested_level": level, "pytorch_version": torch.__version__},
            ),
        )
    try:
        import gact
        from gact.controller import Controller

        gact.set_optimization_level(level)
        native_controller = Controller(model)
        native_auto_precision = bool(
            auto_precision or getattr(native_controller, "auto_prec", False)
        )
        controller = GACTController(
            model=model,
            level=level,
            auto_precision=native_auto_precision,
            native_controller=native_controller,
        )
        return (
            model,
            controller,
            GACTResult(
                status=GACTStatus.ENABLED,
                reason=f"GACT enabled at level {level}",
                meta={
                    "level": level,
                    "version": get_gact_version(),
                    "auto_precision": native_auto_precision,
                    "bits": _get_bits_for_level(level),
                    "pytorch_compat_warning": compat_reason,
                    "tensor_dim_limit": "2D/3D/4D only",
                    "skips_5d_tensors": True,
                },
            ),
        )
    except Exception as e:
        return (
            model,
            MockGACTController(model, level),
            GACTResult(
                status=GACTStatus.ERROR,
                reason=f"Failed to enable GACT: {e}",
                meta={"requested_level": level, "error": str(e)},
            ),
        )


def _get_bits_for_level(level: str) -> int:
    """Get approximate bits per activation for a level."""
    level_bits = {
        "L1": 4,
        "L1.1": 8,
        "L1.2": 2,
        "L2": 4,
        "L2.1": 3,
        "L2.2": 2,
        "L3": 4,
        "L4bit-swap": 4,
    }
    return level_bits.get(level, 4)


class GACTController:
    """Controller for GACT integration in training loops.

    Wraps GACT's native Controller with a unified interface.

    Usage:
        model, controller, result = try_enable_gact(model, level="L1")

        if result.ok:
            controller.install_hook()

            for step, batch in enumerate(dataloader):
                output = model(batch)
                loss = criterion(output)
                loss.backward()
                optimizer.step()

                # Optional: call iterate() for auto-precision
                if controller.auto_precision:
                    controller.iterate(get_grad_fn)

            controller.uninstall_hook()

    IMPORTANT: GACT does not support gradient_accumulation_steps > 1.
    Each forward-backward pair must be independent.
    """

    def __init__(
        self,
        model: nn.Module,
        level: str,
        auto_precision: bool = False,
        native_controller: Optional[Any] = None,
    ):
        self.model = model
        self.level = level
        self.auto_precision = auto_precision
        self._native = native_controller
        self._hooks_installed = False
        self._iter = 0

    def install_hook(self) -> None:
        """Install saved_tensors compression hooks."""
        if self._hooks_installed:
            return
        if self._native is not None:
            self._native.install_hook()
            self._hooks_installed = True

    def uninstall_hook(self) -> None:
        """Remove saved_tensors compression hooks."""
        if not self._hooks_installed:
            return
        if self._native is not None:
            self._native.uninstall_hook()
            try:
                self._native.uninstall_hook = lambda: None
            except Exception:
                pass
            self._hooks_installed = False

    def iterate(self, get_grad_fn: Optional[Callable] = None) -> None:
        """Call after each step for auto-precision updates.

        Args:
            get_grad_fn: Function that runs forward+backward pass
                         (needed for sensitivity estimation)
        """
        if self._native is not None:
            if self.auto_precision and get_grad_fn is None:
                raise ValueError(
                    "GACT auto-precision levels require a replay backprop callback when controller.iterate() is called."
                )
            self._native.iterate(get_grad_fn)
        self._iter += 1

    @property
    def enabled(self) -> bool:
        return self._hooks_installed

    @property
    def iteration(self) -> int:
        return self._iter

    def get_config(self) -> Dict[str, Any]:
        """Get current configuration for logging."""
        return {
            "baseline": "gact",
            "level": self.level,
            "auto_precision": self.auto_precision,
            "hooks_installed": self._hooks_installed,
            "iteration": self._iter,
        }

    def __enter__(self) -> "GACTController":
        self.install_hook()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.uninstall_hook()


class MockGACTController:
    """Mock controller when GACT is not available.

    Provides the same interface but does nothing.
    Useful for code that conditionally uses GACT.
    """

    def __init__(self, model: nn.Module, level: str = "L1"):
        self.model = model
        self.level = level
        self.auto_precision = False
        self._iter = 0

    def install_hook(self) -> None:
        pass

    def uninstall_hook(self) -> None:
        pass

    def iterate(self, get_grad_fn: Optional[Callable] = None) -> None:
        self._iter += 1

    @property
    def enabled(self) -> bool:
        return False

    @property
    def iteration(self) -> int:
        return self._iter

    def get_config(self) -> Dict[str, Any]:
        return {
            "baseline": "gact",
            "level": self.level,
            "enabled": False,
            "reason": "mock_controller",
        }

    def __enter__(self) -> "MockGACTController":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        pass


def create_gact_final_json(result: GACTResult, output_dir: str) -> Dict[str, Any]:
    """Create final.json content for a GACT run.

    Even if GACT is not available, we create a proper final.json
    so sweeps remain reproducible and plottable.
    """
    return {
        "baseline": "gact",
        "status": result.status.value,
        "reason": result.reason,
        "meta": result.meta or {},
        "output_dir": output_dir,
    }
