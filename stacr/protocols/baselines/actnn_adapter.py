"""
ActNN adapter for T12.

ActNN is an activation-compressed training library using 2-4 bit quantization.
Paper: https://arxiv.org/abs/2104.14129
Repo: https://github.com/ucbrise/actnn

This adapter provides:
- Graceful skip if ActNN not installed (older PyTorch requirement)
- Unified interface matching other baselines
- Level-based configuration (L0-L4, L5 skipped)

Levels:
- L0: Disabled (baseline)
- L1: 4-bit conv + 32-bit bn
- L2: 4-bit uniform
- L3: 2-bit
- L4: 2-bit + swap to CPU
"""

import collections.abc
import os
import queue
import sys
import types
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Literal, Optional, Tuple
import torch
import torch.nn as nn


class ActNNStatus(Enum):
    """Status of ActNN adapter."""

    ENABLED = "enabled"
    SKIPPED_MISSING_DEP = "skipped_missing_dep"
    SKIPPED_INCOMPATIBLE = "skipped_incompatible"
    ERROR = "error"


@dataclass
class ActNNResult:
    """Result of ActNN enable attempt."""

    status: ActNNStatus
    reason: Optional[str] = None
    meta: Optional[Dict[str, Any]] = None

    @property
    def ok(self) -> bool:
        return self.status == ActNNStatus.ENABLED

    def to_dict(self) -> Dict[str, Any]:
        return {"status": self.status.value, "reason": self.reason, "meta": self.meta or {}}


ACTNN_LEVELS = ["L0", "L1", "L2", "L3", "L4"]
ACTNN_L5_REASON = (
    "L5 requires special PyTorch fork: https://github.com/merrymercy/pytorch/tree/actnn_exp"
)
ACTNN_PT2_EXPERIMENT_ENV = "ACTNN_ALLOW_PT2_EXPERIMENTAL"


def _pt2_experimental_enabled() -> bool:
    return os.environ.get(ACTNN_PT2_EXPERIMENT_ENV, "").lower() in {"1", "true", "yes"}


def _install_torch_six_shim() -> None:
    """ActNN imports torch._six through copied PyTorch-1.x DataLoader helpers.

    PyTorch 2.x removed torch._six. The shim is only installed for the explicit
    experimental PT2 path; it is not a claim that upstream ActNN officially
    supports modern PyTorch.
    """
    if "torch._six" in sys.modules:
        return
    torch_six = types.ModuleType("torch._six")
    torch_six.string_classes = (str,)
    torch_six.int_classes = (int,)
    torch_six.container_abcs = collections.abc
    torch_six.queue = queue
    sys.modules["torch._six"] = torch_six


def check_actnn_available() -> Tuple[bool, Optional[str]]:
    """Check if ActNN is available and compatible.

    Returns:
        Tuple of (available, reason_if_not)
    """
    try:
        if _pt2_experimental_enabled():
            _install_torch_six_shim()
        import actnn

        return (True, None)
    except ImportError as e:
        return (False, f"ActNN not installed: {e}")
    except Exception as e:
        return (False, f"ActNN import error: {e}")


def get_actnn_version() -> Optional[str]:
    """Get ActNN version if available."""
    try:
        import actnn

        return getattr(actnn, "__version__", "unknown")
    except ImportError:
        return None


def check_pytorch_compatibility() -> Tuple[bool, Optional[str]]:
    """Check if PyTorch version is compatible with ActNN.

    ActNN typically targets PyTorch 1.x; newer versions often break.
    """
    version = torch.__version__
    try:
        parts = version.split(".")[:2]
        major = int(parts[0])
        minor_str = parts[1].split("+")[0].split("a")[0].split("b")[0].split("rc")[0]
        minor = int(minor_str)
    except (ValueError, IndexError):
        return (True, f"Could not parse PyTorch version '{version}', assuming compatible")
    if major >= 2:
        if _pt2_experimental_enabled():
            return (
                True,
                f"ActNN PT2 experimental override enabled via {ACTNN_PT2_EXPERIMENT_ENV}; upstream ActNN expects PyTorch 1.x, current torch={version}",
            )
        return (False, f"ActNN expects PyTorch 1.x, got {version}")
    if major == 1 and minor < 8:
        return (False, f"ActNN requires PyTorch >= 1.8, got {version}")
    return (True, None)


def try_enable_actnn(model: nn.Module, level: str = "L2") -> Tuple[nn.Module, ActNNResult]:
    """Try to enable ActNN compression on a model.

    Args:
        model: PyTorch model to wrap
        level: ActNN optimization level (L0-L4)

    Returns:
        Tuple of (model, result):
        - model: Original or wrapped model
        - result: ActNNResult with status and metadata
    """
    if level == "L5":
        return (
            model,
            ActNNResult(
                status=ActNNStatus.SKIPPED_INCOMPATIBLE,
                reason=ACTNN_L5_REASON,
                meta={"requested_level": level},
            ),
        )
    if level not in ACTNN_LEVELS:
        return (
            model,
            ActNNResult(
                status=ActNNStatus.ERROR, reason=f"Invalid level '{level}'. Valid: {ACTNN_LEVELS}"
            ),
        )
    if level == "L0":
        return (
            model,
            ActNNResult(
                status=ActNNStatus.ENABLED,
                reason="L0 = disabled (passthrough)",
                meta={"level": level, "compression": False},
            ),
        )
    if not torch.cuda.is_available():
        return (
            model,
            ActNNResult(
                status=ActNNStatus.SKIPPED_INCOMPATIBLE,
                reason="CUDA not available (ActNN requires GPU)",
                meta={"requested_level": level},
            ),
        )
    (compatible, compat_reason) = check_pytorch_compatibility()
    if not compatible:
        return (
            model,
            ActNNResult(
                status=ActNNStatus.SKIPPED_INCOMPATIBLE,
                reason=compat_reason,
                meta={"requested_level": level, "pytorch_version": torch.__version__},
            ),
        )
    (available, reason) = check_actnn_available()
    if not available:
        return (
            model,
            ActNNResult(
                status=ActNNStatus.SKIPPED_MISSING_DEP,
                reason=reason,
                meta={"requested_level": level},
            ),
        )
    try:
        if _pt2_experimental_enabled():
            _install_torch_six_shim()
        import actnn
        from actnn import QModule

        actnn.set_optimization_level(level)
        wrapped_model = QModule(model)
        return (
            wrapped_model,
            ActNNResult(
                status=ActNNStatus.ENABLED,
                reason=f"ActNN enabled at level {level}",
                meta={
                    "level": level,
                    "version": get_actnn_version(),
                    "compression": True,
                    "bits": _get_bits_for_level(level),
                    "pytorch_compat_warning": compat_reason,
                    "pt2_experimental": _pt2_experimental_enabled(),
                },
            ),
        )
    except Exception as e:
        return (
            model,
            ActNNResult(
                status=ActNNStatus.ERROR,
                reason=f"Failed to enable ActNN: {e}",
                meta={"requested_level": level, "error": str(e)},
            ),
        )


def _get_bits_for_level(level: str) -> int:
    """Get approximate bits per activation for a level."""
    level_bits = {"L0": 32, "L1": 4, "L2": 4, "L3": 2, "L4": 2}
    return level_bits.get(level, 4)


class ActNNController:
    """Controller for ActNN integration in training loops.

    Provides a unified interface similar to GACT's controller pattern.

    Usage:
        controller = ActNNController(model, level="L2")
        result = controller.enable()

        if result.ok:
            model = controller.model
            # training loop...

        # At end:
        controller.disable()
    """

    def __init__(self, model: nn.Module, level: str = "L2"):
        if level not in ACTNN_LEVELS:
            raise ValueError(f"Invalid level '{level}'. Valid: {ACTNN_LEVELS}")
        self.original_model = model
        self.level = level
        self.model = model
        self._enabled = False
        self._result: Optional[ActNNResult] = None

    def enable(self) -> ActNNResult:
        """Enable ActNN compression."""
        if self._result is not None:
            return self._result
        (self.model, self._result) = try_enable_actnn(self.original_model, level=self.level)
        self._enabled = self._result.ok
        return self._result

    def disable(self) -> None:
        """Disable ActNN compression (restore original model)."""
        if not self._enabled:
            return
        self.model = self.original_model
        self._enabled = False

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def result(self) -> Optional[ActNNResult]:
        return self._result

    def get_config(self) -> Dict[str, Any]:
        """Get current configuration for logging."""
        return {
            "baseline": "actnn",
            "level": self.level,
            "enabled": self._enabled,
            "status": self._result.status.value if self._result else "not_initialized",
        }

    def __enter__(self) -> "ActNNController":
        """Context manager entry - enable compression."""
        self.enable()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit - disable compression."""
        self.disable()


def create_actnn_final_json(result: ActNNResult, output_dir: str) -> Dict[str, Any]:
    """Create final.json content for an ActNN run.

    Even if ActNN is not available, we create a proper final.json
    so sweeps remain reproducible and plottable.
    """
    return {
        "baseline": "actnn",
        "status": result.status.value,
        "reason": result.reason,
        "meta": result.meta or {},
        "output_dir": output_dir,
    }
