"""
Automatic mixed precision with dtype selection and gradient scaling.

Provides:
- Context manager for AMP with proper dtype selection
- Auto-detection of bf16 support
- GradScaler management for fp16
"""

from contextlib import contextmanager
from typing import Optional, Tuple, Generator
import torch
from torch.cuda.amp import GradScaler


def resolve_amp_dtype(amp_mode: str, device: str = "cuda") -> Tuple[Optional[torch.dtype], bool]:
    """Resolve AMP mode to dtype and whether to use GradScaler.

    Args:
        amp_mode: AMP mode string ("none", "fp16", "bf16", "auto")
        device: Device type or indexed device ("cuda", "cuda:1", "cpu")

    Returns:
        Tuple of (dtype, use_scaler):
        - dtype: torch.float16, torch.bfloat16, or None (for full precision)
        - use_scaler: Whether to use GradScaler (only for fp16 on CUDA)
    """
    device = torch.device(device)
    if amp_mode == "none":
        return (None, False)
    if amp_mode == "fp16":
        return (torch.float16, device.type == "cuda")
    if amp_mode == "bf16":
        return (torch.bfloat16, False)
    if amp_mode == "auto":
        if _is_bf16_supported(device):
            return (torch.bfloat16, False)
        else:
            return (torch.float16, device.type == "cuda")
    raise ValueError(f"Unknown AMP mode: {amp_mode}")


def _is_bf16_supported(device: str) -> bool:
    """Check if bf16 is supported on the device."""
    device = torch.device(device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            return False
        try:
            capability = torch.cuda.get_device_capability(device)
            return capability[0] >= 8
        except Exception:
            return False
    elif device.type == "cpu":
        return hasattr(torch, "bfloat16")
    return False


@contextmanager
def amp_context(
    amp_mode: str, device: str = "cuda", enabled: bool = True
) -> Generator[Optional[GradScaler], None, None]:
    """Context manager for AMP with optional GradScaler.

    Args:
        amp_mode: AMP mode ("none", "fp16", "bf16", "auto")
        device: Device type ("cuda", "cpu")
        enabled: Whether AMP is actually enabled (allows runtime toggle)

    Yields:
        GradScaler if fp16 on CUDA, else None

    Usage:
        with amp_context("bf16", device="cuda") as scaler:
            output = model(input)
            loss = criterion(output, target)
            if scaler:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
    """
    if not enabled or amp_mode == "none":
        yield None
        return
    (dtype, use_scaler) = resolve_amp_dtype(amp_mode, device)
    if dtype is None:
        yield None
        return
    device_type = torch.device(device).type if device else "cpu"
    scaler = GradScaler() if use_scaler else None
    with torch.autocast(device_type=device_type, dtype=dtype):
        yield scaler


class AMPManager:
    """Stateful AMP manager for training loops.

    Handles GradScaler state across multiple steps.

    Usage:
        amp_mgr = AMPManager("bf16", device="cuda")

        for batch in dataloader:
            with amp_mgr.autocast():
                output = model(batch)
                loss = criterion(output, target)

            amp_mgr.backward(loss)
            amp_mgr.step(optimizer)
            optimizer.zero_grad()
    """

    def __init__(self, amp_mode: str, device: str = "cuda"):
        self.amp_mode = amp_mode
        self.device = device
        (self.dtype, use_scaler) = resolve_amp_dtype(amp_mode, device)
        self.scaler = GradScaler() if use_scaler else None
        self._device_type = torch.device(device).type if device else "cpu"

    @property
    def enabled(self) -> bool:
        """Check if AMP is enabled."""
        return self.dtype is not None

    @contextmanager
    def autocast(self):
        """Context manager for autocast."""
        if self.dtype is None:
            yield
        else:
            with torch.autocast(device_type=self._device_type, dtype=self.dtype):
                yield

    def backward(self, loss: torch.Tensor) -> None:
        """Backward pass with optional scaling."""
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

    def step(self, optimizer: torch.optim.Optimizer) -> None:
        """Optimizer step with optional unscaling."""
        if self.scaler is not None:
            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            optimizer.step()

    def get_scale(self) -> float:
        """Get current scale factor (1.0 if no scaler)."""
        if self.scaler is not None:
            return self.scaler.get_scale()
        return 1.0

    def state_dict(self) -> dict:
        """Get state dict for checkpointing."""
        return {
            "amp_mode": self.amp_mode,
            "scaler_state": self.scaler.state_dict() if self.scaler else None,
        }

    def load_state_dict(self, state_dict: dict) -> None:
        """Load state dict from checkpoint."""
        if self.scaler is not None and state_dict.get("scaler_state"):
            self.scaler.load_state_dict(state_dict["scaler_state"])
