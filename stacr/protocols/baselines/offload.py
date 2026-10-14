"""
Saved tensor offloading baseline.

Implements GPU -> CPU offloading of saved tensors during backward pass
using torch.autograd.graph.saved_tensors_hooks.

This baseline helps answer: "What if we just offload activations to CPU?"
"""

import time
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Tuple
import torch


class SavedTensorOffloader:
    """Offload saved tensors to CPU during backward pass.

    Uses saved_tensors_hooks to intercept saved tensors and move them to CPU,
    then back to GPU during backward. This trades GPU memory for CPU memory
    and transfer time.

    Attributes:
        mode: Offloading mode ("none", "cpu", "cpu_pinned")
        min_bytes: Minimum tensor size to offload (skip small tensors)
        pack_times: List of pack (GPU->CPU) times in ms
        unpack_times: List of unpack (CPU->GPU) times in ms

    Usage:
        offloader = SavedTensorOffloader(mode="cpu", min_bytes=1_000_000)
        with offloader.context():
            output = model(input)
            loss = criterion(output, target)
            loss.backward()  # Tensors offloaded during backward
        print(offloader.get_timing_stats())
    """

    def __init__(self, mode: str = "cpu", min_bytes: int = 1000000):
        """Initialize offloader.

        Args:
            mode: Offloading mode:
                - "none": No offloading
                - "cpu": Regular CPU memory
                - "cpu_pinned": Pinned CPU memory (faster async transfer)
            min_bytes: Minimum tensor size in bytes to offload
        """
        if mode not in ("none", "cpu", "cpu_pinned"):
            raise ValueError(f"Unknown offload mode: {mode}")
        self.mode = mode
        self.min_bytes = min_bytes
        self.pack_times: List[float] = []
        self.unpack_times: List[float] = []
        self._offloaded_count = 0
        self._skipped_count = 0

    def pack(self, tensor: torch.Tensor) -> Tuple[Any, torch.device, torch.dtype]:
        """Pack hook: move tensor to CPU if large enough.

        Args:
            tensor: Tensor to pack (from forward pass)

        Returns:
            Tuple of (packed_data, original_device, original_dtype)
        """
        if self.mode == "none":
            return (tensor, tensor.device, tensor.dtype)
        tensor_bytes = tensor.numel() * tensor.element_size()
        if tensor_bytes < self.min_bytes:
            self._skipped_count += 1
            return (tensor, tensor.device, tensor.dtype)
        self._offloaded_count += 1
        original_device = tensor.device
        original_dtype = tensor.dtype
        start = time.perf_counter()
        if self.mode == "cpu_pinned" and tensor.is_cuda and tensor.layout == torch.strided:
            # Allocate pinned storage first. Copying an unfinished asynchronous
            # CPU transfer into a second pinned buffer can capture stale data.
            cpu_tensor = torch.empty_like(
                tensor, device="cpu", pin_memory=True, memory_format=torch.preserve_format
            )
            cpu_tensor.copy_(tensor)
        else:
            cpu_tensor = tensor.to("cpu")
        if tensor.is_cuda:
            torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - start) * 1000
        self.pack_times.append(elapsed_ms)
        return (cpu_tensor, original_device, original_dtype)

    def unpack(self, packed: Tuple[Any, torch.device, torch.dtype]) -> torch.Tensor:
        """Unpack hook: move tensor back to original device.

        Args:
            packed: Tuple from pack()

        Returns:
            Tensor on original device
        """
        (tensor, original_device, original_dtype) = packed
        if self.mode == "none":
            return tensor
        if tensor.device == original_device:
            return tensor
        start = time.perf_counter()
        if self.mode == "cpu_pinned":
            result = tensor.to(original_device, non_blocking=True)
        else:
            result = tensor.to(original_device)
        if original_device.type == "cuda":
            torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - start) * 1000
        self.unpack_times.append(elapsed_ms)
        return result

    @contextmanager
    def context(self):
        """Context manager for saved_tensors_hooks.

        Usage:
            with offloader.context():
                loss.backward()
        """
        if self.mode == "none":
            yield self
            return
        with torch.autograd.graph.saved_tensors_hooks(self.pack, self.unpack):
            yield self

    def get_timing_stats(self) -> Dict[str, Any]:
        """Get timing statistics.

        Returns:
            Dict with pack/unpack timing stats and counts
        """
        import numpy as np

        def stats(times: List[float]) -> Dict[str, float]:
            if not times:
                return {"count": 0, "total_ms": 0.0, "mean_ms": 0.0, "p50_ms": 0.0, "p90_ms": 0.0}
            return {
                "count": len(times),
                "total_ms": sum(times),
                "mean_ms": float(np.mean(times)),
                "p50_ms": float(np.percentile(times, 50)),
                "p90_ms": float(np.percentile(times, 90)),
            }

        return {
            "mode": self.mode,
            "min_bytes": self.min_bytes,
            "offloaded_count": self._offloaded_count,
            "skipped_count": self._skipped_count,
            "pack": stats(self.pack_times),
            "unpack": stats(self.unpack_times),
        }

    def reset(self) -> None:
        """Reset timing statistics."""
        self.pack_times.clear()
        self.unpack_times.clear()
        self._offloaded_count = 0
        self._skipped_count = 0


class OffloadManager:
    """High-level manager for tensor offloading with RSS tracking.

    Combines SavedTensorOffloader with CPU RSS monitoring.

    Usage:
        mgr = OffloadManager(mode="cpu", min_bytes=1_000_000)
        with mgr.training_step():
            output = model(input)
            loss = criterion(output, target)
            loss.backward()
        print(mgr.get_stats())
    """

    def __init__(self, mode: str = "cpu", min_bytes: int = 1000000):
        self.offloader = SavedTensorOffloader(mode=mode, min_bytes=min_bytes)
        self._rss_before: Optional[int] = None
        self._rss_peak: Optional[int] = None

    @contextmanager
    def training_step(self):
        """Context manager for a single training step with offloading."""
        self._rss_before = self._get_rss()
        with self.offloader.context():
            yield
        self._rss_peak = self._get_rss()

    def _get_rss(self) -> int:
        """Get current RSS in bytes."""
        try:
            import psutil

            return psutil.Process().memory_info().rss
        except ImportError:
            return 0

    def get_stats(self) -> Dict[str, Any]:
        """Get combined timing and memory stats."""
        stats = self.offloader.get_timing_stats()
        stats["cpu_rss"] = {
            "before_bytes": self._rss_before,
            "peak_bytes": self._rss_peak,
            "delta_bytes": self._rss_peak - self._rss_before
            if self._rss_before is not None and self._rss_peak is not None
            else None,
        }
        return stats

    def reset(self) -> None:
        """Reset all statistics."""
        self.offloader.reset()
        self._rss_before = None
        self._rss_peak = None
