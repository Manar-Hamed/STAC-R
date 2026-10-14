"""
Saved tensor compression baseline using saved_tensors_hooks.

Compresses tensors saved for backward to reduce memory, then reconstructs
them on demand in the backward pass. Uses tensorly for TT/Tucker compression.
"""

from __future__ import annotations
from contextlib import contextmanager
from typing import Any, Dict, Optional, Tuple
import time
import torch
from stacr.protocols.baselines.checkpointing import get_global_recomputation_guard


class SavedTensorCompressor:
    """Compress saved tensors with TT/Tucker during backward.

    Uses torch.autograd.graph.saved_tensors_hooks, similar to SavedTensorOffloader,
    but stores a compressed representation instead of offloading to CPU.
    """

    def __init__(
        self,
        method: str = "tt",
        rank: int = 4,
        min_bytes: int = 100000,
        skip_during_recomputation: bool = True,
        tensorize_dims: int = 4,
    ) -> None:
        if method not in ("tt", "tucker", "none"):
            raise ValueError(f"Unknown compression method: {method}")
        if tensorize_dims < 2:
            raise ValueError("tensorize_dims must be >= 2")
        self.method = method
        self.rank = rank
        self.min_bytes = min_bytes
        self.skip_recompute = skip_during_recomputation
        self.tensorize_dims = tensorize_dims
        self._guard = get_global_recomputation_guard()
        self._stats = {
            "compressed": 0,
            "skipped_small": 0,
            "skipped_nonfloat": 0,
            "skipped_recompute": 0,
            "skipped_missing_dep": 0,
            "skipped_leaf": 0,
            "skipped_leaf_view": 0,
            "errors": 0,
        }
        self._timing = {
            "pack": {"count": 0, "total_ms": 0.0},
            "unpack": {"count": 0, "total_ms": 0.0},
        }
        self._rss_peak_bytes: Optional[int] = None
        self._last_error: Optional[str] = None
        self._tensorly_ok: Optional[bool] = None

    def _ensure_tensorly(self) -> bool:
        if self.method == "none":
            return False
        if self._tensorly_ok is not None:
            return self._tensorly_ok
        try:
            import tensorly as tl

            tl.set_backend("pytorch")
            self._tensorly_ok = True
        except Exception as exc:
            self._tensorly_ok = False
            self._last_error = f"tensorly unavailable: {exc}"
        return self._tensorly_ok

    def pack(self, tensor: torch.Tensor) -> Tuple[Any, ...]:
        """Pack hook: compress tensor if eligible."""
        start = time.perf_counter()
        if self.method == "none":
            self._record_pack_time(start)
            return ("passthrough", tensor)
        if self.skip_recompute and self._guard.is_recomputing:
            self._stats["skipped_recompute"] += 1
            self._record_pack_time(start)
            return ("passthrough", tensor)
        tensor_bytes = tensor.numel() * tensor.element_size()
        if tensor_bytes < self.min_bytes:
            self._stats["skipped_small"] += 1
            self._record_pack_time(start)
            return ("passthrough", tensor)
        if not tensor.is_floating_point():
            self._stats["skipped_nonfloat"] += 1
            self._record_pack_time(start)
            return ("passthrough", tensor)
        if tensor.is_leaf:
            self._stats["skipped_leaf"] += 1
            self._record_pack_time(start)
            return ("passthrough", tensor)
        base = getattr(tensor, "_base", None)
        if base is not None and base.is_leaf:
            self._stats["skipped_leaf_view"] += 1
            self._record_pack_time(start)
            return ("passthrough", tensor)
        if not self._ensure_tensorly():
            self._stats["skipped_missing_dep"] += 1
            self._record_pack_time(start)
            return ("passthrough", tensor)
        try:
            factors = self._compress(tensor)
            self._stats["compressed"] += 1
            self._record_pack_time(start)
            return ("compressed", factors, tuple(tensor.shape), tensor.dtype, tensor.device)
        except Exception as exc:
            self._stats["errors"] += 1
            self._last_error = str(exc)
            self._record_pack_time(start)
            return ("passthrough", tensor)

    def unpack(self, packed: Tuple[Any, ...]) -> torch.Tensor:
        """Unpack hook: reconstruct tensor if compressed."""
        start = time.perf_counter()
        if not isinstance(packed, tuple):
            self._record_unpack_time(start)
            return packed
        tag = packed[0]
        if tag == "passthrough":
            self._record_unpack_time(start)
            return packed[1]
        if tag != "compressed":
            self._record_unpack_time(start)
            return packed
        (_, factors, shape, dtype, device) = packed
        tensor = self._decompress(factors)
        self._record_unpack_time(start)
        return tensor.reshape(shape).to(dtype=dtype, device=device)

    def _compress(self, tensor: torch.Tensor) -> Any:
        tensor_cpu = tensor.detach().to("cpu")
        tensor_shape = self._tensorize_shape(tensor_cpu.numel())
        reshaped = tensor_cpu.reshape(tensor_shape)
        if self.method == "tt":
            from tensorly.decomposition import tensor_train

            tt_obj = tensor_train(reshaped, rank=self.rank)
            if hasattr(tt_obj, "factors"):
                return [core.detach().cpu() for core in tt_obj.factors]
            return [core.detach().cpu() for core in tt_obj]
        if self.method == "tucker":
            from tensorly.decomposition import tucker

            rank = self._normalize_tucker_rank(tensor_shape)
            (core, factors) = tucker(reshaped, rank=rank)
            core = core.detach().cpu()
            factors = [f.detach().cpu() for f in factors]
            return (core, factors)
        raise RuntimeError(f"Unsupported compression method: {self.method}")

    def _decompress(self, factors: Any) -> torch.Tensor:
        if self.method == "tt":
            from tensorly import tt_to_tensor

            return tt_to_tensor(factors)
        if self.method == "tucker":
            from tensorly import tucker_to_tensor

            (core, factor_mats) = factors
            return tucker_to_tensor((core, factor_mats))
        raise RuntimeError(f"Unsupported compression method: {self.method}")

    def _tensorize_shape(self, numel: int) -> Tuple[int, ...]:
        target = int(round(numel ** (1.0 / self.tensorize_dims)))
        remaining = numel
        dims = []
        for _ in range(self.tensorize_dims - 1):
            factor = min(target, remaining)
            while factor > 1 and remaining % factor != 0:
                factor -= 1
            dims.append(factor)
            remaining //= factor
        dims.append(remaining)
        return tuple(dims)

    def _normalize_tucker_rank(self, shape: Tuple[int, ...]) -> Tuple[int, ...]:
        if isinstance(self.rank, (list, tuple)):
            rank = list(self.rank)
        else:
            rank = [int(self.rank)] * len(shape)
        return tuple((min(int(r), int(dim)) for (r, dim) in zip(rank, shape)))

    @contextmanager
    def context(self):
        if self.method == "none":
            yield self
            return
        with torch.autograd.graph.saved_tensors_hooks(self.pack, self.unpack):
            yield self

    def get_stats(self) -> Dict[str, Any]:
        return {
            "method": self.method,
            "rank": self.rank,
            "min_bytes": self.min_bytes,
            "tensorize_dims": self.tensorize_dims,
            "stats": dict(self._stats),
            "timing": dict(self._timing),
            "cpu_rss_peak_bytes": self._rss_peak_bytes,
            "last_error": self._last_error,
        }

    def reset(self) -> None:
        for key in self._stats:
            self._stats[key] = 0
        for key in self._timing:
            self._timing[key]["count"] = 0
            self._timing[key]["total_ms"] = 0.0
        self._rss_peak_bytes = None
        self._last_error = None

    def _record_pack_time(self, start: float) -> None:
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        self._timing["pack"]["count"] += 1
        self._timing["pack"]["total_ms"] += elapsed_ms
        self._update_rss_peak()

    def _record_unpack_time(self, start: float) -> None:
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        self._timing["unpack"]["count"] += 1
        self._timing["unpack"]["total_ms"] += elapsed_ms
        self._update_rss_peak()

    def _update_rss_peak(self) -> None:
        try:
            import psutil

            rss = psutil.Process().memory_info().rss
        except Exception:
            return
        if self._rss_peak_bytes is None or rss > self._rss_peak_bytes:
            self._rss_peak_bytes = rss
