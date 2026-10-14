"""Adaptive STAC-R with coefficient quantization, sparse residuals, and basis tracking.

This engine supports fixed or adaptive rank, refresh schedules, basis sharing,
index compression, CPU offload, and optional fused projection kernels.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import asdict, dataclass, field
from typing import Optional, Literal, Tuple, List
from enum import Enum
import math
import warnings
from contextlib import nullcontext
import threading
import copy
import json

# Index compression imports
from .index_compression import (
    IndexCompressionMode,
    CompressedIndices,
    compress_indices,
    decompress_indices,
)
from .rank_controller import choose_runtime_rank, should_adjust_rank

_FUSED_LOGGED = False
_FUSED_FALLBACK_LOGGED = False


def _layer_phase_context(layer_state: dict, name: str):
    profiler = layer_state.get("phase_profiler") if isinstance(layer_state, dict) else None
    if profiler is not None and getattr(profiler, "enabled", False):
        return profiler.phase(name)
    return nullcontext()


# === Quantization Helpers ===


def quantize_int8(tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Symmetric per-tensor int8 quantization.

    Uses symmetric quantization: scale = max(|x|) / 127

    Args:
        tensor: Float tensor to quantize

    Returns:
        Tuple of (quantized int8 tensor, scale float tensor)
    """
    if tensor.numel() == 0:
        return tensor.to(torch.int8), torch.ones(1, device=tensor.device, dtype=tensor.dtype)

    # Symmetric quantization: scale = max(|x|) / 127
    abs_max = tensor.abs().max()
    scale = abs_max / 127.0

    # Avoid division by zero
    if scale == 0:
        if abs_max == 0:
            return torch.zeros_like(tensor, dtype=torch.int8), torch.ones(
                1, device=tensor.device, dtype=tensor.dtype
            )
        # Preserve tiny fp16 values when only the scale division underflows.
        if tensor.dtype != torch.float64:
            return quantize_int8(tensor.float() if tensor.element_size() < 4 else tensor.double())
        scale = abs_max

    # Quantize: round(x / scale), clamp to [-127, 127]
    quantized = torch.clamp(torch.round(tensor / scale), -127, 127).to(torch.int8)

    return quantized, scale.unsqueeze(0)


def dequantize_int8(quantized: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Dequantize int8 tensor back to float.

    Args:
        quantized: Int8 tensor
        scale: Scale factor from quantization

    Returns:
        Dequantized float tensor
    """
    return quantized.to(scale.dtype) * scale


def quantize_int4(tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """Symmetric per-tensor int4 quantization with packing (2 values per byte)."""
    orig_len = tensor.numel()
    if orig_len == 0:
        return tensor.to(torch.uint8), torch.ones(1, device=tensor.device, dtype=tensor.dtype), 0

    abs_max = tensor.abs().max()
    scale = abs_max / 7.0
    if scale == 0:
        if abs_max != 0:
            if tensor.dtype != torch.float64:
                return quantize_int4(
                    tensor.float() if tensor.element_size() < 4 else tensor.double()
                )
            scale = abs_max
        else:
            # Signed zero is encoded by nibble 8, matching the ordinary q + 8 path.
            packed = torch.full(
                ((orig_len + 1) // 2,), 0x88, device=tensor.device, dtype=torch.uint8
            )
            return packed, torch.ones(1, device=tensor.device, dtype=tensor.dtype), orig_len

    q = torch.clamp(torch.round(tensor / scale), -7, 7).to(torch.int8)
    q = (q + 8).to(torch.uint8)  # map to [0, 15]

    if q.numel() % 2 == 1:
        q = torch.cat([q, torch.tensor([8], device=q.device, dtype=q.dtype)])

    q0 = q[0::2]
    q1 = q[1::2]
    packed = q0 | (q1 << 4)
    return packed, scale.unsqueeze(0), orig_len


def dequantize_int4(packed: torch.Tensor, scale: torch.Tensor, orig_len: int) -> torch.Tensor:
    """Dequantize packed int4 back to float."""
    if orig_len == 0:
        return packed.to(scale.dtype)

    low = packed & 0x0F
    high = (packed >> 4) & 0x0F
    q = torch.stack([low, high], dim=1).reshape(-1)
    q = q[:orig_len].to(torch.int8) - 8
    return q.to(scale.dtype) * scale


def _choose_index_dtype(total_elements: int) -> torch.dtype:
    """Choose optimal index dtype based on tensor size.

    Memory optimization: use int16 when possible (saves 50% vs int32).

    Args:
        total_elements: Total number of elements that could be indexed

    Returns:
        torch.int16 if total_elements < 32768, else torch.int32

    Note:
        Using 32767 (2^15 - 1) as threshold instead of 65535 (2^16 - 1)
        because PyTorch sparse tensors require signed indices for some ops.
    """
    # int16 range: -32768 to 32767, but we only use non-negative indices
    # Use conservative threshold to avoid edge cases
    if total_elements < 32768:
        return torch.int16
    elif total_elements <= 2**31:
        return torch.int32
    return torch.int64


def _resolve_basis_dtype(value: object, auto_detect_amp: bool = True) -> torch.dtype:
    """Resolve basis dtype from string or torch.dtype.

    Args:
        value: dtype specification ("float32", "float16", "auto", etc.)
        auto_detect_amp: If True and value is "auto", detect from AMP context

    Returns:
        Resolved torch.dtype

    Note on "auto" mode:
        Returns float16 if CUDA autocast is active, otherwise float32.
        This enables fused Triton kernels automatically when AMP is used.
    """
    if isinstance(value, torch.dtype):
        return value
    if not isinstance(value, str):
        return torch.float32
    name = value.lower()
    if name == "auto":
        # Auto-detect: use FP16 if AMP autocast is active (enables fused kernel)
        if auto_detect_amp and torch.cuda.is_available():
            try:
                if torch.is_autocast_enabled():
                    return torch.float16
            except:
                pass
        return torch.float32
    if name in ("float32", "fp32", "torch.float32"):
        return torch.float32
    if name in ("float16", "fp16", "half", "torch.float16"):
        return torch.float16
    if name in ("bfloat16", "bf16", "torch.bfloat16"):
        return torch.bfloat16
    raise ValueError(
        f"Unsupported basis dtype: {value}. Use 'auto', 'float32', 'float16', or 'bfloat16'."
    )


def _disable_autocast(device: torch.device):
    """Return a context manager that disables autocast on CUDA when available."""
    if device.type != "cuda":
        return nullcontext()
    amp_mod = getattr(torch, "amp", None)
    if amp_mod is not None and hasattr(amp_mod, "autocast"):
        return amp_mod.autocast(device_type="cuda", enabled=False)
    if (
        hasattr(torch, "cuda")
        and hasattr(torch.cuda, "amp")
        and hasattr(torch.cuda.amp, "autocast")
    ):
        return torch.cuda.amp.autocast(enabled=False)
    return nullcontext()


def _orthonormalize_mgs(B: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Orthonormalize columns of B using Modified Gram-Schmidt (MGS).

    MGS is O(N * R^2) and uses column-wise operations. On GPU, QR is often
    faster due to optimized fused kernels, so MGS is best treated as optional.

    Key advantages over torch.linalg.qr:
    - Avoids internal buffer allocations for Householder reflectors
    - Better cache locality for column-wise operations
    - No need for full Q matrix (we only need the first R columns)

    Args:
        B: (N, R) matrix to orthonormalize
        eps: Small constant for numerical stability

    Returns:
        Q: (N, R) orthonormal matrix
    """
    N, R = B.shape
    Q = B.clone()

    for i in range(R):
        # First, orthogonalize column i against all previous columns
        for j in range(i):
            dot = Q[:, j].dot(Q[:, i])
            Q[:, i] = Q[:, i] - dot * Q[:, j]

        # Normalize column i
        norm_i = Q[:, i].norm()
        if norm_i > eps:
            Q[:, i] = Q[:, i] / norm_i
        else:
            # Column is zero/near-zero, replace with random orthogonal to previous
            Q[:, i] = torch.randn(N, device=B.device, dtype=B.dtype)
            # Orthogonalize against previous columns
            for j in range(i):
                dot = Q[:, j].dot(Q[:, i])
                Q[:, i] = Q[:, i] - dot * Q[:, j]
            # Normalize
            Q[:, i] = Q[:, i] / Q[:, i].norm()

    return Q


# === Basis Sharing ===


class BasisPool:
    """Thread-safe global pool for sharing basis matrices across layers.

    Shares bases across layers to reduce memory when multiple layers
    share the same activation subspace (e.g., transformer encoder layers).

    Sharing modes:
    - "none": Each layer has its own basis (default, no sharing)
    - "every_2": Layers (0,1), (2,3), (4,5)... share a basis
    - "every_3": Layers (0,1,2), (3,4,5)... share a basis
    - "every_4": Layers (0,1,2,3), (4,5,6,7)... share a basis
    - "all": All layers with same in_features share one basis

    Thread-safe: Multiple layers can request/update the same basis concurrently.
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        # Storage: {(group_id, in_features, rank): basis_tensor}
        self._bases: dict = {}
        # Per-basis locks for update synchronization
        self._basis_locks: dict = {}

    def _get_basis_key(self, group_id: int, in_features: int, rank: int) -> tuple:
        """Generate unique key for basis storage."""
        return (group_id, in_features, rank)

    def get_or_create_basis(
        self, group_id: int, in_features: int, rank: int, device, dtype: torch.dtype
    ) -> torch.Tensor:
        """Get existing shared basis or create a new zero-initialized one.

        Args:
            group_id: Group index (from basis_sharing mode calculation)
            in_features: Input feature dimension
            rank: Basis rank
            device: Target device for basis tensor
            dtype: Data type for basis tensor

        Returns:
            Shared basis tensor (in_features, rank)
        """
        key = self._get_basis_key(group_id, in_features, rank)

        with self._lock:
            if key not in self._bases:
                # Create new basis
                self._bases[key] = torch.zeros(in_features, rank, dtype=dtype, device=device)
                # Create per-basis lock
                self._basis_locks[key] = threading.Lock()

            basis = self._bases[key]

            # Move to target device/dtype if needed
            if basis.device != device or basis.dtype != dtype:
                basis = basis.to(device=device, dtype=dtype)
                self._bases[key] = basis

            return basis

    def update_basis(self, group_id: int, in_features: int, rank: int, new_basis: torch.Tensor):
        """Update shared basis with new values (thread-safe).

        Args:
            group_id: Group index
            in_features: Input feature dimension
            rank: Basis rank
            new_basis: New basis tensor to copy from
        """
        key = self._get_basis_key(group_id, in_features, rank)

        # Get basis-specific lock for this update
        with self._lock:
            if key not in self._basis_locks:
                self._basis_locks[key] = threading.Lock()
            basis_lock = self._basis_locks[key]

        # Update basis under its own lock (allows concurrent updates to different bases)
        with basis_lock:
            # Outstanding forwards can retain this version for their backward.
            self._bases[key] = new_basis.detach().clone()

    def is_initialized(self, group_id: int, in_features: int, rank: int) -> bool:
        """Check if basis has been initialized (non-zero)."""
        key = self._get_basis_key(group_id, in_features, rank)
        with self._lock:
            if key not in self._bases:
                return False
            basis = self._bases[key]
            return basis.abs().sum().item() > 0

    def clear(self):
        """Clear all shared bases (for testing/debugging)."""
        with self._lock:
            self._bases.clear()
            self._basis_locks.clear()


# Global singleton instance
_global_basis_pool = BasisPool()


def get_basis_pool() -> BasisPool:
    """Get global basis pool singleton."""
    return _global_basis_pool


# === Workspace Buffer Pool ===


class WorkspacePool:
    """Thread-safe workspace buffer pool for chunked computations.

    Provides pre-allocated buffers for intermediate computations during
    forward pass, reducing memory fragmentation from repeated alloc/free.

    Key buffers managed:
    - chunk_projected: (chunk_size, N) for projection A_chunk @ B^T
    - chunk_perp: (chunk_size, N) for residual A_chunk - projected

    Without buffer reuse: ~12 allocations per forward (6 chunks × 2 buffers)
    With buffer reuse: 2 allocations total (reused across chunks)

    Memory savings: Reduces peak_reserved by avoiding allocator fragmentation.
    Measure allocator peaks for the specific model and batch to assess savings.

    Thread-safe: Each layer gets its own workspace buffers (no sharing).
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        # Storage: {layer_id: {'chunk_projected': tensor, 'chunk_perp': tensor, ...}}
        self._workspaces: dict = {}
        # Per-layer locks for workspace access
        self._workspace_locks: dict = {}
        # Allocation statistics
        self._stats = {
            "allocations_avoided": 0,
            "allocations_created": 0,
            "buffer_resizes": 0,
        }

    def _get_workspace_key(self, layer_id: int, in_features: int, chunk_size: int) -> tuple:
        """Generate unique key for workspace storage."""
        return (layer_id, in_features, chunk_size)

    def get_or_create_workspace(
        self,
        layer_id: int,
        in_features: int,
        chunk_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> dict:
        """Get existing workspace buffers or create new ones.

        Args:
            layer_id: Unique layer identifier (e.g., hash of (in_features, out_features, layer_idx))
            in_features: Input feature dimension (N)
            chunk_size: Chunk size for processing
            dtype: Data type for buffers
            device: Target device

        Returns:
            Dict with workspace buffers:
            - 'chunk_projected': (chunk_size, N) buffer
            - 'chunk_perp': (chunk_size, N) buffer
            - 'valid': bool indicating if buffers are usable
        """
        key = self._get_workspace_key(layer_id, in_features, chunk_size)

        with self._lock:
            if key not in self._workspace_locks:
                self._workspace_locks[key] = threading.Lock()
            workspace_lock = self._workspace_locks[key]

        with workspace_lock:
            needs_create = False
            if key not in self._workspaces:
                needs_create = True
            else:
                workspace = self._workspaces[key]
                # Check if existing buffers match requirements
                if (
                    workspace["chunk_projected"].device != device
                    or workspace["chunk_projected"].dtype != dtype
                    or workspace["chunk_projected"].shape[0] < chunk_size
                ):
                    needs_create = True
                    self._stats["buffer_resizes"] += 1

            if needs_create:
                # Create new workspace buffers
                self._workspaces[key] = {
                    "chunk_projected": torch.empty(
                        chunk_size, in_features, dtype=dtype, device=device
                    ),
                    "chunk_perp": torch.empty(chunk_size, in_features, dtype=dtype, device=device),
                    "valid": True,
                }
                self._stats["allocations_created"] += 1
            else:
                self._stats["allocations_avoided"] += 1

            return self._workspaces[key]

    def release_workspace(self, layer_id: int, in_features: int, chunk_size: int):
        """Mark workspace as available (optional - buffers persist anyway)."""
        # Currently no-op: buffers persist for reuse in next forward
        pass

    def clear_layer(self, layer_id: int):
        """Clear all workspaces for a specific layer."""
        with self._lock:
            keys_to_remove = [k for k in self._workspaces if k[0] == layer_id]
            for key in keys_to_remove:
                del self._workspaces[key]
                if key in self._workspace_locks:
                    del self._workspace_locks[key]

    def clear(self):
        """Clear all workspace buffers (for testing/debugging)."""
        with self._lock:
            self._workspaces.clear()
            self._workspace_locks.clear()
            self._stats = {
                "allocations_avoided": 0,
                "allocations_created": 0,
                "buffer_resizes": 0,
            }

    def get_stats(self) -> dict:
        """Get pool statistics for debugging."""
        with self._lock:
            stats = self._stats.copy()
            stats["active_workspaces"] = len(self._workspaces)
            stats["total_memory_bytes"] = sum(
                ws["chunk_projected"].numel() * ws["chunk_projected"].element_size()
                + ws["chunk_perp"].numel() * ws["chunk_perp"].element_size()
                for ws in self._workspaces.values()
            )
            return stats


# Global singleton instance
_global_workspace_pool = WorkspacePool()


def get_workspace_pool() -> WorkspacePool:
    """Get global workspace pool singleton."""
    return _global_workspace_pool


class ResidualMode(Enum):
    """Residual storage mode for STAC-R."""

    NONE = "none"  # STAC ablation: no residual stored
    TOPK = "topk"  # Keep top-k elements of a_perp
    THRESHOLD = "threshold"  # Keep elements above threshold
    BLOCK_SPARSE = "block_sparse"  # Keep block-structured residual
    ADAPTIVE_THRESHOLD = "adaptive_threshold"  # Budget-based O(n) selection via quantile


class BasisUpdateTrigger(Enum):
    """When to update the basis."""

    FIXED_INTERVAL = "fixed"  # Every K steps
    ADAPTIVE = "adaptive"  # When residual energy exceeds threshold
    HYBRID = "hybrid"  # Fixed + adaptive (update early if energy spikes)


class BasisRefreshTrigger(Enum):
    """When to refresh a frozen basis after warmup."""

    OFF = "off"  # Preserve current freeze semantics
    FIXED = "fixed"  # Refresh every K optimizer steps after warmup
    RESIDUAL_ENERGY = "residual_energy"  # Refresh when residual energy rises
    HYBRID = "hybrid"  # Fixed cadence OR residual-energy trigger


def _compute_refresh_offset(
    layer_index: Optional[int], num_layers: int, refresh_period: int
) -> int:
    """Deterministically spread fixed-refresh events across layers."""
    if layer_index is None:
        return 0
    safe_num_layers = max(1, int(num_layers))
    safe_period = max(1, int(refresh_period))
    normalized_index = int(layer_index) % safe_num_layers
    return int((normalized_index * safe_period) / safe_num_layers) % safe_period


def _is_fixed_refresh_step(
    schedule_step: int,
    warmup_steps: int,
    refresh_period: int,
    layer_index: Optional[int] = None,
    num_layers: int = 1,
) -> bool:
    """Return True when this layer should refresh on the current schedule step."""
    post_warmup_step = max(0, int(schedule_step) - int(warmup_steps))
    safe_period = max(1, int(refresh_period))
    if post_warmup_step < safe_period:
        return False
    refresh_offset = _compute_refresh_offset(layer_index, num_layers, safe_period)
    return (post_warmup_step % safe_period) == refresh_offset


@dataclass
class STACRConfig:
    """Configuration for STAC-R layer.

    STAC (ablation baseline): Set residual_mode="none"
    STAC-R (full): Set residual_mode="topk", "threshold", or "adaptive_threshold"
    Sparse-only control: Set residual_mode="sparse_only" (top-k on raw activation, no basis)

    The topk and block_sparse selectors offer fixed residual budgets; threshold
    selectors adapt the retained entries to activation magnitudes.
    """

    method: str = "stacr"

    # Low-rank basis dimension

    rank: int = 32

    # Residual handling

    residual_mode: str = "block_sparse"  # "none", "topk", "threshold", "block_sparse", "adaptive_threshold", "sparse_only"
    residual_budget: float = 0.005  # Fraction of activation size to store as residual
    residual_threshold: float = 0.1  # For threshold mode: absolute element threshold on a_perp

    # Basis update schedule
    update_trigger: str = "hybrid"  # "fixed", "adaptive", "hybrid"
    update_interval: int = 100  # K: update every K steps (for fixed/hybrid)
    energy_threshold: float = 0.05  # Update when residual_energy > threshold (for adaptive/hybrid)

    # Streaming SVD algorithm
    streaming_algorithm: str = "oja"  # "oja", "incremental_svd", "power_iteration"
    oja_learning_rate: float = 0.01  # Base learning rate for Oja's algorithm

    # Warmup: use baseline for first W steps while basis stabilizes

    warmup_steps: int = 0

    # Freeze basis after warmup: skip Oja updates after warmup_steps
    # When True, basis is initialized during warmup and then frozen (no further updates)

    freeze_basis_after_warmup: bool = True

    # Optional quantization of coefficients (4x memory reduction)

    quantize_coefficients: bool = True
    quantize_bits: int = 8  # Currently only int8 supported
    quantize_residuals: bool = True
    residual_quantize_bits: int = 4  # 8 or 4

    # Randomized SVD for large features (faster init, lower workspace)
    # Use randomized SVD when in_features > this threshold
    randomized_svd_threshold: int = 1024
    randomized_svd_oversampling: int = 4  # q = rank + oversampling
    randomized_svd_niter: int = 2  # Power iterations for accuracy

    # Adaptive Oja learning rate
    # Scale LR by (1 + residual_energy * adaptive_oja_scale)
    adaptive_oja_lr: bool = False
    adaptive_oja_scale: float = 5.0  # How much to scale LR based on residual energy
    adaptive_oja_max_lr: float = 0.1  # Cap the maximum LR

    basis_dtype: str = "auto"  # "auto" uses FP16 under AMP (enables fused kernel), else FP32
    use_random_qr_init: bool = False
    min_adaptation_steps: int = 50
    allow_unadapted_random_qr: bool = False
    backward_mode: str = "sparse"  # "dense" or "sparse" residual branch in STAC-R backward
    residual_block_size: int = 16

    # Throughput optimizations
    use_mgs: bool = False  # Use Modified Gram-Schmidt instead of QR (often slower on GPU)
    lazy_update_threshold: float = 0.01  # Skip basis update if residual_energy < threshold
    lazy_update_enabled: bool = True  # Enable lazy basis updates
    use_fused_kernel: bool = False  # Use fused projection+residual+topk when available

    # Memory optimization: Streaming top-k selection
    # When True, uses threshold-based streaming selection instead of global merge
    # This avoids the O(k) merge buffer spike at the cost of slightly approximate selection
    # Recommended when memory is tight and k is large
    streaming_topk: bool = False
    streaming_topk_sample_ratio: float = (
        0.1  # Fraction of elements to sample for threshold estimation
    )

    # Chunk size for forward pass residual computation
    # Smaller chunks = lower peak memory, but higher kernel launch overhead
    # NOTE: When B_flat <= chunk_size, no chunking occurs (full batch processed at once)
    # For small batch experiments (batch=4-8), consider using chunk_size=256 to force chunking
    chunk_size: int = 512

    # Adaptive rank selection
    rank_mode: str = "fixed"  # "fixed", "variance", "sensitivity"
    rank_variance_threshold: float = (
        0.95  # For variance mode: keep ranks until this fraction of energy captured
    )
    rank_min: int = 4  # Minimum rank (floor)
    rank_max: int = 64  # Maximum rank (ceiling)
    rank_adjust_interval: int = 100  # Optimizer/forward steps between active-rank adjustments
    rank_sensitivity_low: float = 0.03  # Shrink rank below this residual-energy band
    rank_sensitivity_high: float = 0.10  # Grow rank above this residual-energy band
    rank_step_fraction: float = 0.125  # Fractional rank step for sensitivity controller

    # Basis sharing across layers
    basis_sharing: str = "none"  # "none", "every_2", "every_3", "every_4", "all"
    basis_sharing_group: Optional[int] = None  # Explicit basis-sharing group

    # Joint Q/K compression shares a basis between attention projections
    # Reuse a basis across selected layers; quality depends on their activations.
    joint_qk: bool = False

    # CPU offload for memory savings
    # Moves saved tensors (coefficients, residuals) to CPU after forward,
    # then back to GPU during backward. Trades compute time for memory.
    cpu_offload: bool = False

    # Index compression for memory savings
    # Compresses flat 1D residual indices using CSR-like format:
    # - "none": No compression (baseline)
    # - "csr": CSR format with row pointers + column indices
    # - "delta": Delta encoding on sorted indices (good for clustered patterns)
    # Savings depend on activation shape and residual sparsity.
    index_compression: str = "none"  # "none", "csr", "delta"

    # Energy-based residual skip
    # When residual energy is below threshold, the basis already captures the activation
    # well, so we skip storing residuals entirely for that layer/step.
    # This saves memory when compression is "good enough" without residual correction.
    # Different from lazy_update which skips BASIS updates - this skips RESIDUAL storage.
    skip_residual_enabled: bool = False
    skip_residual_energy_threshold: float = 0.05  # Skip residual if energy < 5%

    # Buffer reuse for chunk workspace
    # Pre-allocates workspace buffers for chunked computation, avoiding repeated
    # allocation/deallocation that causes CUDA memory fragmentation.
    # Reuse chunk workspaces to reduce repeated allocations.
    # Safe to enable - falls back to standard allocation if buffers unavailable.
    buffer_reuse: bool = False

    # Adaptive per-layer rank allocation
    # Allocate different ranks to different layers based on importance:
    # - "none": All layers use the same rank (default)
    # - "heuristic": Early=1.5x, Middle=1.0x, Late=0.5x (based on layer importance)
    adaptive_layer_rank: str = "none"  # "none" or "heuristic"

    # Dynamic residual schedule
    # Start with higher residual budget and decay to target over training:
    # - "none": Fixed budget throughout training (default)
    # - "linear": Linear decay from start to end
    # - "cosine": Cosine annealing from start to end
    residual_schedule: str = "none"  # "none", "linear", "cosine"
    residual_start: float = 0.05  # Starting residual budget (5%)
    residual_end: float = 0.001  # Ending residual budget (0.1%)

    # Adaptive Turbo: switch residual mode after basis stabilizes
    # Full STAC-R early (accurate basis formation + residual correction),
    # then drop to faster mode once the basis has converged.
    # - "full": keep original residual_mode for entire training (default, no change)
    # - "none": switch to low-rank-only (no residual) after residual_warmup_steps
    # - "threshold": switch to energy-thresholded residual after residual_warmup_steps
    post_warmup_residual_mode: str = "full"  # "full", "none", "threshold"
    residual_warmup_steps: int = 0  # 0 = disabled (use residual_mode for all steps)
    post_warmup_residual_threshold: float = (
        0.1  # For threshold mode: skip residual if energy < this
    )
    refresh_trigger: str = "off"  # "off", "fixed", "residual_energy", "hybrid"
    refresh_period: int = (
        1000  # Optimizer-step cadence after warmup when refresh_trigger=fixed/hybrid
    )
    refresh_energy_threshold: float = 0.1  # Residual-energy trigger after warmup
    cache_basis_per_optimizer_step: bool = False  # Reuse one basis across grad-accum microbatches
    oja_update_per_optimizer_step: bool = False  # Schedule Oja/refresh using optimizer steps
    enable_low_rank_only_fast_path: bool = True  # allow dedicated low-rank-only route

    def __post_init__(self):
        """Reject unsupported settings before changing a model."""
        choices = {
            "residual_mode": {
                "none",
                "topk",
                "threshold",
                "block_sparse",
                "adaptive_threshold",
                "sparse_only",
            },
            "update_trigger": {"fixed", "adaptive", "hybrid"},
            "streaming_algorithm": {"oja"},
            "backward_mode": {"dense", "sparse"},
            "rank_mode": {"fixed", "variance", "sensitivity"},
            "basis_sharing": {"none", "every_2", "every_3", "every_4", "all"},
            "index_compression": {"none", "csr", "delta"},
            "adaptive_layer_rank": {"none", "heuristic"},
            "residual_schedule": {"none", "linear", "cosine"},
            "post_warmup_residual_mode": {"full", "none", "threshold"},
            "refresh_trigger": {"off", "fixed", "residual_energy", "hybrid"},
        }
        for name, accepted in choices.items():
            if getattr(self, name) not in accepted:
                raise ValueError(f"{name} must be one of {sorted(accepted)}")
        for name in (
            "rank",
            "update_interval",
            "chunk_size",
            "residual_block_size",
            "rank_min",
            "rank_max",
            "rank_adjust_interval",
            "refresh_period",
            "randomized_svd_threshold",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or int(value) != value or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in (
            "warmup_steps",
            "residual_warmup_steps",
            "min_adaptation_steps",
            "randomized_svd_oversampling",
            "randomized_svd_niter",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or int(value) != value or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        for name in (
            "residual_budget",
            "residual_start",
            "residual_end",
            "rank_variance_threshold",
            "streaming_topk_sample_ratio",
            "skip_residual_energy_threshold",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{name} must be between 0 and 1")
        if self.quantize_bits != 8 or self.residual_quantize_bits not in (4, 8):
            raise ValueError(
                "Coefficient quantization requires 8 bits; residual quantization requires 4 or 8"
            )
        if self.rank_min > self.rank_max:
            raise ValueError("rank_min cannot exceed rank_max")
        if self.rank_sensitivity_low > self.rank_sensitivity_high:
            raise ValueError("rank_sensitivity_low cannot exceed rank_sensitivity_high")
        for name in (
            "oja_learning_rate",
            "energy_threshold",
            "residual_threshold",
            "adaptive_oja_scale",
            "adaptive_oja_max_lr",
            "lazy_update_threshold",
            "refresh_energy_threshold",
            "rank_sensitivity_low",
            "rank_sensitivity_high",
            "rank_step_fraction",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        _resolve_basis_dtype(self.basis_dtype)
        if self.basis_sharing != "none" and self.rank_mode != "fixed":
            raise ValueError("Basis sharing requires rank_mode='fixed'")

    def get_layer_rank(self, layer_idx: int, base_rank: int, num_layers: int = 12) -> int:
        """Get rank for a specific layer based on adaptive strategy (A1).

        Args:
            layer_idx: Index of the layer (0-indexed)
            base_rank: Base rank configured in config.rank
            num_layers: Total number of layers (default 12 for BERT)

        Returns:
            Effective rank for this layer
        """
        if self.adaptive_layer_rank == "none":
            return base_rank
        elif self.adaptive_layer_rank == "heuristic":
            # Early layers (first 1/3): 1.0x rank (preserve input information)
            # Middle layers (middle 1/3): 0.7x rank (balanced)
            # Late layers (last 1/3): 0.4x rank (more compressible, task-specific)
            # Average multiplier = 0.7, giving ~30% storage savings vs uniform rank
            if layer_idx < num_layers // 3:
                multiplier = 1.0
            elif layer_idx < 2 * num_layers // 3:
                multiplier = 0.7
            else:
                multiplier = 0.4
            return max(8, int(base_rank * multiplier))  # min rank = 8
        return base_rank

    def get_residual_budget(self, step: int, total_steps: int) -> float:
        """Get residual budget for current step based on schedule (A2).

        Args:
            step: Current training step (0-indexed)
            total_steps: Total number of training steps

        Returns:
            Residual budget for this step
        """
        if self.residual_schedule == "none":
            return self.residual_budget

        progress = step / max(1, total_steps)

        if self.residual_schedule == "linear":
            return self.residual_start - (self.residual_start - self.residual_end) * progress
        elif self.residual_schedule == "cosine":
            import math

            return self.residual_end + 0.5 * (self.residual_start - self.residual_end) * (
                1 + math.cos(math.pi * progress)
            )
        return self.residual_budget

    def to_dict(self) -> dict:
        """Convert to dict for CompressedLinear compatibility."""
        return {
            "method": self.method,
            "rank": self.rank,
            "residual_mode": self.residual_mode,
            "residual_budget": self.residual_budget,
            "residual_threshold": self.residual_threshold,
            "update_trigger": self.update_trigger,
            "update_interval": self.update_interval,
            "energy_threshold": self.energy_threshold,
            "streaming_algorithm": self.streaming_algorithm,
            "oja_learning_rate": self.oja_learning_rate,
            "warmup_steps": self.warmup_steps,
            "freeze_basis_after_warmup": self.freeze_basis_after_warmup,
            "quantize_coefficients": self.quantize_coefficients,
            "quantize_bits": self.quantize_bits,
            "quantize_residuals": self.quantize_residuals,
            "residual_quantize_bits": self.residual_quantize_bits,
            "randomized_svd_threshold": self.randomized_svd_threshold,
            "randomized_svd_oversampling": self.randomized_svd_oversampling,
            "randomized_svd_niter": self.randomized_svd_niter,
            "adaptive_oja_lr": self.adaptive_oja_lr,
            "adaptive_oja_scale": self.adaptive_oja_scale,
            "adaptive_oja_max_lr": self.adaptive_oja_max_lr,
            "basis_dtype": self.basis_dtype,
            "use_random_qr_init": self.use_random_qr_init,
            "min_adaptation_steps": self.min_adaptation_steps,
            "allow_unadapted_random_qr": self.allow_unadapted_random_qr,
            "backward_mode": self.backward_mode,
            "residual_block_size": self.residual_block_size,
            "use_mgs": self.use_mgs,
            "lazy_update_threshold": self.lazy_update_threshold,
            "lazy_update_enabled": self.lazy_update_enabled,
            "use_fused_kernel": self.use_fused_kernel,
            "streaming_topk": self.streaming_topk,
            "streaming_topk_sample_ratio": self.streaming_topk_sample_ratio,
            "rank_mode": self.rank_mode,
            "rank_variance_threshold": self.rank_variance_threshold,
            "rank_min": self.rank_min,
            "rank_max": self.rank_max,
            "rank_adjust_interval": self.rank_adjust_interval,
            "rank_sensitivity_low": self.rank_sensitivity_low,
            "rank_sensitivity_high": self.rank_sensitivity_high,
            "rank_step_fraction": self.rank_step_fraction,
            "basis_sharing": self.basis_sharing,
            "basis_sharing_group": self.basis_sharing_group,
            "cpu_offload": self.cpu_offload,
            "index_compression": self.index_compression,
            "skip_residual_enabled": self.skip_residual_enabled,
            "skip_residual_energy_threshold": self.skip_residual_energy_threshold,
            "buffer_reuse": self.buffer_reuse,
            # Adaptive per-layer rank
            "adaptive_layer_rank": self.adaptive_layer_rank,
            # Dynamic residual schedule
            "residual_schedule": self.residual_schedule,
            "residual_start": self.residual_start,
            "residual_end": self.residual_end,
            # Adaptive Turbo
            "post_warmup_residual_mode": self.post_warmup_residual_mode,
            "residual_warmup_steps": self.residual_warmup_steps,
            "post_warmup_residual_threshold": self.post_warmup_residual_threshold,
            "refresh_trigger": self.refresh_trigger,
            "refresh_period": self.refresh_period,
            "refresh_energy_threshold": self.refresh_energy_threshold,
            "cache_basis_per_optimizer_step": self.cache_basis_per_optimizer_step,
            "oja_update_per_optimizer_step": self.oja_update_per_optimizer_step,
            "enable_low_rank_only_fast_path": self.enable_low_rank_only_fast_path,
        }


def profile_layer_spectrum(
    activations: torch.Tensor, threshold: float = 0.95, return_full: bool = False
) -> dict:
    """Profile activation spectrum to determine adaptive rank.

    Uses SVD to compute singular value spectrum and find the rank
    that captures `threshold` fraction of total variance/energy.

    Args:
        activations: (B, N) input activations to profile
        threshold: Fraction of variance to capture (default 0.95 = 95%)
        return_full: If True, return full spectrum for analysis

    Returns:
        dict with:
            - 'adaptive_rank': int, rank needed to capture threshold variance
            - 'total_energy': float, total spectral energy (sum of sigma^2)
            - 'energy_at_rank': float, energy captured at adaptive_rank
            - 'spectrum': (optional) full singular value array if return_full=True
    """
    with torch.no_grad():
        # Flatten to 2D if needed
        if activations.dim() > 2:
            activations = activations.reshape(-1, activations.shape[-1])

        # Edge case: empty tensor
        if activations.numel() == 0:
            result = {"adaptive_rank": 1, "total_energy": 0.0, "energy_at_rank": 0.0}
            if return_full:
                result["spectrum"] = []
            return result

        # Compute singular values
        S = torch.linalg.svdvals(activations.float())

        # Compute energy (squared singular values)
        energy = S**2
        total_energy = energy.sum().item()

        if total_energy < 1e-12:
            result = {"adaptive_rank": 1, "total_energy": total_energy, "energy_at_rank": 0.0}
            if return_full:
                result["spectrum"] = S.cpu().numpy() if hasattr(S, "numpy") else S.cpu().tolist()
            return result

        # Find rank that captures threshold fraction
        cumulative = torch.cumsum(energy, dim=0) / total_energy
        adaptive_rank = (cumulative < threshold).sum().item() + 1
        adaptive_rank = max(1, min(adaptive_rank, S.numel()))

        energy_at_rank = (
            cumulative[adaptive_rank - 1].item() if adaptive_rank <= len(cumulative) else 1.0
        )

        result = {
            "adaptive_rank": adaptive_rank,
            "total_energy": total_energy,
            "energy_at_rank": energy_at_rank,
        }
        if return_full:
            result["spectrum"] = S.cpu().numpy() if hasattr(S, "numpy") else S.cpu().tolist()

        return result


class STACRFunction(torch.autograd.Function):
    """
    Custom autograd function for STAC-R.

    Forward:
        1. Compute exact output: Y = A @ W^T + b
        2. If basis uninitialized: init from first batch
        3. Project activation: c = B^T @ a
        4. Compute residual: a_perp = a - B @ c
        5. If residual mode enabled: store sparse residual
        6. Save: coefficients c, (sparse residual), weight, bias

    Backward:
        1. Factor-native: grad_W = (G^T @ c) @ B^T + sparse_residual_contrib (no dense reconstruction)
        2. Compute input gradient: grad_input = G @ W (exact)
        3. Residual handled via sparse COO tensor for efficiency
        4. Basis update triggered based on residual energy (in forward pass)
    """

    @staticmethod
    def forward(ctx, input_tensor, weight, bias, basis, config, step_counter, layer_state):
        """
        Forward pass with subspace projection.

        Args:
            input_tensor: (B, *, N) input activations
            weight: (out_features, N) weight matrix
            bias: (out_features,) or None
            basis: (N, R) orthonormal basis matrix
            config: STACRConfig dict
            step_counter: Current training step (for update scheduling)
            layer_state: Mutable dict for tracking layer statistics
        """
        rank = config["rank"]
        residual_mode = config["residual_mode"]

        # Adaptive Turbo: switch residual handling after basis has stabilized.
        # `none` becomes pure low-rank after warmup; `threshold` keeps the
        # original residual mode but activates the existing energy-based
        # residual skip gate with a post-warmup threshold.
        post_warmup_mode = config.get("post_warmup_residual_mode", "full")
        residual_warmup = config.get("residual_warmup_steps", 0)
        schedule_step = int(layer_state.get("_schedule_step", 0))
        turbo_skip_residual_enabled = False
        turbo_skip_residual_threshold = float(config.get("post_warmup_residual_threshold", 0.1))
        if post_warmup_mode != "full" and residual_warmup > 0:
            if schedule_step >= residual_warmup:
                original_mode = residual_mode
                if post_warmup_mode == "none":
                    residual_mode = "none"
                elif post_warmup_mode == "threshold":
                    turbo_skip_residual_enabled = True
                # Log the switch once
                if not layer_state.get("_turbo_switched", False):
                    layer_state["_turbo_switched"] = True
                    layer_state["_turbo_switch_step"] = schedule_step
                    layer_state["_turbo_original_mode"] = original_mode
                    layer_state["_turbo_mode"] = post_warmup_mode

        # Dynamic residual schedule
        residual_schedule = config.get("residual_schedule", "none")
        if residual_schedule != "none":
            total_steps = layer_state.get("total_steps")
            if total_steps is not None and total_steps > 0:
                # Compute budget based on schedule
                step = (
                    step_counter[0].item()
                    if hasattr(step_counter[0], "item")
                    else int(step_counter[0])
                )
                residual_start = float(config.get("residual_start", 0.05))
                residual_end = float(config.get("residual_end", 0.001))
                progress = step / max(1, total_steps)
                if residual_schedule == "linear":
                    residual_budget = residual_start - (residual_start - residual_end) * progress
                elif residual_schedule == "cosine":
                    import math

                    residual_budget = residual_end + 0.5 * (residual_start - residual_end) * (
                        1 + math.cos(math.pi * progress)
                    )
                else:
                    residual_budget = float(config["residual_budget"])
            else:
                residual_budget = float(config["residual_budget"])
        else:
            residual_budget = float(config["residual_budget"])
        residual_budget = max(0.0, min(1.0, residual_budget))
        # Store current budget in layer_state for logging
        layer_state["current_residual_budget"] = residual_budget

        warmup_steps = config["warmup_steps"]

        in_features = weight.shape[1]
        leading_shape = input_tensor.shape[:-1]
        input_2d = input_tensor.reshape(-1, in_features)  # (B_flat, N)
        B_flat = input_2d.shape[0]

        # Compute exact forward (no approximation here)
        # Cast weight to input dtype if needed (for FP16/BF16 inputs with FP32 weights)
        weight_compute = weight
        if weight.dtype != input_2d.dtype:
            weight_compute = weight.to(input_2d.dtype)
        output_2d = input_2d.mm(weight_compute.t())
        if bias is not None:
            bias_compute = bias if bias.dtype == output_2d.dtype else bias.to(output_2d.dtype)
            output_2d = output_2d + bias_compute.unsqueeze(0)
        output = output_2d.reshape(*leading_shape, weight.shape[0])

        # === SPARSE-ONLY mode: top-k on raw activation, no basis projection ===
        if residual_mode == "sparse_only":
            total_elements = B_flat * in_features
            k = int(total_elements * residual_budget)
            residual_sparse = None
            residual_indices = None

            if k > 0:
                k = min(k, total_elements)
                target_chunk_size = config.get("chunk_size", 512)
                chunk_size = min(target_chunk_size, B_flat)
                all_values = []
                all_indices = []
                per_chunk_k = max(1, (k * chunk_size + B_flat - 1) // B_flat)

                for chunk_start in range(0, B_flat, chunk_size):
                    chunk_end = min(chunk_start + chunk_size, B_flat)
                    chunk_input = input_2d[chunk_start:chunk_end]
                    chunk_flat = chunk_input.reshape(-1)
                    actual_k = min(per_chunk_k, chunk_flat.numel())
                    _, local_indices = torch.topk(chunk_flat.abs(), actual_k)
                    local_values = chunk_flat[local_indices]
                    global_offset = chunk_start * in_features
                    global_indices = local_indices + global_offset
                    all_values.append(local_values)
                    all_indices.append(global_indices)
                    del chunk_flat

                merged_values = torch.cat(all_values)
                merged_indices = torch.cat(all_indices)
                index_dtype = _choose_index_dtype(total_elements)
                if merged_values.numel() > k:
                    _, top_indices = torch.topk(merged_values.abs(), k)
                    residual_sparse = merged_values[top_indices]
                    residual_indices = merged_indices[top_indices].to(index_dtype)
                else:
                    residual_sparse = merged_values
                    residual_indices = merged_indices.to(index_dtype)

            # Quantize residuals if configured
            quantize_residuals = bool(config.get("quantize_residuals", False))
            residual_bits = int(config.get("residual_quantize_bits", 8))
            residual_scale = None
            residual_orig_len = None
            residual_sparse_to_save = residual_sparse
            if quantize_residuals and residual_sparse is not None:
                if residual_bits == 8:
                    residual_sparse_to_save, residual_scale = quantize_int8(residual_sparse)
                elif residual_bits == 4:
                    residual_sparse_to_save, residual_scale, residual_orig_len = quantize_int4(
                        residual_sparse
                    )

            # Save: [weight, bias?, sparse_values?, sparse_indices?, residual_scale?]
            tensors_to_save = [weight]
            if bias is not None:
                tensors_to_save.append(bias)
            if residual_sparse is not None:
                tensors_to_save.append(residual_sparse_to_save)
                tensors_to_save.append(residual_indices)
                if residual_scale is not None:
                    tensors_to_save.append(residual_scale)

            ctx.save_for_backward(*tensors_to_save)
            ctx.stacr_mode = "sparse_only"
            ctx.leading_shape = leading_shape
            ctx.has_bias = bias is not None
            ctx.has_residual = residual_sparse is not None
            ctx.residual_shape = (B_flat, in_features)
            ctx.quantize_residuals = quantize_residuals
            ctx.residual_bits = residual_bits
            ctx.residual_orig_len = residual_orig_len
            ctx.cpu_offload = False

            # Update layer state
            skip_layer_state = False
            try:
                from .adaptive_checkpointing import get_global_recomputation_guard

                skip_layer_state = get_global_recomputation_guard().is_recomputing
            except Exception:
                skip_layer_state = False
            if not skip_layer_state:
                layer_state["last_residual_energy"] = 1.0
                layer_state["step"] = schedule_step

            return output

        # During warmup or if basis not ready, store full activation (baseline mode)
        if schedule_step < warmup_steps or basis is None or basis.numel() == 0:
            tensors_to_save = [input_2d, weight]
            if bias is not None:
                tensors_to_save.append(bias)
            ctx.save_for_backward(*tensors_to_save)
            ctx.stacr_mode = "baseline"
            ctx.leading_shape = leading_shape
            ctx.has_bias = bias is not None
            return output

        # === STAC-R Core: Subspace Projection ===
        basis_compute = basis
        if basis_compute.dtype != input_2d.dtype:
            basis_compute = basis_compute.to(input_2d.dtype)
        # Use actual basis rank (may differ from config.rank under adaptive per-layer rank)
        rank = basis_compute.shape[1]

        # Handle residual based on mode
        coefficients = None
        residual_sparse = None
        residual_indices = None
        residual_energy = 0.0
        fused_used = False
        triton_ok = False
        skipped_residual = False  # Track if residual was skipped due to low energy
        log_fused = os.getenv("STACR_FUSED_LOG") == "1"

        if residual_mode == "topk":
            use_fused = bool(config.get("use_fused_kernel", False))
            if use_fused and residual_budget > 0:
                try:
                    from .triton_kernels import stacr_forward_fused, triton_available

                    triton_ok = triton_available()
                    coefficients, topk_vals, topk_idx, fused_meta = stacr_forward_fused(
                        input_2d,
                        basis_compute,
                        residual_budget,
                        topk_mode="global",
                        return_metadata=True,
                    )
                    fused_used = True
                    layer_state["b4_fused_kernel_route"] = fused_meta.get(
                        "route", "pytorch_chunked"
                    )
                    layer_state["b4_fused_kernel_route_counts"] = fused_meta.get("route_counts", {})
                    layer_state["b4_numerical_parity_error"] = fused_meta.get("parity_error")
                except Exception:
                    fused_used = False
                    layer_state["b4_fused_kernel_route"] = "pytorch_fallback"
                    layer_state["b4_fused_kernel_route_counts"] = {"pytorch_fallback": 1}
                    layer_state["b4_numerical_parity_error"] = None

            if log_fused and use_fused:
                global _FUSED_LOGGED, _FUSED_FALLBACK_LOGGED
                if fused_used and not _FUSED_LOGGED:
                    print(f"[STACR] Triton fused kernel ENGAGED (shape={tuple(input_2d.shape)})")
                    _FUSED_LOGGED = True
                elif not fused_used and not _FUSED_FALLBACK_LOGGED:
                    print(
                        "[STACR] Fused kernel FALLBACK "
                        f"(triton_ok={triton_ok}, input={input_2d.dtype}, basis={basis_compute.dtype}, "
                        f"input_cuda={input_2d.is_cuda}, basis_cuda={basis_compute.is_cuda})"
                    )
                    _FUSED_FALLBACK_LOGGED = True

            if fused_used:
                input_norm_sq = (input_2d * input_2d).sum()
                coeff_norm_sq = (coefficients * coefficients).sum()
                residual_norm_sq = torch.clamp(input_norm_sq - coeff_norm_sq, min=0.0)
                residual_energy = (
                    torch.sqrt(residual_norm_sq) / (torch.sqrt(input_norm_sq) + 1e-8)
                ).item()
                # Energy-based residual skip should also apply to fused path
                skip_residual_enabled = turbo_skip_residual_enabled or config.get(
                    "skip_residual_enabled", False
                )
                skip_residual_threshold = (
                    turbo_skip_residual_threshold
                    if turbo_skip_residual_enabled
                    else config.get("skip_residual_energy_threshold", 0.05)
                )
                if skip_residual_enabled and residual_energy < skip_residual_threshold:
                    residual_sparse = None
                    residual_indices = None
                    skipped_residual = True
                else:
                    residual_sparse = topk_vals.reshape(-1)
                    residual_indices = topk_idx.reshape(-1)

        if coefficients is None:
            # Project activation to coefficients: c = a @ B  (B_flat, N) @ (N, R) -> (B_flat, R)
            coefficients = input_2d.mm(basis_compute)

        if residual_mode == "none":
            # STAC mode: no residual stored, no dense a_perp needed
            # Estimate residual energy without materializing a_perp:
            # ||a_perp||^2 = ||a||^2 - ||c||^2 (by Pythagoras since B is orthonormal)
            input_norm_sq = (input_2d * input_2d).sum()
            coeff_norm_sq = (coefficients * coefficients).sum()
            residual_norm_sq = torch.clamp(input_norm_sq - coeff_norm_sq, min=0.0)
            residual_energy = (
                torch.sqrt(residual_norm_sq) / (torch.sqrt(input_norm_sq) + 1e-8)
            ).item()

        elif residual_mode in ("topk", "threshold"):
            if residual_mode == "topk" and fused_used:
                pass
            else:
                # Use chunked processing to avoid materializing full BxN a_perp tensor
                # This reduces peak memory from O(BxN) to O(chunk_sizexN)
                total_elements = B_flat * in_features
                k = 0
                if residual_mode == "topk":
                    k = int(total_elements * residual_budget)
                    if k <= 0:
                        residual_sparse = None
                        residual_indices = None
                        k = 0
                    else:
                        k = min(k, total_elements)

                # Chunk size: balance memory vs overhead
                # Use smaller chunks to reduce forward pass peak memory
                # 512 rows * 3072 features * 4 bytes = 6.3 MB per chunk
                # NOTE: When B_flat <= chunk_size, no chunking occurs
                target_chunk_size = config.get("chunk_size", 512)
                chunk_size = min(target_chunk_size, B_flat)

                # Buffer reuse - get workspace buffers to avoid repeated allocations
                buffer_reuse = config.get("buffer_reuse", False)
                workspace = None
                if buffer_reuse:
                    layer_id = config.get("_layer_id")
                    if layer_id is not None:
                        ws_pool = get_workspace_pool()
                        workspace = ws_pool.get_or_create_workspace(
                            layer_id=layer_id,
                            in_features=in_features,
                            chunk_size=chunk_size,
                            dtype=input_2d.dtype,
                            device=input_2d.device,
                        )

                # Estimate residual energy using Pythagorean identity (no dense computation)
                input_norm_sq = (input_2d * input_2d).sum()
                coeff_norm_sq = (coefficients * coefficients).sum()
                residual_norm_sq = torch.clamp(input_norm_sq - coeff_norm_sq, min=0.0)
                residual_energy = (
                    torch.sqrt(residual_norm_sq) / (torch.sqrt(input_norm_sq) + 1e-8)
                ).item()

                # Energy-based residual skip
                # If residual energy is low, basis captures activations well - skip residual storage
                skip_residual_enabled = turbo_skip_residual_enabled or config.get(
                    "skip_residual_enabled", False
                )
                skip_residual_threshold = (
                    turbo_skip_residual_threshold
                    if turbo_skip_residual_enabled
                    else config.get("skip_residual_energy_threshold", 0.05)
                )
                if skip_residual_enabled and residual_energy < skip_residual_threshold:
                    # Skip residual storage - basis is good enough
                    residual_sparse = None
                    residual_indices = None
                    k = 0
                    skipped_residual = True

                if residual_mode == "topk" and k > 0:
                    # Check if streaming top-k is enabled (reduces merge buffer spike)
                    use_streaming = config.get("streaming_topk", False)

                    if use_streaming:
                        # Streaming top-k: Use threshold-based selection to avoid global merge
                        # Two-pass approach with bounded memory:
                        # Pass 1: Sample residual values to estimate threshold
                        # Pass 2: Select elements above threshold

                        sample_ratio = config.get("streaming_topk_sample_ratio", 0.1)
                        max_samples = min(100000, max(10000, int(total_elements * sample_ratio)))
                        samples_per_chunk = max(1, (max_samples * chunk_size) // B_flat)

                        # Pass 1: Collect samples for threshold estimation
                        all_samples = []
                        for chunk_start in range(0, B_flat, chunk_size):
                            chunk_end = min(chunk_start + chunk_size, B_flat)
                            actual_chunk_size = chunk_end - chunk_start
                            chunk_input = input_2d[chunk_start:chunk_end]
                            chunk_coeff = coefficients[chunk_start:chunk_end]

                            # Use workspace buffers if available for in-place ops
                            if workspace is not None and workspace.get("valid", False):
                                chunk_projected = workspace["chunk_projected"][:actual_chunk_size]
                                chunk_perp = workspace["chunk_perp"][:actual_chunk_size]
                                torch.mm(chunk_coeff, basis_compute.t(), out=chunk_projected)
                                torch.sub(chunk_input, chunk_projected, out=chunk_perp)
                            else:
                                chunk_projected = chunk_coeff.mm(basis_compute.t())
                                chunk_perp = chunk_input - chunk_projected
                                del chunk_projected

                            chunk_flat = chunk_perp.reshape(-1)
                            chunk_abs = chunk_flat.abs()

                            # Random sampling
                            n_samples = min(samples_per_chunk, chunk_flat.numel())
                            if n_samples > 0:
                                sample_indices = torch.randint(
                                    0, chunk_flat.numel(), (n_samples,), device=chunk_flat.device
                                )
                                all_samples.append(chunk_abs[sample_indices])

                            if workspace is None or not workspace.get("valid", False):
                                del chunk_perp
                            del chunk_flat, chunk_abs

                        # Compute threshold from samples
                        samples = torch.cat(all_samples)
                        keep_ratio = k / total_elements
                        quantile_value = 1.0 - keep_ratio
                        if samples.dtype in (torch.float16, torch.bfloat16):
                            threshold_val = torch.quantile(samples.float(), quantile_value).to(
                                samples.dtype
                            )
                        else:
                            threshold_val = torch.quantile(samples, quantile_value)
                        del samples, all_samples

                        # Pass 2: Select elements above threshold
                        all_values = []
                        all_indices = []
                        for chunk_start in range(0, B_flat, chunk_size):
                            chunk_end = min(chunk_start + chunk_size, B_flat)
                            actual_chunk_size = chunk_end - chunk_start
                            chunk_input = input_2d[chunk_start:chunk_end]
                            chunk_coeff = coefficients[chunk_start:chunk_end]

                            # Use workspace buffers if available for in-place ops
                            if workspace is not None and workspace.get("valid", False):
                                chunk_projected = workspace["chunk_projected"][:actual_chunk_size]
                                chunk_perp = workspace["chunk_perp"][:actual_chunk_size]
                                torch.mm(chunk_coeff, basis_compute.t(), out=chunk_projected)
                                torch.sub(chunk_input, chunk_projected, out=chunk_perp)
                            else:
                                chunk_projected = chunk_coeff.mm(basis_compute.t())
                                chunk_perp = chunk_input - chunk_projected
                                del chunk_projected

                            chunk_flat = chunk_perp.reshape(-1)
                            mask = chunk_flat.abs() > threshold_val
                            local_indices = mask.nonzero(as_tuple=False).reshape(-1)

                            if local_indices.numel() > 0:
                                local_values = chunk_flat[local_indices]
                                global_offset = chunk_start * in_features
                                global_indices = local_indices + global_offset
                                all_values.append(local_values)
                                all_indices.append(global_indices)

                            if workspace is None or not workspace.get("valid", False):
                                del chunk_perp
                            del chunk_flat

                        index_dtype = _choose_index_dtype(total_elements)
                        if all_values:
                            residual_sparse = torch.cat(all_values)
                            residual_indices = torch.cat(all_indices).to(index_dtype)
                        else:
                            residual_sparse = torch.tensor(
                                [], device=input_2d.device, dtype=input_2d.dtype
                            )
                            residual_indices = torch.tensor(
                                [], device=input_2d.device, dtype=index_dtype
                            )

                    else:
                        # Standard chunked top-k: collect local top-k, then global merge
                        all_values = []
                        all_indices = []
                        # Per-chunk k: proportional to chunk size
                        per_chunk_k = max(1, (k * chunk_size + B_flat - 1) // B_flat)

                        for chunk_start in range(0, B_flat, chunk_size):
                            chunk_end = min(chunk_start + chunk_size, B_flat)
                            actual_chunk_size = chunk_end - chunk_start
                            chunk_input = input_2d[chunk_start:chunk_end]  # (chunk, N)
                            chunk_coeff = coefficients[chunk_start:chunk_end]  # (chunk, R)

                            # Compute chunk residual (only chunk_size x N, not full BxN)
                            # Use workspace buffers if available for in-place ops
                            if workspace is not None and workspace.get("valid", False):
                                chunk_projected = workspace["chunk_projected"][:actual_chunk_size]
                                chunk_perp = workspace["chunk_perp"][:actual_chunk_size]
                                torch.mm(chunk_coeff, basis_compute.t(), out=chunk_projected)
                                torch.sub(chunk_input, chunk_projected, out=chunk_perp)
                            else:
                                chunk_projected = chunk_coeff.mm(basis_compute.t())  # (chunk, N)
                                chunk_perp = chunk_input - chunk_projected  # (chunk, N)
                                del chunk_projected

                            # Find local top-k in this chunk
                            chunk_flat = chunk_perp.reshape(-1)
                            actual_k = min(per_chunk_k, chunk_flat.numel())
                            _, local_indices = torch.topk(chunk_flat.abs(), actual_k)
                            local_values = chunk_flat[local_indices]

                            # Convert to global indices
                            global_offset = chunk_start * in_features
                            global_indices = local_indices + global_offset

                            all_values.append(local_values)
                            all_indices.append(global_indices)
                            if workspace is None or not workspace.get("valid", False):
                                del chunk_perp
                            del chunk_flat

                        # Merge all chunks and find global top-k
                        merged_values = torch.cat(all_values)
                        merged_indices = torch.cat(all_indices)

                        # Choose optimal index dtype (int16 if small, int32 otherwise)
                        index_dtype = _choose_index_dtype(total_elements)
                        if merged_values.numel() > k:
                            _, top_indices = torch.topk(merged_values.abs(), k)
                            residual_sparse = merged_values[top_indices]
                            residual_indices = merged_indices[top_indices].to(index_dtype)
                        else:
                            residual_sparse = merged_values
                            residual_indices = merged_indices.to(index_dtype)

                elif residual_mode == "threshold":
                    # Chunked threshold: process chunks, collect all above threshold
                    threshold_val = config["residual_threshold"]
                    all_values = []
                    all_indices = []

                    for chunk_start in range(0, B_flat, chunk_size):
                        chunk_end = min(chunk_start + chunk_size, B_flat)
                        actual_chunk_size = chunk_end - chunk_start
                        chunk_input = input_2d[chunk_start:chunk_end]
                        chunk_coeff = coefficients[chunk_start:chunk_end]

                        # Compute chunk residual
                        # Use workspace buffers if available for in-place ops
                        if workspace is not None and workspace.get("valid", False):
                            chunk_projected = workspace["chunk_projected"][:actual_chunk_size]
                            chunk_perp = workspace["chunk_perp"][:actual_chunk_size]
                            torch.mm(chunk_coeff, basis_compute.t(), out=chunk_projected)
                            torch.sub(chunk_input, chunk_projected, out=chunk_perp)
                        else:
                            chunk_projected = chunk_coeff.mm(basis_compute.t())
                            chunk_perp = chunk_input - chunk_projected
                            del chunk_projected

                        # Find elements above threshold
                        chunk_flat = chunk_perp.reshape(-1)
                        mask = chunk_flat.abs() > threshold_val
                        local_indices = mask.nonzero(as_tuple=False).reshape(-1)

                        if local_indices.numel() > 0:
                            local_values = chunk_flat[local_indices]
                            global_offset = chunk_start * in_features
                            global_indices = local_indices + global_offset
                            all_values.append(local_values)
                            all_indices.append(global_indices)

                        if workspace is None or not workspace.get("valid", False):
                            del chunk_perp
                        del chunk_flat

                    # Choose optimal index dtype (int16 if small, int32 otherwise)
                    index_dtype = _choose_index_dtype(total_elements)
                    if all_values:
                        residual_sparse = torch.cat(all_values)
                        residual_indices = torch.cat(all_indices).to(index_dtype)
                    else:
                        residual_sparse = torch.tensor(
                            [], device=input_2d.device, dtype=input_2d.dtype
                        )
                        residual_indices = torch.tensor(
                            [], device=input_2d.device, dtype=index_dtype
                        )
                # If k == 0, residual_sparse/residual_indices are None (no residual stored).

        elif residual_mode == "adaptive_threshold":
            # Budget-based O(n) selection via quantile estimation
            # Uses torch.quantile() to find threshold that keeps ~residual_budget fraction
            # Complexity: O(n) for quantile + O(n) for mask, vs O(n log k) for topk
            total_elements = B_flat * in_features
            k = int(total_elements * residual_budget)
            if k <= 0:
                residual_sparse = None
                residual_indices = None
            else:
                k = min(k, total_elements)

                # Estimate residual energy using Pythagorean identity
                input_norm_sq = (input_2d * input_2d).sum()
                coeff_norm_sq = (coefficients * coefficients).sum()
                residual_norm_sq = torch.clamp(input_norm_sq - coeff_norm_sq, min=0.0)
                residual_energy = (
                    torch.sqrt(residual_norm_sq) / (torch.sqrt(input_norm_sq) + 1e-8)
                ).item()

                # Energy-based residual skip
                skip_residual_enabled = turbo_skip_residual_enabled or config.get(
                    "skip_residual_enabled", False
                )
                skip_residual_threshold = (
                    turbo_skip_residual_threshold
                    if turbo_skip_residual_enabled
                    else config.get("skip_residual_energy_threshold", 0.05)
                )
                if skip_residual_enabled and residual_energy < skip_residual_threshold:
                    # Skip residual storage - basis is good enough
                    residual_sparse = None
                    residual_indices = None
                    k = 0
                    skipped_residual = True

                # Use smaller chunks to reduce forward pass peak memory
                target_chunk_size = config.get("chunk_size", 512)
                chunk_size = min(target_chunk_size, B_flat)

                # Buffer reuse - get workspace buffers to avoid repeated allocations
                buffer_reuse = config.get("buffer_reuse", False)
                workspace = None
                if buffer_reuse:
                    layer_id = config.get("_layer_id")
                    if layer_id is not None:
                        ws_pool = get_workspace_pool()
                        workspace = ws_pool.get_or_create_workspace(
                            layer_id=layer_id,
                            in_features=in_features,
                            chunk_size=chunk_size,
                            dtype=input_2d.dtype,
                            device=input_2d.device,
                        )

                # Two-pass approach:
                # Pass 1: Sample residuals to estimate quantile threshold
                # Pass 2: Apply threshold to collect residuals

                # Pass 1: Sample to estimate threshold
                # Sample ~10% of elements or 100k, whichever is smaller
                sample_size = min(100000, max(10000, total_elements // 10))

                # Collect samples from each chunk proportionally
                samples_per_chunk = max(1, (sample_size * chunk_size) // B_flat)
                all_samples = []

                for chunk_start in range(0, B_flat, chunk_size):
                    chunk_end = min(chunk_start + chunk_size, B_flat)
                    actual_chunk_size = chunk_end - chunk_start
                    chunk_input = input_2d[chunk_start:chunk_end]
                    chunk_coeff = coefficients[chunk_start:chunk_end]

                    # Compute chunk residual
                    # Use workspace buffers if available for in-place ops
                    if workspace is not None and workspace.get("valid", False):
                        chunk_projected = workspace["chunk_projected"][:actual_chunk_size]
                        chunk_perp = workspace["chunk_perp"][:actual_chunk_size]
                        torch.mm(chunk_coeff, basis_compute.t(), out=chunk_projected)
                        torch.sub(chunk_input, chunk_projected, out=chunk_perp)
                    else:
                        chunk_projected = chunk_coeff.mm(basis_compute.t())
                        chunk_perp = chunk_input - chunk_projected
                        del chunk_projected

                    chunk_flat = chunk_perp.reshape(-1)
                    chunk_abs = chunk_flat.abs()

                    # Random sampling using randint (O(k) memory, not O(n) like randperm)
                    n_samples = min(samples_per_chunk, chunk_flat.numel())
                    if n_samples > 0:
                        sample_indices = torch.randint(
                            0, chunk_flat.numel(), (n_samples,), device=chunk_flat.device
                        )
                        all_samples.append(chunk_abs[sample_indices])

                    if workspace is None or not workspace.get("valid", False):
                        del chunk_perp
                    del chunk_flat, chunk_abs

                # Compute threshold from samples
                all_samples = torch.cat(all_samples)
                # Handle FP16/BF16 - quantile needs float32
                if all_samples.dtype in (torch.float16, torch.bfloat16):
                    quantile_value = 1.0 - residual_budget  # Keep top residual_budget fraction
                    threshold_val = torch.quantile(all_samples.float(), quantile_value).to(
                        all_samples.dtype
                    )
                else:
                    quantile_value = 1.0 - residual_budget
                    threshold_val = torch.quantile(all_samples, quantile_value)
                del all_samples

                # Pass 2: Apply threshold to collect residuals
                all_values = []
                all_indices = []

                for chunk_start in range(0, B_flat, chunk_size):
                    chunk_end = min(chunk_start + chunk_size, B_flat)
                    actual_chunk_size = chunk_end - chunk_start
                    chunk_input = input_2d[chunk_start:chunk_end]
                    chunk_coeff = coefficients[chunk_start:chunk_end]

                    # Compute chunk residual
                    # Use workspace buffers if available for in-place ops
                    if workspace is not None and workspace.get("valid", False):
                        chunk_projected = workspace["chunk_projected"][:actual_chunk_size]
                        chunk_perp = workspace["chunk_perp"][:actual_chunk_size]
                        torch.mm(chunk_coeff, basis_compute.t(), out=chunk_projected)
                        torch.sub(chunk_input, chunk_projected, out=chunk_perp)
                    else:
                        chunk_projected = chunk_coeff.mm(basis_compute.t())
                        chunk_perp = chunk_input - chunk_projected
                        del chunk_projected

                    # Find elements above threshold (O(n) mask operation)
                    chunk_flat = chunk_perp.reshape(-1)
                    mask = chunk_flat.abs() > threshold_val
                    local_indices = mask.nonzero(as_tuple=False).reshape(-1)

                    if local_indices.numel() > 0:
                        local_values = chunk_flat[local_indices]
                        global_offset = chunk_start * in_features
                        global_indices = local_indices + global_offset
                        all_values.append(local_values)
                        all_indices.append(global_indices)

                    if workspace is None or not workspace.get("valid", False):
                        del chunk_perp
                    del chunk_flat

                # Choose optimal index dtype (int16 if small, int32 otherwise)
                index_dtype = _choose_index_dtype(total_elements)
                if all_values:
                    residual_sparse = torch.cat(all_values)
                    residual_indices = torch.cat(all_indices).to(index_dtype)
                else:
                    residual_sparse = torch.tensor([], device=input_2d.device, dtype=input_2d.dtype)
                    residual_indices = torch.tensor([], device=input_2d.device, dtype=index_dtype)

        elif residual_mode == "block_sparse":
            block_size = int(config.get("residual_block_size", 16))
            if block_size <= 0:
                raise ValueError("residual_block_size must be >= 1")

            total_elements = B_flat * in_features
            k_elements = int(total_elements * residual_budget)
            if k_elements <= 0:
                residual_sparse = None
                residual_indices = None
                k_elements = 0
            else:
                k_elements = min(k_elements, total_elements)
            blocks_per_row = (in_features + block_size - 1) // block_size
            k_blocks = max(1, (k_elements + block_size - 1) // block_size) if k_elements > 0 else 0

            # Estimate residual energy without materializing full a_perp
            input_norm_sq = (input_2d * input_2d).sum()
            coeff_norm_sq = (coefficients * coefficients).sum()
            residual_norm_sq = torch.clamp(input_norm_sq - coeff_norm_sq, min=0.0)
            residual_energy = (
                torch.sqrt(residual_norm_sq) / (torch.sqrt(input_norm_sq) + 1e-8)
            ).item()

            # Energy-based residual skip
            skip_residual_enabled = turbo_skip_residual_enabled or config.get(
                "skip_residual_enabled", False
            )
            skip_residual_threshold = (
                turbo_skip_residual_threshold
                if turbo_skip_residual_enabled
                else config.get("skip_residual_energy_threshold", 0.05)
            )
            if skip_residual_enabled and residual_energy < skip_residual_threshold:
                # Skip residual storage - basis is good enough
                residual_sparse = None
                residual_indices = None
                k_elements = 0
                skipped_residual = True

            if k_elements > 0:
                # Use smaller chunks to reduce forward pass peak memory
                target_chunk_size = config.get("chunk_size", 512)
                chunk_size = min(target_chunk_size, B_flat)
                pad_features = blocks_per_row * block_size - in_features

                # Buffer reuse - get workspace buffers to avoid repeated allocations
                buffer_reuse = config.get("buffer_reuse", False)
                workspace = None
                if buffer_reuse:
                    layer_id = config.get("_layer_id")
                    if layer_id is not None:
                        ws_pool = get_workspace_pool()
                        workspace = ws_pool.get_or_create_workspace(
                            layer_id=layer_id,
                            in_features=in_features,
                            chunk_size=chunk_size,
                            dtype=input_2d.dtype,
                            device=input_2d.device,
                        )

                candidate_scores = []
                candidate_block_indices = []
                candidate_block_values = []
                per_chunk_blocks = max(1, (k_blocks * chunk_size + B_flat - 1) // B_flat)

                for chunk_start in range(0, B_flat, chunk_size):
                    chunk_end = min(chunk_start + chunk_size, B_flat)
                    actual_chunk_size = chunk_end - chunk_start
                    chunk_input = input_2d[chunk_start:chunk_end]
                    chunk_coeff = coefficients[chunk_start:chunk_end]

                    # Use workspace buffers if available for in-place ops
                    if workspace is not None and workspace.get("valid", False):
                        chunk_projected = workspace["chunk_projected"][:actual_chunk_size]
                        chunk_perp_raw = workspace["chunk_perp"][:actual_chunk_size]
                        torch.mm(chunk_coeff, basis_compute.t(), out=chunk_projected)
                        torch.sub(chunk_input, chunk_projected, out=chunk_perp_raw)
                        # For block_sparse, we may need to pad - create copy if padding needed
                        if pad_features > 0:
                            chunk_perp = F.pad(chunk_perp_raw.clone(), (0, pad_features))
                        else:
                            chunk_perp = chunk_perp_raw
                    else:
                        chunk_projected = chunk_coeff.mm(basis_compute.t())
                        chunk_perp = chunk_input - chunk_projected
                        del chunk_projected
                        if pad_features > 0:
                            chunk_perp = F.pad(chunk_perp, (0, pad_features))

                    block_view = chunk_perp.contiguous().view(
                        chunk_end - chunk_start, blocks_per_row, block_size
                    )
                    block_scores = (block_view * block_view).sum(dim=2)
                    block_scores_flat = block_scores.reshape(-1)

                    actual_k = min(per_chunk_blocks, block_scores_flat.numel())
                    _, local_block_idx = torch.topk(block_scores_flat, actual_k)
                    selected_scores = block_scores_flat[local_block_idx]
                    selected_values = block_view.view(-1, block_size)[local_block_idx]

                    row_in_chunk = torch.div(local_block_idx, blocks_per_row, rounding_mode="floor")
                    block_in_row = local_block_idx % blocks_per_row
                    global_block_idx = (chunk_start + row_in_chunk) * blocks_per_row + block_in_row

                    candidate_scores.append(selected_scores)
                    candidate_block_indices.append(global_block_idx)
                    candidate_block_values.append(selected_values)
                    del chunk_perp, block_view, block_scores, block_scores_flat

                if candidate_scores:
                    merged_scores = torch.cat(candidate_scores)
                    merged_indices = torch.cat(candidate_block_indices)
                    merged_values = torch.cat(candidate_block_values, dim=0)

                    if merged_scores.numel() > k_blocks:
                        _, top_block_idx = torch.topk(merged_scores, k_blocks)
                        block_indices = merged_indices[top_block_idx]
                        block_values = merged_values[top_block_idx]
                    else:
                        block_indices = merged_indices
                        block_values = merged_values

                    rows = torch.div(block_indices, blocks_per_row, rounding_mode="floor")
                    block_in_row = block_indices % blocks_per_row
                    offsets = torch.arange(block_size, device=block_values.device)

                    elem_indices = (
                        rows[:, None] * in_features
                        + block_in_row[:, None] * block_size
                        + offsets[None, :]
                    )
                    flat_indices = elem_indices.reshape(-1)
                    flat_values = block_values.reshape(-1)

                    if pad_features > 0:
                        valid = (
                            block_in_row[:, None] * block_size + offsets[None, :]
                        ) < in_features
                        valid_flat = valid.reshape(-1)
                        flat_indices = flat_indices[valid_flat]
                        flat_values = flat_values[valid_flat]

                    # Choose optimal index dtype (int16 if small, int32 otherwise)
                    index_dtype = _choose_index_dtype(total_elements)
                    residual_sparse = flat_values
                    residual_indices = flat_indices.to(index_dtype)
                else:
                    index_dtype = _choose_index_dtype(total_elements)
                    residual_sparse = torch.tensor([], device=input_2d.device, dtype=input_2d.dtype)
                    residual_indices = torch.tensor([], device=input_2d.device, dtype=index_dtype)

        else:
            raise ValueError(f"Unsupported residual_mode: {residual_mode}")

        # Update layer state for monitoring (skip during checkpoint recomputation)
        skip_layer_state = False
        try:
            from .adaptive_checkpointing import get_global_recomputation_guard

            skip_layer_state = get_global_recomputation_guard().is_recomputing
        except Exception:
            skip_layer_state = False

        if not skip_layer_state:
            coeff_energy = None
            if coefficients.numel() > 0:
                coeff_energy = coefficients.detach().float().pow(2).mean(dim=0).cpu().tolist()
            layer_state["last_residual_energy"] = residual_energy
            layer_state["step"] = schedule_step
            layer_state["last_coeff_energy_by_rank"] = coeff_energy
            # Track skipped residuals for monitoring
            if skipped_residual:
                layer_state["skipped_residuals"] = layer_state.get("skipped_residuals", 0) + 1
            if not bool(config.get("use_fused_kernel", False)):
                layer_state["b4_fused_kernel_route"] = "disabled"
                layer_state["b4_numerical_parity_error"] = None

        # Optionally quantize coefficients for memory reduction
        quantize_coefficients = config.get("quantize_coefficients", False)
        quant_bits = int(config.get("quantize_bits", 8))
        coeff_scale = None
        if quantize_coefficients:
            if quant_bits != 8:
                raise ValueError(f"Only int8 coefficient quantization supported, got {quant_bits}")
            coefficients_q, coeff_scale = quantize_int8(coefficients)
            coefficients_to_save = coefficients_q
        else:
            coefficients_to_save = coefficients

        # Optionally quantize residuals
        quantize_residuals = bool(config.get("quantize_residuals", False))
        residual_bits = int(config.get("residual_quantize_bits", 8))
        residual_scale = None
        residual_orig_len = None

        # Index compression for memory savings
        # Key insight: Sparse tensor construction doesn't care about order, so we can
        # sort both indices AND values together, eliminating the need to store sort_indices.
        # This achieves ~47% memory savings vs storing sort_indices (which adds 4 bytes/element).
        index_compression_mode = config.get("index_compression", "none")
        compressed_indices = None
        residual_sparse_to_save = residual_sparse  # May be reordered for compression
        if (
            residual_sparse is not None
            and residual_indices is not None
            and index_compression_mode != "none"
        ):
            if residual_indices.numel() > 0:
                flat_indices = residual_indices.reshape(-1)
                flat_values = residual_sparse.reshape(-1)

                # Sort indices and values together (so we don't need sort_indices)
                if index_compression_mode == "csr":
                    # CSR requires row-sorted order; sort by row then compress
                    rows = torch.div(flat_indices.long(), in_features, rounding_mode="floor")
                    sort_idx = torch.argsort(rows, stable=True)
                elif index_compression_mode == "delta":
                    # Delta requires sorted indices
                    sort_idx = torch.argsort(flat_indices)
                else:
                    sort_idx = None

                if sort_idx is not None:
                    sorted_indices = flat_indices[sort_idx]
                    sorted_values = flat_values[sort_idx]

                    # Compress the pre-sorted indices (no need for preserve_order)
                    compressed_indices = compress_indices(
                        flat_indices=sorted_indices,
                        B_flat=B_flat,
                        N=in_features,
                        mode=index_compression_mode,
                        preserve_order=False,  # Values are already reordered
                    )
                    # Both tensors are already 1D, no reshape needed
                    residual_sparse_to_save = sorted_values

        # Quantize residuals AFTER any index reordering/compression.
        # This avoids int4 packing misalignment with residual indices.
        if quantize_residuals and residual_sparse_to_save is not None:
            if residual_bits == 8:
                residual_sparse_to_save, residual_scale = quantize_int8(residual_sparse_to_save)
            elif residual_bits == 4:
                residual_sparse_to_save, residual_scale, residual_orig_len = quantize_int4(
                    residual_sparse_to_save
                )
            else:
                raise ValueError(f"Unsupported residual_quantize_bits: {residual_bits}")

        # Save for backward
        tensors_to_save = [coefficients_to_save, weight]
        if coeff_scale is not None:
            tensors_to_save.append(coeff_scale)
        if bias is not None:
            tensors_to_save.append(bias)
        if residual_sparse is not None:
            tensors_to_save.append(residual_sparse_to_save)
            # Save compressed index tensors if compression enabled
            if compressed_indices is not None:
                # CSR mode: row_ptrs, col_indices (no sort_indices needed)
                # Delta mode: col_indices (deltas), delta_base (no sort_indices needed)
                if compressed_indices.mode == IndexCompressionMode.CSR:
                    tensors_to_save.append(compressed_indices.row_ptrs)
                    tensors_to_save.append(compressed_indices.col_indices)
                elif compressed_indices.mode == IndexCompressionMode.DELTA:
                    tensors_to_save.append(compressed_indices.col_indices)  # deltas
                    tensors_to_save.append(compressed_indices.delta_base)
            else:
                # No compression - store original indices
                tensors_to_save.append(residual_indices)
            if residual_scale is not None:
                tensors_to_save.append(residual_scale)

        # CPU offload for memory savings
        # Move activation-related tensors to CPU, keep weight/bias on GPU
        cpu_offload = bool(config.get("cpu_offload", False))
        if cpu_offload and tensors_to_save[0].is_cuda:
            # Offload coefficients (index 0) and residuals (indices 4+) to CPU
            # Keep weight, coeff_scale, bias on GPU (small or needed for other ops)
            tensors_to_save[0] = tensors_to_save[0].to("cpu", non_blocking=True)
            # Residual tensors start at index 4 (after coefficients, weight, coeff_scale, bias)
            # But their position depends on whether coeff_scale and bias exist
            residual_start = 2  # After coefficients, weight
            if coeff_scale is not None:
                residual_start += 1
            if bias is not None:
                residual_start += 1
            for i in range(residual_start, len(tensors_to_save)):
                tensors_to_save[i] = tensors_to_save[i].to("cpu", non_blocking=True)

        ctx.save_for_backward(*tensors_to_save)
        ctx.cpu_offload = cpu_offload
        ctx.stacr_mode = "stacr"
        ctx.leading_shape = leading_shape
        ctx.has_bias = bias is not None
        ctx.has_residual = residual_sparse is not None
        ctx.residual_shape = (B_flat, in_features)  # Shape of a_perp if it existed

        # Store index compression metadata for backward pass
        # Note: sort_indices no longer stored - values are pre-sorted to match indices
        ctx.index_compression_mode = index_compression_mode
        if compressed_indices is not None:
            ctx.index_compression_nnz = compressed_indices.nnz
            ctx.index_compression_B_flat = compressed_indices.B_flat
            ctx.index_compression_N = compressed_indices.N
            ctx.index_compression_original_dtype = compressed_indices.original_dtype
        else:
            ctx.index_compression_nnz = 0
            ctx.index_compression_B_flat = B_flat
            ctx.index_compression_N = in_features
            ctx.index_compression_original_dtype = torch.int32

        _set_basis_snapshot_for_backward(
            ctx,
            basis_compute=basis_compute,
            config=config,
            layer_state=layer_state,
            warmup_steps=warmup_steps,
            residual_energy=residual_energy,
        )

        ctx.rank = rank
        ctx.quantize_coefficients = quantize_coefficients
        ctx.quantize_residuals = quantize_residuals
        ctx.residual_bits = residual_bits
        ctx.residual_orig_len = residual_orig_len
        ctx.backward_mode = config.get("backward_mode", "sparse")

        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass using stored coefficients and basis.

        Key insight: We never store or reconstruct the full BxN activation.
        Instead, we compute:
            grad_W = G^T @ (B @ c + residual)

        With factor-native approach, this becomes:
            grad_W = G^T @ B @ c + G^T @ residual
                   = (G^T @ B) @ c + G^T @ residual

        The term (G^T @ B) is (out_features, R), much smaller than (out_features, N).
        """
        if ctx.stacr_mode == "baseline":
            # Baseline mode (warmup): standard backward
            saved = ctx.saved_tensors
            input_2d = saved[0]
            weight = saved[1]
            bias = saved[2] if ctx.has_bias else None

            grad_output_2d = grad_output.reshape(-1, grad_output.shape[-1])

            grad_input = grad_weight = grad_bias = None

            if ctx.needs_input_grad[0]:
                # AMP compatibility: cast weight to grad dtype if needed
                weight_compute = (
                    weight
                    if weight.dtype == grad_output_2d.dtype
                    else weight.to(grad_output_2d.dtype)
                )
                grad_input_2d = grad_output_2d.mm(weight_compute)
                grad_input = grad_input_2d.reshape(*ctx.leading_shape, weight.shape[1])

            if ctx.needs_input_grad[1]:
                # AMP compatibility: ensure matching dtypes
                compute_dtype = grad_output_2d.dtype
                input_compute = (
                    input_2d if input_2d.dtype == compute_dtype else input_2d.to(compute_dtype)
                )
                grad_weight = grad_output_2d.t().mm(input_compute)
                if grad_weight.dtype != weight.dtype:
                    grad_weight = grad_weight.to(weight.dtype)

            if bias is not None and ctx.needs_input_grad[2]:
                grad_bias = grad_output_2d.sum(0)
                if grad_bias.dtype != bias.dtype:
                    grad_bias = grad_bias.to(bias.dtype)

            return grad_input, grad_weight, grad_bias, None, None, None, None

        if ctx.stacr_mode == "sparse_only":
            # Sparse-only backward: reconstruct from sparse, approximate weight gradient
            saved = ctx.saved_tensors
            weight = saved[0]
            idx = 1
            bias = None
            if ctx.has_bias:
                bias = saved[idx]
                idx += 1

            B_flat, in_features = ctx.residual_shape
            grad_output_2d = grad_output.reshape(-1, grad_output.shape[-1])
            grad_input = grad_weight = grad_bias = None

            # Input gradient (exact - uses true weights)
            if ctx.needs_input_grad[0]:
                weight_compute = (
                    weight
                    if weight.dtype == grad_output_2d.dtype
                    else weight.to(grad_output_2d.dtype)
                )
                grad_input_2d = grad_output_2d.mm(weight_compute)
                grad_input = grad_input_2d.reshape(*ctx.leading_shape, in_features)

            # Weight gradient (approximate via sparse activation)
            if ctx.needs_input_grad[1]:
                has_sparse = False
                if ctx.has_residual:
                    residual_sparse = saved[idx]
                    residual_indices = saved[idx + 1]
                    idx += 2

                    if ctx.quantize_residuals:
                        residual_scale = saved[idx]
                        idx += 1
                        if residual_sparse.numel() > 0:
                            if ctx.residual_bits == 8:
                                residual_sparse = dequantize_int8(residual_sparse, residual_scale)
                            elif ctx.residual_bits == 4:
                                orig_len = ctx.residual_orig_len or 0
                                residual_sparse = dequantize_int4(
                                    residual_sparse, residual_scale, orig_len
                                )

                    indices_1d = residual_indices.reshape(-1).long()
                    values_1d = residual_sparse.reshape(-1)

                    valid_mask = (indices_1d >= 0) & (indices_1d < B_flat * in_features)
                    if valid_mask.any() and values_1d.numel() > 0:
                        indices_1d = indices_1d[valid_mask]
                        values_1d = values_1d[valid_mask]
                        if values_1d.dtype != torch.float32:
                            values_1d = values_1d.float()
                        rows = torch.div(indices_1d, in_features, rounding_mode="floor")
                        cols = indices_1d % in_features
                        indices = torch.stack([rows, cols], dim=0)
                        A_sparse = torch.sparse_coo_tensor(
                            indices,
                            values_1d,
                            size=(B_flat, in_features),
                            device=grad_output_2d.device,
                            dtype=torch.float32,
                        ).coalesce()
                        grad_output_f32 = (
                            grad_output_2d.float()
                            if grad_output_2d.dtype != torch.float32
                            else grad_output_2d
                        )
                        grad_weight = torch.sparse.mm(A_sparse.t(), grad_output_f32).t()
                        if grad_weight.dtype != weight.dtype:
                            grad_weight = grad_weight.to(weight.dtype)
                        has_sparse = True

                if not has_sparse:
                    grad_weight = torch.zeros_like(weight)

            # Bias gradient
            if bias is not None and ctx.needs_input_grad[2]:
                grad_bias = grad_output_2d.sum(0)
                if grad_bias.dtype != bias.dtype:
                    grad_bias = grad_bias.to(bias.dtype)

            return grad_input, grad_weight, grad_bias, None, None, None, None

        # === STAC-R Backward ===

        saved = ctx.saved_tensors
        coefficients_saved = saved[0]  # (B_flat, R) - may be int8 if quantized
        weight = saved[1]  # (out_features, in_features)

        # Move offloaded tensors back to GPU
        target_device = weight.device
        if ctx.cpu_offload and coefficients_saved.device.type == "cpu":
            coefficients_saved = coefficients_saved.to(target_device, non_blocking=True)

        idx = 2

        # Dequantize coefficients if they were quantized
        if ctx.quantize_coefficients:
            coeff_scale = saved[idx]
            coefficients = dequantize_int8(coefficients_saved, coeff_scale)
            idx += 1
        else:
            coefficients = coefficients_saved

        if ctx.has_bias:
            bias = saved[idx]
            idx += 1
        else:
            bias = None

        residual_sparse = None
        residual_indices = None
        if ctx.has_residual:
            residual_sparse = saved[idx]
            idx += 1

            # Decompress indices if compression was used
            # Note: Values are stored in sorted order to match indices, so no restore_order needed.
            index_compression_mode = ctx.index_compression_mode
            if index_compression_mode != "none" and ctx.index_compression_nnz > 0:
                # Reconstruct CompressedIndices from saved tensors
                if index_compression_mode == "csr":
                    row_ptrs = saved[idx]
                    col_indices = saved[idx + 1]
                    idx += 2
                    # Move offloaded tensors back to GPU (sync to ensure completion)
                    if ctx.cpu_offload:
                        if row_ptrs.device.type == "cpu":
                            row_ptrs = row_ptrs.to(target_device)  # Sync transfer
                        if col_indices.device.type == "cpu":
                            col_indices = col_indices.to(target_device)  # Sync transfer
                    # NOTE: Indices/values are in SORTED order to save memory (no sort_indices stored).
                    # This is safe because torch.sparse_coo_tensor(...).coalesce() reorders anyway.
                    compressed = CompressedIndices(
                        mode=IndexCompressionMode.CSR,
                        row_ptrs=row_ptrs,
                        col_indices=col_indices,
                        sort_indices=None,  # Values already sorted to match
                        nnz=ctx.index_compression_nnz,
                        B_flat=ctx.index_compression_B_flat,
                        N=ctx.index_compression_N,
                        original_dtype=ctx.index_compression_original_dtype,
                    )
                    residual_indices = decompress_indices(compressed, restore_order=False)
                elif index_compression_mode == "delta":
                    deltas = saved[idx]
                    delta_base = saved[idx + 1]
                    idx += 2
                    # Move offloaded tensors back to GPU (sync to ensure completion)
                    if ctx.cpu_offload:
                        if deltas.device.type == "cpu":
                            deltas = deltas.to(target_device)  # Sync transfer
                        if delta_base.device.type == "cpu":
                            delta_base = delta_base.to(target_device)  # Sync transfer
                    compressed = CompressedIndices(
                        mode=IndexCompressionMode.DELTA,
                        col_indices=deltas,
                        delta_base=delta_base,
                        sort_indices=None,  # Values already sorted to match
                        nnz=ctx.index_compression_nnz,
                        B_flat=ctx.index_compression_B_flat,
                        N=ctx.index_compression_N,
                        original_dtype=ctx.index_compression_original_dtype,
                    )
                    residual_indices = decompress_indices(compressed, restore_order=False)
                else:
                    # Unknown mode, fall back to direct load
                    residual_indices = saved[idx]
                    idx += 1
            else:
                # No compression - load indices directly
                residual_indices = saved[idx]
                idx += 1

            # Move offloaded residuals back to GPU
            if ctx.cpu_offload and residual_sparse.device.type == "cpu":
                residual_sparse = residual_sparse.to(target_device, non_blocking=True)
            if (
                index_compression_mode == "none"
                and ctx.cpu_offload
                and residual_indices.device.type == "cpu"
            ):
                residual_indices = residual_indices.to(target_device, non_blocking=True)

            if ctx.quantize_residuals:
                residual_scale = saved[idx]
                idx += 1
                # Move offloaded scale back to GPU
                if ctx.cpu_offload and residual_scale.device.type == "cpu":
                    residual_scale = residual_scale.to(target_device, non_blocking=True)
                if residual_sparse.numel() > 0:
                    if ctx.residual_bits == 8:
                        residual_sparse = dequantize_int8(residual_sparse, residual_scale)
                    elif ctx.residual_bits == 4:
                        orig_len = ctx.residual_orig_len or 0
                        residual_sparse = dequantize_int4(residual_sparse, residual_scale, orig_len)
                    else:
                        raise ValueError(f"Unsupported residual_quantize_bits: {ctx.residual_bits}")
                else:
                    residual_sparse = residual_sparse.to(residual_scale.dtype)

        basis = ctx.basis  # (N, R)
        B_flat = coefficients.shape[0]
        in_features = weight.shape[1]

        grad_output_2d = grad_output.reshape(-1, grad_output.shape[-1])  # (B_flat, out_features)

        grad_input = grad_weight = grad_bias = None

        # Input gradient (exact - uses true weights)
        if ctx.needs_input_grad[0]:
            # AMP compatibility: cast weight to grad dtype if needed
            weight_compute = (
                weight if weight.dtype == grad_output_2d.dtype else weight.to(grad_output_2d.dtype)
            )
            grad_input_2d = grad_output_2d.mm(weight_compute)  # (B_flat, in_features)
            grad_input = grad_input_2d.reshape(*ctx.leading_shape, in_features)

        # Weight gradient (approximate via subspace projection)
        if ctx.needs_input_grad[1]:
            # Factor-native approach: avoid reconstructing full B x N activation
            # grad_W = G^T @ (C @ B^T) + G^T @ residual
            #       = (G^T @ C) @ B^T + (R^T @ G)^T

            # AMP compatibility: ensure all tensors have matching dtype
            # grad_output_2d may be FP16 under AMP, while coefficients/basis are FP32
            compute_dtype = grad_output_2d.dtype
            if coefficients.dtype != compute_dtype:
                coefficients = coefficients.to(compute_dtype)
            basis_compute = basis if basis.dtype == compute_dtype else basis.to(compute_dtype)

            # Core term: (G^T @ C) @ B^T
            gtc = grad_output_2d.t().mm(coefficients)  # (out, R)
            grad_weight_compute = gtc.mm(basis_compute.t())  # (out, in)

            # Residual term: sparse or dense contribution without dense activation reconstruction
            if (
                residual_sparse is not None
                and residual_indices is not None
                and residual_sparse.numel() > 0
            ):
                indices_1d = residual_indices.reshape(-1)
                values_1d = residual_sparse.reshape(-1)
                valid_mask = (indices_1d >= 0) & (indices_1d < B_flat * in_features)
                if valid_mask.any():
                    indices_1d = indices_1d[valid_mask]
                    values_1d = values_1d[valid_mask]
                    # AMP compatibility: cast residual values to compute dtype
                    if values_1d.dtype != compute_dtype:
                        values_1d = values_1d.to(compute_dtype)
                    # Cast to int64 for sparse tensor (requires long indices)
                    indices_1d_long = indices_1d.long()
                    rows = torch.div(indices_1d_long, in_features, rounding_mode="floor")
                    cols = indices_1d_long % in_features
                    backward_mode = getattr(ctx, "backward_mode", None)
                    if backward_mode is None:
                        raise RuntimeError("STACR backward missing ctx.backward_mode.")
                    if backward_mode == "dense":
                        grad_rows = grad_output_2d.index_select(0, rows)
                        scaled = values_1d.unsqueeze(1) * grad_rows
                        if grad_weight_compute.dtype != torch.float32:
                            grad_weight_accum = grad_weight_compute.float()
                            grad_weight_accum.index_add_(1, cols, scaled.float().t().contiguous())
                            grad_weight_compute = grad_weight_accum.to(compute_dtype)
                        else:
                            grad_weight_compute.index_add_(1, cols, scaled.t().contiguous())
                    else:
                        indices = torch.stack([rows, cols], dim=0)
                        # Sparse ops don't support FP16 on CUDA - always use float32
                        sparse_dtype = torch.float32
                        residual_coo = torch.sparse_coo_tensor(
                            indices,
                            values_1d.float(),  # Cast to float32 for sparse ops
                            size=(B_flat, in_features),
                            device=grad_output_2d.device,
                            dtype=sparse_dtype,
                        ).coalesce()
                        # Compute sparse contribution in float32, then cast back
                        grad_output_f32 = (
                            grad_output_2d.float()
                            if grad_output_2d.dtype != torch.float32
                            else grad_output_2d
                        )
                        sparse_contrib = torch.sparse.mm(residual_coo.t(), grad_output_f32).t()
                        grad_weight_compute = grad_weight_compute + sparse_contrib.to(compute_dtype)

            grad_weight = grad_weight_compute
            if grad_weight.dtype != weight.dtype:
                grad_weight = grad_weight.to(weight.dtype)

        # Bias gradient
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output_2d.sum(0)
            if grad_bias.dtype != bias.dtype:
                grad_bias = grad_bias.to(bias.dtype)

        # Return gradients for all forward args
        return grad_input, grad_weight, grad_bias, None, None, None, None


def _set_basis_snapshot_for_backward(
    ctx,
    basis_compute: torch.Tensor,
    config: dict,
    layer_state: dict,
    warmup_steps: int,
    residual_energy: float,
) -> None:
    """Save either a basis reference or a cloned snapshot for backward."""
    freeze_after_warmup = config.get("freeze_basis_after_warmup", False)
    schedule_step = int(layer_state.get("_schedule_step", 0))
    past_warmup = schedule_step >= warmup_steps
    update_trigger = config.get("update_trigger", "hybrid")
    update_interval = int(config.get("update_interval", 100))
    lazy_enabled = config.get("lazy_update_enabled", True)
    lazy_threshold = float(config.get("lazy_update_threshold", 0.01))
    refresh_trigger = config.get("refresh_trigger", "off")
    uses_optimizer_schedule = bool(
        config.get("oja_update_per_optimizer_step", False)
        or config.get("cache_basis_per_optimizer_step", False)
        or refresh_trigger != BasisRefreshTrigger.OFF.value
        or config.get("post_warmup_residual_mode", "full") != "full"
    )
    cache_basis_per_optimizer_step = bool(config.get("cache_basis_per_optimizer_step", False))
    microbatch_index = int(layer_state.get("_microbatch_index_runtime", 0))
    accum_steps = max(1, int(layer_state.get("_accum_steps_runtime", 1)))

    skip_clone = False
    skip_reason = None

    if cache_basis_per_optimizer_step and accum_steps > 1 and microbatch_index > 0:
        skip_clone = True
        skip_reason = "microbatch_cache"
    elif freeze_after_warmup and past_warmup:
        if refresh_trigger == BasisRefreshTrigger.OFF.value:
            skip_clone = True
            skip_reason = "frozen_after_warmup"
        else:
            refresh_period = max(1, int(config.get("refresh_period", 1000)))
            refresh_energy_threshold = float(config.get("refresh_energy_threshold", 0.1))
            fixed_refresh = _is_fixed_refresh_step(
                schedule_step=schedule_step,
                warmup_steps=warmup_steps,
                refresh_period=refresh_period,
                layer_index=layer_state.get("layer_index"),
                num_layers=max(1, int(layer_state.get("num_layers", 1))),
            )
            energy_refresh = residual_energy > refresh_energy_threshold
            should_refresh = False
            if refresh_trigger == BasisRefreshTrigger.FIXED.value:
                should_refresh = fixed_refresh
            elif refresh_trigger == BasisRefreshTrigger.RESIDUAL_ENERGY.value:
                should_refresh = energy_refresh
            elif refresh_trigger == BasisRefreshTrigger.HYBRID.value:
                should_refresh = fixed_refresh or energy_refresh
            if not should_refresh:
                skip_clone = True
                skip_reason = "refresh_not_scheduled"
    elif update_trigger == "fixed":
        effective_update_step = schedule_step if uses_optimizer_schedule else (schedule_step + 1)
        will_update = effective_update_step > 0 and effective_update_step % update_interval == 0
        if not will_update:
            skip_clone = True
            skip_reason = "fixed_interval_not_update_step"
    elif lazy_enabled and residual_energy < lazy_threshold:
        skip_clone = True
        skip_reason = "lazy_update_skip"

    if skip_clone:
        ctx.basis = basis_compute.detach()
        ctx.basis_clone_skipped = True
        ctx.basis_skip_reason = skip_reason
    else:
        ctx.basis = basis_compute.detach().clone()
        ctx.basis_clone_skipped = False
        ctx.basis_skip_reason = None


class STACRLowRankOnlyFunction(torch.autograd.Function):
    """Dedicated low-rank-only STAC/Turbo path with no residual branches."""

    @staticmethod
    def forward(ctx, input_tensor, weight, bias, basis, config, step_counter, layer_state):
        del step_counter  # Routing ensures schedule is already encoded in layer_state.
        schedule_step = int(layer_state.get("_schedule_step", 0))
        effective_residual_mode = config.get("residual_mode", "topk")
        residual_warmup = int(config.get("residual_warmup_steps", 0))
        post_warmup_mode = config.get("post_warmup_residual_mode", "full")
        if post_warmup_mode != "full" and residual_warmup > 0 and schedule_step >= residual_warmup:
            if post_warmup_mode == "none":
                effective_residual_mode = "none"
            elif post_warmup_mode == "threshold":
                effective_residual_mode = "threshold"
        if effective_residual_mode != "none":
            raise AssertionError(
                f"STACRLowRankOnlyFunction received effective_residual_mode={effective_residual_mode}; "
                "B1 route is only valid when residuals are truly off."
            )

        in_features = weight.shape[1]
        leading_shape = input_tensor.shape[:-1]
        input_2d = input_tensor.reshape(-1, in_features)

        weight_compute = weight if weight.dtype == input_2d.dtype else weight.to(input_2d.dtype)
        output_2d = input_2d.mm(weight_compute.t())
        if bias is not None:
            bias_compute = bias if bias.dtype == output_2d.dtype else bias.to(output_2d.dtype)
            output_2d = output_2d + bias_compute.unsqueeze(0)
        output = output_2d.reshape(*leading_shape, weight.shape[0])

        basis_compute = basis if basis.dtype == input_2d.dtype else basis.to(input_2d.dtype)
        coefficients = input_2d.mm(basis_compute)

        input_norm_sq = (input_2d * input_2d).sum()
        coeff_norm_sq = (coefficients * coefficients).sum()
        residual_norm_sq = torch.clamp(input_norm_sq - coeff_norm_sq, min=0.0)
        residual_energy = (torch.sqrt(residual_norm_sq) / (torch.sqrt(input_norm_sq) + 1e-8)).item()

        skip_layer_state = False
        try:
            from .adaptive_checkpointing import get_global_recomputation_guard

            skip_layer_state = get_global_recomputation_guard().is_recomputing
        except Exception:
            skip_layer_state = False

        if not skip_layer_state:
            coeff_energy = None
            if coefficients.numel() > 0:
                coeff_energy = coefficients.detach().float().pow(2).mean(dim=0).cpu().tolist()
            layer_state["last_residual_energy"] = residual_energy
            layer_state["step"] = int(layer_state.get("_schedule_step", 0))
            layer_state["last_coeff_energy_by_rank"] = coeff_energy

        coefficients_to_save = coefficients

        tensors_to_save = [coefficients_to_save, weight]
        if bias is not None:
            tensors_to_save.append(bias)

        cpu_offload = bool(config.get("cpu_offload", False))
        if cpu_offload and tensors_to_save[0].is_cuda:
            tensors_to_save[0] = tensors_to_save[0].to("cpu", non_blocking=True)

        ctx.save_for_backward(*tensors_to_save)
        ctx.cpu_offload = cpu_offload
        ctx.leading_shape = leading_shape
        ctx.has_bias = bias is not None
        ctx.quantize_coefficients = False
        _set_basis_snapshot_for_backward(
            ctx,
            basis_compute=basis_compute,
            config=config,
            layer_state=layer_state,
            warmup_steps=int(config.get("warmup_steps", 0)),
            residual_energy=residual_energy,
        )

        return output

    @staticmethod
    def backward(ctx, grad_output):
        saved = ctx.saved_tensors
        coefficients_saved = saved[0]
        weight = saved[1]
        target_device = weight.device

        if ctx.cpu_offload and coefficients_saved.device.type == "cpu":
            coefficients_saved = coefficients_saved.to(target_device, non_blocking=True)

        idx = 2
        if ctx.quantize_coefficients:
            coeff_scale = saved[idx]
            coefficients = dequantize_int8(coefficients_saved, coeff_scale)
            idx += 1
        else:
            coefficients = coefficients_saved

        if ctx.has_bias:
            bias = saved[idx]
        else:
            bias = None

        basis = ctx.basis
        in_features = weight.shape[1]
        grad_output_2d = grad_output.reshape(-1, grad_output.shape[-1])

        grad_input = grad_weight = grad_bias = None

        if ctx.needs_input_grad[0]:
            weight_compute = (
                weight if weight.dtype == grad_output_2d.dtype else weight.to(grad_output_2d.dtype)
            )
            grad_input_2d = grad_output_2d.mm(weight_compute)
            grad_input = grad_input_2d.reshape(*ctx.leading_shape, in_features)

        if ctx.needs_input_grad[1]:
            compute_dtype = grad_output_2d.dtype
            if coefficients.dtype != compute_dtype:
                coefficients = coefficients.to(compute_dtype)
            basis_compute = basis if basis.dtype == compute_dtype else basis.to(compute_dtype)
            grad_weight_compute = grad_output_2d.t().mm(coefficients).mm(basis_compute.t())
            grad_weight = (
                grad_weight_compute
                if grad_weight_compute.dtype == weight.dtype
                else grad_weight_compute.to(weight.dtype)
            )

        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output_2d.sum(0)
            if grad_bias.dtype != bias.dtype:
                grad_bias = grad_bias.to(bias.dtype)

        return grad_input, grad_weight, grad_bias, None, None, None, None


class STACRLinear(nn.Module):
    """
    STAC-R Linear layer: drop-in replacement for nn.Linear with subspace compression.

    Usage:
        # STAC-R with residual correction
        layer = STACRLinear(512, 256, STACRConfig(rank=32, residual_mode="topk"))

        # STAC ablation (no residual)
        layer = STACRLinear(512, 256, STACRConfig(rank=32, residual_mode="none"))
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        config: Optional[STACRConfig] = None,
        bias: bool = True,
        layer_index: Optional[int] = None,  # For basis sharing, A1: For adaptive rank
        num_layers: int = 12,  # Total number of layers (for heuristic rank allocation)
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.config = config or STACRConfig()
        if self.config.streaming_algorithm != "oja":
            warnings.warn(
                f"streaming_algorithm={self.config.streaming_algorithm} is not implemented; "
                "falling back to 'oja'."
            )
            self.config.streaming_algorithm = "oja"

        # Standard linear layer (holds weights)
        self.linear = nn.Linear(in_features, out_features, bias=bias)

        # Compute effective rank based on layer position
        base_rank = self.config.rank
        if layer_index is not None and self.config.adaptive_layer_rank != "none":
            self.effective_rank = self.config.get_layer_rank(layer_index, base_rank, num_layers)
        else:
            self.effective_rank = base_rank

        # Persistent basis: (in_features, rank)
        rank = min(self.effective_rank, in_features)
        self.basis_dtype = _resolve_basis_dtype(self.config.basis_dtype)

        # Basis sharing setup, A1: Layer index for adaptive rank
        self.layer_index = layer_index
        self.num_layers = num_layers  # Store for logging
        self.use_basis_sharing = self.config.basis_sharing != "none"
        self.basis_group_id: Optional[int] = None

        if self.use_basis_sharing and self.config.rank_mode != "fixed":
            warnings.warn(
                "basis_sharing only supports rank_mode='fixed'; disabling basis sharing "
                f"for rank_mode={self.config.rank_mode!r}."
            )
            self.use_basis_sharing = False

        if self.use_basis_sharing:
            # Calculate group ID based on sharing mode
            if self.config.basis_sharing_group is not None:
                # Manual override
                self.basis_group_id = self.config.basis_sharing_group
            elif layer_index is not None:
                # Calculate from layer_index
                if self.config.basis_sharing == "all":
                    # All layers with same in_features share one basis
                    self.basis_group_id = 0
                elif self.config.basis_sharing == "every_2":
                    self.basis_group_id = layer_index // 2
                elif self.config.basis_sharing == "every_3":
                    self.basis_group_id = layer_index // 3
                elif self.config.basis_sharing == "every_4":
                    self.basis_group_id = layer_index // 4
                else:
                    warnings.warn(
                        f"Unknown basis_sharing mode: {self.config.basis_sharing}, using 'none'"
                    )
                    self.use_basis_sharing = False
                    self.basis_group_id = None
            else:
                # layer_index not provided but sharing enabled -> warn and disable
                warnings.warn(
                    "basis_sharing enabled but layer_index not provided, disabling sharing"
                )
                self.use_basis_sharing = False
                self.basis_group_id = None

        if self.use_basis_sharing:
            # Use shared basis from pool (no local buffer needed)
            self.basis = None  # Will be retrieved from pool
        else:
            # Use local basis buffer
            self.register_buffer("basis", torch.zeros(in_features, rank, dtype=self.basis_dtype))
        # Persist initialization so a resumed run keeps its fitted basis.
        self.register_buffer(
            "_basis_initialized", torch.zeros(1, dtype=torch.bool), persistent=True
        )

        # Step counter (as buffer so it persists across forward calls)
        self.register_buffer("step_counter", torch.zeros(1, dtype=torch.long))
        # Tensor-only checkpoint metadata works with torch.save and safetensors.
        self.register_buffer("_runtime_state", torch.empty(0, dtype=torch.uint8))
        self.register_buffer("_shared_basis_state", torch.empty(0))

        # Layer state for monitoring
        self.layer_state = {
            "last_residual_energy": 0.0,
            "step": 0,
            "basis_updates": 0,
            "skipped_updates": 0,  # Lazy update optimization counter
            "skipped_recompute": 0,  # Checkpoint recomputation counter
            "frozen_updates": 0,
            "refresh_count_total": 0,
            "refresh_count_by_reason": {},
            "microbatch_cache_hits": 0,
            "microbatch_cache_misses": 0,
            "b1_route_step_count": 0,
            "b1_generic_fallback_count": 0,
            "active_rank": rank,
            "allocated_rank": rank,
            "last_coeff_energy_by_rank": None,
            "rank_reason_code_counts": {},
            "rank_trajectory": [],
            "last_rank_reason": "fixed",
            "_last_rank_adjust_step": -1,
            "b4_fused_kernel_route": "disabled",
            "b4_fused_kernel_route_counts": {},
            "b4_numerical_parity_error": None,
            "optimizer_step": 0,
            "microbatch_index": 0,
            "accum_steps": 1,
            # Adaptive layer rank info
            "effective_rank": self.effective_rank,
            "base_rank": base_rank,
            "layer_index": layer_index,
            "num_layers": num_layers,
            # Dynamic residual schedule info
            "total_steps": None,  # Set externally before training
            "current_residual_budget": self.config.residual_budget,
        }

        # Unique layer ID for workspace pool (use id() for uniqueness)
        self._layer_id = id(self)

    @classmethod
    def from_linear(cls, module, config=None, *, layer_index=None, num_layers=12):
        """Reuse a Linear/Conv1D projection's parameters and preserve optimizer references."""
        from .surgery import _is_conv1d

        transposed = _is_conv1d(module)
        if not isinstance(module, nn.Linear) and not transposed:
            raise TypeError("Expected nn.Linear or Transformers Conv1D")
        in_features, out_features = module.weight.shape if transposed else module.weight.shape[::-1]
        # Temporary nn.Linear initialization must not change the caller's RNG.
        with torch.random.fork_rng(devices=[]):
            result = cls(
                in_features,
                out_features,
                copy.deepcopy(config),
                bias=getattr(module, "bias", None) is not None,
                layer_index=layer_index,
                num_layers=num_layers,
            )
        result.to(module.weight.device)
        result.linear = _TransposedProjection(module) if transposed else module
        result.train(module.training)
        return result

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        serializable = {
            key: value
            for key, value in self.layer_state.items()
            if isinstance(value, (str, int, float, bool, list, dict, type(None)))
        }
        config = asdict(self.config)
        if isinstance(config["basis_dtype"], torch.dtype):
            config["basis_dtype"] = str(config["basis_dtype"])
        payload = json.dumps(
            {"layer_state": serializable, "config": config, "effective_rank": self.effective_rank}
        ).encode("utf-8")
        self._runtime_state = torch.tensor(
            list(payload), dtype=torch.uint8, device=self.step_counter.device
        )
        if self.use_basis_sharing and self.basis_initialized:
            self._shared_basis_state = self.effective_basis.detach()
        super()._save_to_state_dict(destination, prefix, keep_vars)

    @property
    def basis_initialized(self) -> bool:
        """Persistent initialization flag exposed as a bool.

        Reads from the registered buffer self._basis_initialized so the value
        survives state_dict round-trips. The setter writes in-place via fill_
        so device placement and the Tensor object identity are preserved.
        """
        return bool(self._basis_initialized.item())

    @basis_initialized.setter
    def basis_initialized(self, value: bool) -> None:
        if not hasattr(self, "_basis_initialized"):
            # Only possible during very early __init__ ordering issues.
            object.__setattr__(self, "_basis_initialized_pending", bool(value))
            return
        self._basis_initialized.fill_(bool(value))

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        """Restore basis initialization and runtime scheduling state.

        Old checkpoints (saved before basis_initialized became a buffer) only
        contain `basis` and `step_counter`. Treat any such basis whose values are
        non-zero as evidence the basis was initialized at save time, so the next
        train() forward does NOT overwrite it. New checkpoints contain
        `_basis_initialized` directly and take the standard path.
        """
        runtime_key, shared_key = prefix + "_runtime_state", prefix + "_shared_basis_state"
        runtime = state_dict.get(runtime_key)
        if runtime is not None and runtime.numel():
            payload = json.loads(bytes(runtime.cpu().tolist()).decode("utf-8"))
            self.layer_state.update(payload["layer_state"])
            self.config = STACRConfig(**payload["config"])
            self.effective_rank = payload["effective_rank"]
            self._runtime_state = torch.empty_like(runtime, device=self.step_counter.device)
        else:
            state_dict[runtime_key] = self._runtime_state
        shared = state_dict.get(shared_key)
        if shared is not None:
            self._shared_basis_state = torch.empty_like(shared, device=self.step_counter.device)
            if shared.numel() and self.use_basis_sharing:
                get_basis_pool().update_basis(
                    self.basis_group_id,
                    self.in_features,
                    shared.shape[1],
                    shared.to(self.linear.weight.device),
                )
        else:
            state_dict[shared_key] = self._shared_basis_state
        loaded_basis = state_dict.get(prefix + "basis")
        if (
            loaded_basis is not None
            and self.basis is not None
            and loaded_basis.shape != self.basis.shape
        ):
            self.basis = torch.empty_like(loaded_basis, device=self.linear.weight.device)
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )
        init_key = prefix + "_basis_initialized"
        basis_key = prefix + "basis"
        if init_key in missing_keys and basis_key in state_dict:
            loaded_basis = state_dict[basis_key]
            if loaded_basis is not None and loaded_basis.numel() > 0:
                try:
                    if torch.any(loaded_basis != 0):
                        self._basis_initialized.fill_(True)
                        missing_keys.remove(init_key)
                except Exception:
                    # Non-fatal: leave the flag at False rather than guessing.
                    pass

    def set_runtime_state(
        self,
        optimizer_step: Optional[int] = None,
        microbatch_index: int = 0,
        accum_steps: int = 1,
    ) -> None:
        """Inject per-forward runtime scheduling state from the training loop."""
        if optimizer_step is None:
            self.layer_state.pop("_optimizer_step_runtime", None)
        else:
            self.layer_state["_optimizer_step_runtime"] = int(optimizer_step)
            self.layer_state["optimizer_step"] = int(optimizer_step)
        self.layer_state["_microbatch_index_runtime"] = int(microbatch_index)
        self.layer_state["_accum_steps_runtime"] = max(1, int(accum_steps))
        self.layer_state["microbatch_index"] = int(microbatch_index)
        self.layer_state["accum_steps"] = max(1, int(accum_steps))

    def _get_schedule_step(self) -> int:
        if self._use_optimizer_step_schedule():
            runtime_step = self.layer_state.get("_optimizer_step_runtime")
            if runtime_step is not None:
                return int(runtime_step)
        return int(self.step_counter.item())

    def _get_forward_schedule_step(self) -> int:
        if self._use_optimizer_step_schedule():
            runtime_step = self.layer_state.get("_optimizer_step_runtime")
            if runtime_step is not None:
                return int(runtime_step)
        return int(self.step_counter.item())

    def _get_microbatch_index(self) -> int:
        return int(self.layer_state.get("_microbatch_index_runtime", 0))

    def _get_accum_steps(self) -> int:
        return max(1, int(self.layer_state.get("_accum_steps_runtime", 1)))

    def _use_optimizer_step_schedule(self) -> bool:
        return bool(
            self.config.oja_update_per_optimizer_step
            or self.config.cache_basis_per_optimizer_step
            or self.config.refresh_trigger != BasisRefreshTrigger.OFF.value
            or self.config.post_warmup_residual_mode != "full"
        )

    def _should_use_low_rank_only_route(self, schedule_step: int) -> bool:
        if not self.config.enable_low_rank_only_fast_path:
            return False
        if schedule_step < self.config.warmup_steps:
            return False
        if not self.basis_initialized:
            return False
        if self.config.residual_mode == "none":
            return True
        return bool(
            self.config.post_warmup_residual_mode == "none"
            and self.config.residual_warmup_steps > 0
            and schedule_step >= self.config.residual_warmup_steps
        )

    def _record_update_reason(self, reason: Optional[str]) -> None:
        if not reason:
            return
        self.layer_state["_pending_update_reason"] = reason

    def _consume_update_reason(self) -> Optional[str]:
        return self.layer_state.pop("_pending_update_reason", None)

    def get_active_rank(self) -> int:
        """Return the active prefix rank used for the current forward path."""
        active_rank = self.layer_state.get("active_rank")
        if active_rank is None:
            return self.get_effective_rank()
        return max(1, min(int(active_rank), self.get_effective_rank()))

    def _select_active_rank(self, schedule_step: int, allocated_rank: int) -> int:
        """Choose the active prefix rank for this forward without resizing buffers."""
        allocated_rank = max(1, int(allocated_rank))
        if self.use_basis_sharing and self.config.rank_mode != "fixed":
            raise RuntimeError("Adaptive runtime rank is incompatible with basis sharing.")
        current_rank = max(
            self.config.rank_min,
            min(int(self.layer_state.get("active_rank", allocated_rank)), allocated_rank),
        )
        rank_mode = self.config.rank_mode
        if rank_mode == "fixed":
            target_rank = allocated_rank
            self.layer_state["last_rank_reason"] = "fixed"
            return target_rank

        if not should_adjust_rank(
            schedule_step=schedule_step,
            last_adjust_step=int(self.layer_state.get("_last_rank_adjust_step", -1)),
            adjust_interval=self.config.rank_adjust_interval,
        ):
            return current_rank

        decision = choose_runtime_rank(
            rank_mode=rank_mode,
            current_rank=current_rank,
            rank_min=self.config.rank_min,
            rank_max=self.config.rank_max,
            max_available_rank=allocated_rank,
            layer_state=self.layer_state,
            variance_threshold=self.config.rank_variance_threshold,
            sensitivity_low=self.config.rank_sensitivity_low,
            sensitivity_high=self.config.rank_sensitivity_high,
            step_fraction=self.config.rank_step_fraction,
        )
        target_rank = max(1, min(int(decision.active_rank), allocated_rank))
        self.layer_state["_last_rank_adjust_step"] = int(schedule_step)
        self.layer_state["last_rank_reason"] = decision.reason_code

        reason_counts = self.layer_state.setdefault("rank_reason_code_counts", {})
        reason_counts[decision.reason_code] = int(reason_counts.get(decision.reason_code, 0)) + 1

        trajectory = self.layer_state.setdefault("rank_trajectory", [])
        trajectory.append(
            {
                "step": int(schedule_step),
                "active_rank": int(target_rank),
                "reason_code": decision.reason_code,
                "metric_value": None
                if decision.metric_value is None
                else float(decision.metric_value),
            }
        )
        max_traj = 128
        if len(trajectory) > max_traj:
            del trajectory[:-max_traj]
        return target_rank

    @property
    def effective_basis(self) -> Optional[torch.Tensor]:
        """Get the effective basis (either local or from pool).

        Retrieves a shared basis from BasisPool when basis sharing is enabled.
        Otherwise: returns local self.basis buffer.
        """
        if self.use_basis_sharing and self.basis_group_id is not None:
            pool = get_basis_pool()
            rank = min(self.effective_rank, self.in_features)
            # Get current device from linear layer weights
            device = self.linear.weight.device
            return pool.get_or_create_basis(
                self.basis_group_id, self.in_features, rank, device, self.basis_dtype
            )
        else:
            return self.basis

    def _init_basis_from_batch(self, input_2d: torch.Tensor):
        """Initialize basis from first batch using SVD.

        Uses randomized SVD (torch.svd_lowrank) for large features to:
        - Reduce O(N^2 R) to O(N R^2) complexity
        - Lower workspace memory from O(min(B,N)^2) to O(N R)

        Supports adaptive rank selection when rank_mode="variance":
        - Calls profile_layer_spectrum() to determine rank needed to capture threshold variance
        - Clamps result between rank_min and rank_max
        - Dynamically resizes basis buffer to the adaptive rank
        """
        with torch.no_grad(), _disable_autocast(input_2d.device):
            # Compute SVD of input activations
            # input_2d: (B, N), we want top-R right singular vectors
            # A = U @ S @ V^T, V columns are right singular vectors

            batch_size, in_features = input_2d.shape
            # Use effective_basis to support basis sharing (avoids None basis)
            current_basis = self.effective_basis
            target_rank = current_basis.shape[1] if current_basis is not None else self.config.rank

            # === Adaptive Rank Selection ===
            if self.config.rank_mode == "variance":
                if self.config.use_random_qr_init:
                    warnings.warn(
                        "rank_mode='variance' is incompatible with use_random_qr_init; "
                        "using configured rank for random-QR init."
                    )
                else:
                    # Use profile_layer_spectrum to determine adaptive rank
                    if input_2d.numel() == 0:
                        # Edge case: empty batch
                        warnings.warn("Empty batch during basis init, using rank_min")
                        adaptive_rank = self.config.rank_min
                    else:
                        try:
                            spectrum_info = profile_layer_spectrum(
                                input_2d,
                                threshold=self.config.rank_variance_threshold,
                                return_full=False,
                            )
                            adaptive_rank = spectrum_info["adaptive_rank"]

                            # Clamp to [rank_min, rank_max]
                            adaptive_rank = max(
                                self.config.rank_min, min(adaptive_rank, self.config.rank_max)
                            )

                            # Also clamp to matrix rank (can't exceed min(B, N))
                            adaptive_rank = min(adaptive_rank, batch_size, in_features)

                            # Log when adaptive rank differs from configured rank
                            if adaptive_rank != self.config.rank:
                                print(
                                    f"[STACRLinear] Adaptive rank selection: configured rank={self.config.rank}, "
                                    f"adaptive rank={adaptive_rank} (threshold={self.config.rank_variance_threshold:.2%}, "
                                    f"energy={spectrum_info['energy_at_rank']:.2%})"
                                )

                            target_rank = adaptive_rank
                            self.effective_rank = adaptive_rank
                            self.layer_state["effective_rank"] = adaptive_rank
                            self.layer_state["allocated_rank"] = adaptive_rank
                            self.layer_state["active_rank"] = adaptive_rank

                            # Resize local basis buffer when this layer owns storage.
                            if not self.use_basis_sharing:
                                basis_device = (
                                    self.basis.device
                                    if self.basis is not None
                                    else self.linear.weight.device
                                )
                                new_basis = torch.zeros(
                                    in_features,
                                    target_rank,
                                    dtype=self.basis_dtype,
                                    device=basis_device,
                                )
                                self._buffers["basis"] = new_basis
                                self.basis = new_basis

                        except Exception as e:
                            warnings.warn(
                                f"Adaptive rank selection failed ({e}), using configured rank={self.config.rank}"
                            )
                            target_rank = self.config.rank

            if self.config.use_random_qr_init:
                target_rank = max(1, min(target_rank, in_features))
                qr_dtype = self.basis_dtype
                if qr_dtype in (torch.float16, torch.bfloat16):
                    qr_dtype = torch.float32
                random_mat = torch.randn(
                    in_features,
                    target_rank,
                    device=input_2d.device,
                    dtype=qr_dtype,
                )
                Q, _ = torch.linalg.qr(random_mat)
                if self.use_basis_sharing and self.basis_group_id is not None:
                    pool = get_basis_pool()
                    pool.update_basis(
                        self.basis_group_id, self.in_features, target_rank, Q.to(self.basis_dtype)
                    )
                else:
                    self.basis = Q.to(self.basis.dtype)
                self.basis_initialized = True
                self.layer_state["allocated_rank"] = target_rank
                self.layer_state["active_rank"] = min(
                    self.layer_state.get("active_rank", target_rank),
                    target_rank,
                )
                return

            compute_input = input_2d
            if input_2d.dtype in (torch.float16, torch.bfloat16):
                compute_input = input_2d.float()
            centered = compute_input

            # Choose SVD method based on feature dimension
            use_randomized = in_features > self.config.randomized_svd_threshold

            try:
                if use_randomized:
                    # Randomized SVD: faster for large features, lower workspace
                    # torch.svd_lowrank returns (U, S, V) where V is already transposed
                    q = target_rank + self.config.randomized_svd_oversampling
                    # q must be <= min(batch_size, in_features)
                    q = min(q, batch_size, in_features)
                    niter = self.config.randomized_svd_niter
                    U, S, V = torch.svd_lowrank(centered, q=q, niter=niter)
                    # V has shape (N, q), we want first target_rank columns
                    available_rank = min(V.shape[1], target_rank)
                    V_k = V[:, :available_rank]  # (N, available_rank)
                else:
                    # Full SVD: more accurate for small features
                    U, S, Vh = torch.linalg.svd(centered, full_matrices=False)
                    # Vh has shape (min(B, N), N)
                    # Top-R right singular vectors (columns of V = rows of Vh transposed)
                    available_rank = min(Vh.shape[0], target_rank)
                    V_k = Vh[:available_rank, :].t()  # (N, available_rank)

            except RuntimeError as e:
                # Fallback: random orthonormal initialization
                warnings.warn(f"SVD failed for basis init ({e}), using random orthonormal basis")
                Q, _ = torch.linalg.qr(
                    torch.randn(
                        in_features, target_rank, device=input_2d.device, dtype=compute_input.dtype
                    )
                )
                # Update basis (local or shared pool)
                if self.use_basis_sharing and self.basis_group_id is not None:
                    pool = get_basis_pool()
                    pool.update_basis(
                        self.basis_group_id, self.in_features, target_rank, Q.to(self.basis_dtype)
                    )
                else:
                    self.basis = Q.to(self.basis.dtype)
                self.basis_initialized = True
                self.layer_state["allocated_rank"] = target_rank
                self.layer_state["active_rank"] = min(
                    self.layer_state.get("active_rank", target_rank),
                    target_rank,
                )
                return

            # If we have fewer singular vectors than target rank, pad with random orthogonal vectors
            if available_rank < target_rank:
                # Generate additional orthogonal vectors
                padding_rank = target_rank - available_rank
                # Create random vectors orthogonal to existing basis
                random_vecs = torch.randn(
                    in_features, padding_rank, device=input_2d.device, dtype=compute_input.dtype
                )
                # Orthogonalize against existing vectors using Gram-Schmidt
                combined = torch.cat([V_k, random_vecs], dim=1)
                Q, _ = torch.linalg.qr(combined)
                V_k = Q[:, :target_rank]

            # Update basis (local or shared pool)
            if self.use_basis_sharing and self.basis_group_id is not None:
                pool = get_basis_pool()
                pool.update_basis(
                    self.basis_group_id, self.in_features, V_k.shape[1], V_k.to(self.basis_dtype)
                )
            else:
                self.basis = V_k.to(self.basis.dtype)
            self.basis_initialized = True
            self.layer_state["allocated_rank"] = V_k.shape[1]
            self.layer_state["active_rank"] = min(
                self.layer_state.get("active_rank", V_k.shape[1]),
                V_k.shape[1],
            )

    def _update_basis_oja(self, input_2d: torch.Tensor):
        """Update basis using Oja's algorithm (online PCA).

        Supports adaptive learning rate based on residual energy:
        - High residual -> higher LR for faster adaptation
        - Low residual energy = basis is good, use gentle updates
        """
        with torch.no_grad(), _disable_autocast(input_2d.device):
            base_lr = self.config.oja_learning_rate

            # Adaptive learning rate: scale by residual energy
            if self.config.adaptive_oja_lr:
                residual_energy = self.layer_state.get("last_residual_energy", 0.0)
                # Formula: lr = base_lr * (1 + residual_energy * scale)
                # High residual -> higher LR for faster adaptation
                adaptive_factor = 1.0 + residual_energy * self.config.adaptive_oja_scale
                lr = min(base_lr * adaptive_factor, self.config.adaptive_oja_max_lr)
            else:
                lr = base_lr

            # For each sample in batch, update basis
            # Oja's rule: B <- B + lr * (a @ a^T @ B - B @ (B^T @ a @ a^T @ B))
            # Simplified for batch: B <- B + lr * (A^T @ A @ B - B @ (B^T @ A^T @ A @ B))

            # Compute covariance update
            # cov_update = A^T @ A  # (N, N) - too expensive!

            # Instead, use streaming update:
            # For each sample a: B <- B + lr * (a @ (a^T @ B) - B @ (B^T @ a @ (a^T @ B)))
            #                  = B + lr * (a @ c^T - B @ (c @ c^T))
            # where c = B^T @ a

            compute_input = input_2d
            if input_2d.dtype in (torch.float16, torch.bfloat16):
                compute_input = input_2d.float()
            # Use effective_basis (local or from pool)
            current_basis = self.effective_basis
            B = current_basis.to(compute_input.dtype)  # (N, R)

            # Batch version: average update over samples
            # c = input_2d @ B  # (batch, R)
            # outer = input_2d.T @ c / batch_size  # (N, R) - approximate update direction
            # B_new = B + lr * (outer - B @ (c.T @ c / batch_size))

            batch_size = input_2d.shape[0]
            c = compute_input.mm(B)  # (batch, R)

            # Update direction
            update = compute_input.t().mm(c) / batch_size  # (N, R)
            correction = B.mm(c.t().mm(c) / batch_size)  # (N, R)

            B_new = B + lr * (update - correction)

            # Re-orthonormalize using MGS (faster) or QR (more stable)
            if self.config.use_mgs:
                Q = _orthonormalize_mgs(B_new)
            else:
                Q, _ = torch.linalg.qr(B_new)

            # Update basis (local or shared pool)
            current_basis = self.effective_basis
            Q_slice = Q[:, : current_basis.shape[1]].to(self.basis_dtype)
            if self.use_basis_sharing and self.basis_group_id is not None:
                pool = get_basis_pool()
                pool.update_basis(self.basis_group_id, self.in_features, Q_slice.shape[1], Q_slice)
            else:
                self.basis = Q_slice
            self.layer_state["basis_updates"] += 1
            self.layer_state["last_oja_lr"] = lr  # Track actual LR used
            if self.config.cache_basis_per_optimizer_step and self._get_accum_steps() > 1:
                self.layer_state["microbatch_cache_misses"] = (
                    self.layer_state.get("microbatch_cache_misses", 0) + 1
                )
            update_reason = self._consume_update_reason()
            if update_reason is not None:
                self.layer_state["last_update_reason"] = update_reason
                if update_reason.startswith("refresh_"):
                    self.layer_state["refresh_count_total"] = (
                        self.layer_state.get("refresh_count_total", 0) + 1
                    )
                    refresh_counts = self.layer_state.setdefault("refresh_count_by_reason", {})
                    refresh_counts[update_reason] = int(refresh_counts.get(update_reason, 0)) + 1
                    self.layer_state["_last_refresh_schedule_step"] = self._get_schedule_step()
                else:
                    self.layer_state["_last_non_refresh_schedule_step"] = self._get_schedule_step()

    def _should_update_basis(self) -> bool:
        """Determine if basis should be updated based on trigger mode.

        Includes:
        - freeze_basis_after_warmup: skip updates after warmup_steps
        - Lazy update optimization: skip updates when residual energy is low
        """
        step = self._get_schedule_step()
        trigger = self.config.update_trigger
        interval = self.config.update_interval
        energy_thresh = self.config.energy_threshold
        residual_energy = self.layer_state["last_residual_energy"]
        microbatch_index = self._get_microbatch_index()
        accum_steps = self._get_accum_steps()
        cache_enabled = bool(self.config.cache_basis_per_optimizer_step and accum_steps > 1)

        # freeze_basis_after_warmup - skip all updates after warmup
        # This allows basis to stabilize during warmup, then freeze for consistent behavior
        past_warmup = step >= self.config.warmup_steps
        if self.config.freeze_basis_after_warmup and past_warmup:
            refresh_trigger = getattr(self.config, "refresh_trigger", "off")
            refresh_period = max(1, int(getattr(self.config, "refresh_period", 1000)))
            refresh_energy_threshold = float(getattr(self.config, "refresh_energy_threshold", 0.1))
            fixed_refresh = _is_fixed_refresh_step(
                schedule_step=step,
                warmup_steps=self.config.warmup_steps,
                refresh_period=refresh_period,
                layer_index=self.layer_index,
                num_layers=self.num_layers,
            )
            energy_refresh = residual_energy > refresh_energy_threshold

            should_refresh = False
            refresh_reason = "off"
            if refresh_trigger == BasisRefreshTrigger.FIXED.value:
                should_refresh = fixed_refresh
                refresh_reason = "fixed"
            elif refresh_trigger == BasisRefreshTrigger.RESIDUAL_ENERGY.value:
                should_refresh = energy_refresh
                refresh_reason = "residual_energy"
            elif refresh_trigger == BasisRefreshTrigger.HYBRID.value:
                should_refresh = fixed_refresh or energy_refresh
                if fixed_refresh and energy_refresh:
                    refresh_reason = "hybrid_both"
                elif fixed_refresh:
                    refresh_reason = "fixed"
                elif energy_refresh:
                    refresh_reason = "residual_energy"

            if should_refresh:
                if cache_enabled and microbatch_index > 0:
                    self.layer_state["microbatch_cache_hits"] = (
                        self.layer_state.get("microbatch_cache_hits", 0) + 1
                    )
                    self.layer_state["last_skip_reason"] = "cache_hit"
                    return False
                self.layer_state["_last_update_schedule_step"] = step
                self._record_update_reason(f"refresh_{refresh_reason}")
                return True

            self.layer_state["frozen_updates"] = self.layer_state.get("frozen_updates", 0) + 1
            self.layer_state["last_skip_reason"] = "frozen_after_warmup"
            return False

        # Lazy update optimization: skip if basis is already good
        # (residual energy is low = basis captures activations well)
        if self.config.lazy_update_enabled:
            if residual_energy < self.config.lazy_update_threshold:
                self.layer_state["skipped_updates"] = self.layer_state.get("skipped_updates", 0) + 1
                self.layer_state["last_skip_reason"] = "lazy_update_skip"
                return False

        if trigger == "fixed":
            should_update = step > 0 and step % interval == 0
            if should_update:
                if cache_enabled and microbatch_index > 0:
                    self.layer_state["microbatch_cache_hits"] = (
                        self.layer_state.get("microbatch_cache_hits", 0) + 1
                    )
                    self.layer_state["last_skip_reason"] = "cache_hit"
                    return False
                self.layer_state["_last_update_schedule_step"] = step
                self._record_update_reason("fixed")
            return should_update

        elif trigger == "adaptive":
            should_update = residual_energy > energy_thresh
            if should_update:
                if cache_enabled and microbatch_index > 0:
                    self.layer_state["microbatch_cache_hits"] = (
                        self.layer_state.get("microbatch_cache_hits", 0) + 1
                    )
                    self.layer_state["last_skip_reason"] = "cache_hit"
                    return False
                self.layer_state["_last_update_schedule_step"] = step
                self._record_update_reason("adaptive")
            return should_update

        elif trigger == "hybrid":
            # Update if either condition is met
            fixed_trigger = step > 0 and step % interval == 0
            adaptive_trigger = residual_energy > energy_thresh
            should_update = fixed_trigger or adaptive_trigger
            if should_update:
                if cache_enabled and microbatch_index > 0:
                    self.layer_state["microbatch_cache_hits"] = (
                        self.layer_state.get("microbatch_cache_hits", 0) + 1
                    )
                    self.layer_state["last_skip_reason"] = "cache_hit"
                    return False
                self.layer_state["_last_update_schedule_step"] = step
                if fixed_trigger and adaptive_trigger:
                    self._record_update_reason("hybrid_both")
                elif fixed_trigger:
                    self._record_update_reason("fixed")
                else:
                    self._record_update_reason("adaptive")
            return should_update

        return False

    def get_effective_rank(self) -> int:
        """Return the allocated basis rank for the layer.

        This may differ from config.rank when:
        - rank_mode="variance": adaptive rank determined from spectrum
        - Basis not yet initialized: returns configured rank as target

        Returns:
            int: The effective rank (second dimension of basis buffer)
        """
        if self.basis_initialized:
            # For basis sharing, use effective_basis (from pool)
            effective = self.effective_basis
            if effective is not None:
                return effective.shape[1]
            # Fallback: local basis
            if self.basis is not None:
                return self.basis.shape[1]
        # Basis not yet initialized, return target rank
        return min(self.effective_rank, self.in_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with STAC-R compression."""
        if (
            not self.training
            or not torch.is_grad_enabled()
            or not self.linear.weight.requires_grad
            or x.numel() == 0
        ):
            return self.linear(x)
        sparse_only_mode = self.config.residual_mode == "sparse_only"
        is_recompute = False
        if self.training:
            try:
                from .adaptive_checkpointing import get_global_recomputation_guard

                is_recompute = get_global_recomputation_guard().is_recomputing
            except Exception:
                is_recompute = False

        # Initialize basis from first batch if needed
        if (
            self.training
            and not sparse_only_mode
            and not self.basis_initialized
            and self.use_basis_sharing
            and self.basis_group_id is not None
        ):
            pool = get_basis_pool()
            if pool.is_initialized(
                self.basis_group_id, self.in_features, min(self.effective_rank, self.in_features)
            ):
                self.basis_initialized = True

        if (
            self.training
            and not sparse_only_mode
            and not self.basis_initialized
            and not is_recompute
        ):
            input_2d = x.reshape(-1, self.in_features)
            with _layer_phase_context(self.layer_state, "basis_init"):
                self._init_basis_from_batch(input_2d)

        step_tensor = self.step_counter
        if is_recompute:
            step_value = max(int(self.step_counter.item()) - 1, 0)
            step_tensor = torch.tensor(
                [step_value],
                device=self.step_counter.device,
                dtype=self.step_counter.dtype,
            )
            self.layer_state["skipped_recompute"] = self.layer_state.get("skipped_recompute", 0) + 1

        schedule_step = self._get_forward_schedule_step()
        self.layer_state["_schedule_step"] = schedule_step

        # Call custom autograd function
        basis_for_forward = (
            None if sparse_only_mode else (self.effective_basis if self.basis_initialized else None)
        )
        if basis_for_forward is not None:
            allocated_rank = basis_for_forward.shape[1]
            self.layer_state["allocated_rank"] = allocated_rank
            if self.training and not is_recompute:
                if self.config.rank_mode == "fixed":
                    active_rank = allocated_rank
                    self.layer_state["last_rank_reason"] = "fixed"
                else:
                    active_rank = self._select_active_rank(schedule_step, allocated_rank)
                self.layer_state["active_rank"] = active_rank
            else:
                active_rank = max(
                    1,
                    min(int(self.layer_state.get("active_rank", allocated_rank)), allocated_rank),
                )
            if active_rank < allocated_rank:
                basis_for_forward = basis_for_forward[:, :active_rank]
        else:
            if self.training and not is_recompute:
                self.layer_state["active_rank"] = self.get_effective_rank()

        # Checkpoint replay must use the basis version of the original call,
        # even if an Oja update or a later forward has since replaced it.
        from .adaptive_checkpointing import get_global_recomputation_guard

        guard = get_global_recomputation_guard()
        replay_state = guard.replay_layer(self) if is_recompute else None
        if replay_state is not None:
            basis_for_forward, schedule_step, recorded_step = replay_state
            self.layer_state["_schedule_step"] = schedule_step
            step_tensor = self.step_counter.new_tensor([recorded_step])
        elif not is_recompute:
            guard.record_layer(self, (basis_for_forward, schedule_step, int(step_tensor.item())))

        # Prepare config dict with workspace info for buffer reuse
        config_dict = self.config.to_dict()
        if self.config.buffer_reuse:
            config_dict["_layer_id"] = self._layer_id
            config_dict["_in_features"] = self.in_features

        low_rank_only_semantics = (
            basis_for_forward is not None
            and not sparse_only_mode
            and schedule_step >= self.config.warmup_steps
            and self.basis_initialized
            and (
                self.config.residual_mode == "none"
                or (
                    self.config.post_warmup_residual_mode == "none"
                    and self.config.residual_warmup_steps > 0
                    and schedule_step >= self.config.residual_warmup_steps
                )
            )
        )
        use_low_rank_only_route = (
            low_rank_only_semantics and self.config.enable_low_rank_only_fast_path
        )

        if use_low_rank_only_route:
            with _layer_phase_context(self.layer_state, "stacr_forward_low_rank_only"):
                output = STACRLowRankOnlyFunction.apply(
                    x,
                    self.linear.weight,
                    self.linear.bias,
                    basis_for_forward,
                    config_dict,
                    step_tensor,
                    self.layer_state,
                )
            if self.training and not is_recompute:
                self.layer_state["b1_route_step_count"] = (
                    self.layer_state.get("b1_route_step_count", 0) + 1
                )
        else:
            if (
                self.training
                and not is_recompute
                and low_rank_only_semantics
                and not use_low_rank_only_route
            ):
                self.layer_state["b1_generic_fallback_count"] = (
                    self.layer_state.get("b1_generic_fallback_count", 0) + 1
                )
            with _layer_phase_context(self.layer_state, "stacr_forward_residual_correction"):
                output = STACRFunction.apply(
                    x,
                    self.linear.weight,
                    self.linear.bias,
                    basis_for_forward,
                    config_dict,
                    step_tensor,
                    self.layer_state,
                )

        # Update step counter
        if self.training and not is_recompute:
            self.step_counter += 1

            # Maybe update basis
            if not sparse_only_mode and self.basis_initialized and self._should_update_basis():
                input_2d = x.reshape(-1, self.in_features)
                with _layer_phase_context(self.layer_state, "basis_update"):
                    self._update_basis_oja(input_2d)

        return output

    def extra_repr(self) -> str:
        """Return string representation showing configuration and effective rank."""
        base = (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"rank={self.config.rank}"
        )

        # Show adaptive rank info when using variance mode
        if self.config.rank_mode != "fixed":
            effective_rank = self.get_effective_rank()
            active_rank = self.get_active_rank()
            base += f", rank_mode={self.config.rank_mode}"
            if self.basis_initialized and effective_rank != self.config.rank:
                base += f", effective_rank={effective_rank}"
            if self.basis_initialized and active_rank != effective_rank:
                base += f", active_rank={active_rank}"

        base += (
            f", residual_mode={self.config.residual_mode}, "
            f"update_interval={self.config.update_interval}"
        )

        return base


def set_stacr_runtime_state(
    model: nn.Module,
    optimizer_step: int,
    microbatch_index: int = 0,
    accum_steps: int = 1,
) -> None:
    """Propagate optimizer-step scheduling state to every STACR layer."""
    for module in model.modules():
        if isinstance(module, STACRLinear):
            module.set_runtime_state(
                optimizer_step=optimizer_step,
                microbatch_index=microbatch_index,
                accum_steps=accum_steps,
            )


# === Memory Accounting for STAC-R ===


def stacr_stored_bytes(
    batch: int,
    features: int,
    rank: int,
    residual_budget: float = 0.0,
    dtype: torch.dtype = torch.float32,
    coeff_bits: Optional[int] = None,
    residual_bits: Optional[int] = None,
    basis_dtype: Optional[torch.dtype] = None,
) -> dict:
    """Calculate storage bytes for STAC-R factors.

    This wrapper delegates to stacr.memory_accounting.stacr_stored_bytes to keep a
    single source of truth, then returns a simple dict for backward compatibility.
    """
    from .memory_accounting import stacr_stored_bytes as _stacr_stored_bytes

    resolved_basis_dtype = _resolve_basis_dtype(basis_dtype) if basis_dtype is not None else None
    breakdown = _stacr_stored_bytes(
        batch,
        features,
        rank,
        residual_budget,
        dtype,
        coeff_bits=coeff_bits,
        residual_bits=residual_bits,
        basis_dtype=resolved_basis_dtype,
        include_meta=False,
        include_basis=False,
    )
    details = breakdown.details
    dtype_bytes = torch.empty((), dtype=dtype).element_size()
    total = breakdown.total_bytes

    return {
        "total_bytes": total,
        "basis_bytes": details.get("basis_bytes_persistent", 0),
        "coefficients_bytes": details.get("coefficients_bytes", 0),
        "residual_bytes": details.get("residual_total_bytes", 0),
        "compression_ratio": (batch * features * dtype_bytes) / max(total, 1),
    }


def stacr_compression_ratio(
    batch: int,
    features: int,
    rank: int,
    residual_budget: float = 0.0,
    coeff_bits: Optional[int] = None,
    residual_bits: Optional[int] = None,
) -> float:
    """Calculate compression ratio for STAC-R.

    Returns:
        original_bytes / stored_bytes
    """
    result = stacr_stored_bytes(
        batch,
        features,
        rank,
        residual_budget,
        coeff_bits=coeff_bits,
        residual_bits=residual_bits,
    )
    return result["compression_ratio"]


class _TransposedProjection(nn.Module):
    """Linear facade over a Conv1D parameter stored in (input, output) order."""

    def __init__(self, module):
        super().__init__()
        self.source_weight = module.weight
        self.bias = getattr(module, "bias", None)
        self.in_features, self.out_features = module.weight.shape

    @property
    def weight(self):
        return self.source_weight.t()

    def forward(self, x):
        flat = x.reshape(-1, self.in_features)
        out = (
            flat.mm(self.source_weight)
            if self.bias is None
            else torch.addmm(self.bias, flat, self.source_weight)
        )
        return out.reshape(*x.shape[:-1], self.out_features)


def apply_adaptive_stacr(model, config=None, **options):
    """Wrap selected linear projections with coefficient-quantized adaptive STAC-R.

    Accepts an AdaptiveSTACRConfig or a dictionary. Selection options are shared
    with apply_stacr: surface, min_features, target_names, target_pattern,
    extra_names, exclude_names and trainable_only. Call set_stacr_runtime_state
    before each optimizer step when using optimizer-based scheduling.
    """
    from .surgery import select_target_names

    values = asdict(config) if isinstance(config, STACRConfig) else dict(config or {})
    values.update(options)
    selection_keys = {
        "surface",
        "min_features",
        "target_names",
        "target_pattern",
        "extra_names",
        "exclude_names",
        "trainable_only",
    }
    selection = {key: values.pop(key) for key in selection_keys if key in values}
    allow_empty = values.pop("allow_empty", False)
    cfg = STACRConfig(**values)
    if cfg.joint_qk:
        raise ValueError(
            "joint_qk requires the attention protocol adapter; it is not a generic linear selection option"
        )
    if not cfg.use_random_qr_init and (
        cfg.allow_unadapted_random_qr or cfg.min_adaptation_steps != 50
    ):
        raise ValueError("Random-QR adaptation controls require use_random_qr_init=True")
    if (
        cfg.use_random_qr_init
        and not cfg.allow_unadapted_random_qr
        and cfg.freeze_basis_after_warmup
        and cfg.warmup_steps < cfg.min_adaptation_steps
    ):
        raise ValueError(
            "Random-QR initialization needs warmup_steps >= min_adaptation_steps before freezing; "
            "set allow_unadapted_random_qr=True for an explicit random-basis control"
        )
    targets = select_target_names(model, **selection)
    if "" in targets:
        raise ValueError(
            "Use AdaptiveSTACRLinear.from_linear(model, config) for a standalone module"
        )
    if not targets and not allow_empty and selection.get("surface") != "none":
        raise ValueError("No trainable projections matched the adaptive STAC-R selection")
    planned = []
    replacements = {}
    for index, name in enumerate(targets):
        parent_name, _, leaf = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        module = getattr(parent, leaf)
        layer = replacements.get(id(module))
        if layer is None:
            layer = STACRLinear.from_linear(module, cfg, layer_index=index, num_layers=len(targets))
            layer.source_name = name
            replacements[id(module)] = layer
        planned.append((parent, leaf, layer))
    for parent, leaf, layer in planned:
        setattr(parent, leaf, layer)
    return list(replacements.values())


AdaptiveSTACRConfig = STACRConfig
AdaptiveSTACRLinear = STACRLinear
