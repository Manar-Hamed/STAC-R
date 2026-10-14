"""
Memory accounting for activation compression and tensor factorizations.

This module provides:
1. Pure byte calculators for SVD, TT, and Tucker decompositions
2. Planning/feasibility functions that return structured PlanResult
3. Explicit rho_max ceiling calculations
4. Standardized reason codes for infeasibility

Storage conventions:
- SVD: U (B x k), S (k,), Vh (k x N) - sigma as vector by default
- TT: cores G_i of shape (r_{i-1}, n_i, r_i) with r_0 = r_d = 1
- Tucker: core (R_1 x ... x R_d), factor matrices (n_i x R_i)

All metadata uses int64 (8 bytes) for conservative accounting.
"""

import torch
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Sequence, Dict, Any, Tuple, List, Union
import math


# =============================================================================
# CONSTANTS
# =============================================================================

META_INT_BYTES = 8  # int64 for metadata (conservative)


# =============================================================================
# ENUMS
# =============================================================================


class FeasibilityReason(Enum):
    """Standard reason codes for feasibility decisions."""

    FEASIBLE = "feasible"
    BUDGET_TOO_LOW_FOR_MIN_PARAMS = "budget_too_low_for_min_params"
    TARGET_RATIO_EXCEEDS_RHO_MAX = "target_ratio_exceeds_rho_max"
    INVALID_PARAMETERS = "invalid_parameters"
    EXCEEDS_METHOD_MAX_RANK = "exceeds_method_max_rank"
    UNSUPPORTED_SHAPE = "unsupported_shape"


class SigmaStorage(Enum):
    """SVD sigma storage conventions."""

    VECTOR = "vector"  # k elements (default)
    MATRIX = "matrix"  # k x k elements
    FOLDED = "folded"  # 0 elements (folded into U)


# =============================================================================
# DATACLASSES
# =============================================================================


@dataclass
class StorageBreakdown:
    """Detailed breakdown of storage requirements."""

    total_bytes: int
    payload_bytes: int
    meta_bytes: int
    persistent_cache_bytes: int = 0
    details: Dict[str, Any] = field(default_factory=dict)

    def __repr__(self):
        return (
            f"StorageBreakdown(total={self.total_bytes}, "
            f"payload={self.payload_bytes}, meta={self.meta_bytes})"
        )


@dataclass
class PlanResult:
    """
    Structured result from planning/feasibility checks.

    Contains all information needed to understand:
    - Whether compression is feasible
    - Why it's infeasible (if applicable)
    - The achieved compression ratio
    - Full storage breakdown
    """

    feasible: bool
    reason: FeasibilityReason
    original_bytes: int
    budget_bytes: Optional[int] = None
    stored_bytes: Optional[int] = None
    achieved_ratio: Optional[float] = None

    # Method-specific parameters
    k: Optional[int] = None  # SVD rank
    ranks: Optional[List[int]] = None  # TT/Tucker ranks
    rho_max: Optional[float] = None  # Maximum achievable ratio

    # Full breakdown
    breakdown: Optional[StorageBreakdown] = None

    # Additional context
    shape: Optional[Tuple[int, ...]] = None
    method: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "feasible": self.feasible,
            "reason": self.reason.value,
            "original_bytes": self.original_bytes,
            "budget_bytes": self.budget_bytes,
            "stored_bytes": self.stored_bytes,
            "achieved_ratio": self.achieved_ratio,
            "k": self.k,
            "ranks": self.ranks,
            "rho_max": self.rho_max,
            "shape": list(self.shape) if self.shape else None,
            "method": self.method,
            "breakdown": {
                "total_bytes": self.breakdown.total_bytes,
                "payload_bytes": self.breakdown.payload_bytes,
                "meta_bytes": self.breakdown.meta_bytes,
                "details": self.breakdown.details,
            }
            if self.breakdown
            else None,
        }


# =============================================================================
# PRIMITIVE FUNCTIONS
# =============================================================================


def bytes_per_elem(dtype: torch.dtype) -> int:
    """
    Get bytes per element for a PyTorch dtype.

    Uses manual mapping for common dtypes, falls back to dynamic for others.
    """
    # Manual mapping for common dtypes (fast, no allocation)
    dtype_bytes = {
        torch.float32: 4,
        torch.float64: 8,
        torch.float16: 2,
        torch.bfloat16: 2,
        torch.int8: 1,
        torch.int16: 2,
        torch.int32: 4,
        torch.int64: 8,
        torch.uint8: 1,
        torch.bool: 1,
        torch.complex64: 8,
        torch.complex128: 16,
    }

    if dtype in dtype_bytes:
        return dtype_bytes[dtype]

    # Fallback to dynamic (robust for new dtypes)
    return torch.tensor([], dtype=dtype).element_size()


def original_bytes(shape: Tuple[int, ...], dtype: torch.dtype) -> int:
    """
    Calculate bytes for dense storage of a tensor with given shape and dtype.

    Args:
        shape: Tensor shape tuple
        dtype: PyTorch dtype

    Returns:
        Number of bytes for dense storage
    """
    numel = 1
    for dim in shape:
        numel *= dim
    return numel * bytes_per_elem(dtype)


def original_bytes_from_tensor(tensor: torch.Tensor) -> int:
    """Calculate bytes for dense storage from an actual tensor."""
    return tensor.numel() * tensor.element_size()


def meta_bytes(
    shape_len: int,
    perm_len: int = 0,
    ranks_len: int = 0,
    extra_fields: int = 2,  # method ID + flags
    meta_int_bytes: int = META_INT_BYTES,
) -> int:
    """
    Calculate metadata bytes for decomposition storage.

    Args:
        shape_len: Length of shape vector
        perm_len: Length of permutation vector
        ranks_len: Length of rank vector
        extra_fields: Number of extra integer fields (method ID, flags)
        meta_int_bytes: Bytes per metadata integer (default: 8 for int64)

    Returns:
        Total metadata bytes
    """
    total_ints = shape_len + perm_len + ranks_len + extra_fields
    return total_ints * meta_int_bytes


# =============================================================================
# SVD STORAGE MODEL
# =============================================================================


def svd_stored_elements(
    batch: int, features: int, rank: int, store_sigma: SigmaStorage = SigmaStorage.VECTOR
) -> int:
    """
    Calculate stored elements for SVD decomposition.

    Storage: U (batch x rank), Vh (rank x features), sigma (varies)

    Args:
        batch: Batch dimension (B)
        features: Feature dimension (N)
        rank: SVD rank (k)
        store_sigma: How sigma is stored

    Returns:
        Number of floating-point elements stored
    """
    u_elems = batch * rank
    vh_elems = rank * features

    if store_sigma == SigmaStorage.VECTOR:
        sigma_elems = rank
    elif store_sigma == SigmaStorage.MATRIX:
        sigma_elems = rank * rank
    elif store_sigma == SigmaStorage.FOLDED:
        sigma_elems = 0  # Folded into U
    else:
        raise ValueError(f"Unknown sigma storage mode: {store_sigma}")

    return u_elems + vh_elems + sigma_elems


def svd_stored_bytes(
    batch: int,
    features: int,
    rank: int,
    dtype: torch.dtype,
    store_sigma: SigmaStorage = SigmaStorage.VECTOR,
    include_meta: bool = True,
) -> StorageBreakdown:
    """
    Calculate stored bytes for SVD decomposition.

    Args:
        batch: Batch dimension (B)
        features: Feature dimension (N)
        rank: SVD rank (k)
        dtype: Element dtype
        store_sigma: Sigma storage mode
        include_meta: Whether to include metadata bytes

    Returns:
        StorageBreakdown with detailed byte accounting
    """
    bpe = bytes_per_elem(dtype)

    u_elems = batch * rank
    vh_elems = rank * features

    if store_sigma == SigmaStorage.VECTOR:
        sigma_elems = rank
    elif store_sigma == SigmaStorage.MATRIX:
        sigma_elems = rank * rank
    elif store_sigma == SigmaStorage.FOLDED:
        sigma_elems = 0
    else:
        raise ValueError(f"Unknown sigma storage mode: {store_sigma}")

    total_elems = u_elems + vh_elems + sigma_elems
    payload_bytes = total_elems * bpe

    # Metadata: shape (2), rank (1), extra (2)
    mb = meta_bytes(shape_len=2, ranks_len=1, extra_fields=2) if include_meta else 0

    return StorageBreakdown(
        total_bytes=payload_bytes + mb,
        payload_bytes=payload_bytes,
        meta_bytes=mb,
        details={
            "U_bytes": u_elems * bpe,
            "S_bytes": sigma_elems * bpe,
            "Vh_bytes": vh_elems * bpe,
            "stored_elements": total_elems,
            "rank": rank,
            "sigma_mode": store_sigma.value,
        },
    )


def svd_rho_max(
    batch: int, features: int, store_sigma: SigmaStorage = SigmaStorage.VECTOR
) -> float:
    """
    Calculate maximum achievable compression ratio for SVD.

    The ceiling is achieved at k=1:
        rho_max = BN / (B + N + sigma_overhead)

    For vector sigma: rho_max = BN / (B + N + 1)
    For matrix sigma: rho_max = BN / (B + N + 1)  (k^2 = 1 at k=1)
    For folded: rho_max = BN / (B + N)

    Args:
        batch: Batch dimension (B)
        features: Feature dimension (N)
        store_sigma: Sigma storage mode

    Returns:
        Maximum compression ratio (original_elements / stored_elements at k=1)
    """
    original = batch * features
    stored_k1 = svd_stored_elements(batch, features, rank=1, store_sigma=store_sigma)
    return original / stored_k1


def svd_rank_from_ratio(
    batch: int, features: int, target_ratio: float, store_sigma: SigmaStorage = SigmaStorage.VECTOR
) -> Optional[int]:
    """
    Calculate SVD rank needed to achieve target compression ratio.

    Solving: BN / (k(B + N + sigma_factor)) = ratio
    For vector sigma: k = BN / (ratio * (B + N + 1))

    Args:
        batch: Batch dimension
        features: Feature dimension
        target_ratio: Target compression ratio
        store_sigma: Sigma storage mode

    Returns:
        Required rank (floored), or None if infeasible
    """
    if target_ratio <= 0:
        return None

    rho_max = svd_rho_max(batch, features, store_sigma)
    if target_ratio > rho_max:
        return None

    original = batch * features

    # For vector sigma: stored = k(B + N + 1)
    # ratio = original / stored = BN / (k(B + N + 1))
    # k = BN / (ratio * (B + N + 1))

    if store_sigma == SigmaStorage.VECTOR:
        denominator = batch + features + 1
        k = original / (target_ratio * denominator)
    elif store_sigma == SigmaStorage.MATRIX:
        # stored = k(B + N) + k^2, solve quadratic for k
        b = batch + features
        disc = b * b + 4 * (original / target_ratio)
        k = (-b + math.sqrt(disc)) / 2.0
    elif store_sigma == SigmaStorage.FOLDED:
        denominator = batch + features
        k = original / (target_ratio * denominator)
    else:
        return None

    k_int = int(math.floor(k))

    # Validate
    if k_int < 1:
        return None
    if k_int > min(batch, features):
        return min(batch, features)

    return k_int


def svd_rank_from_budget(
    batch: int,
    features: int,
    budget_bytes: int,
    dtype: torch.dtype,
    store_sigma: SigmaStorage = SigmaStorage.VECTOR,
    include_meta: bool = True,
) -> Optional[int]:
    """
    Calculate maximum SVD rank that fits within budget.

    Args:
        batch: Batch dimension
        features: Feature dimension
        budget_bytes: Byte budget
        dtype: Element dtype
        store_sigma: Sigma storage mode
        include_meta: Whether to account for metadata

    Returns:
        Maximum feasible rank, or None if even k=1 exceeds budget
    """
    max_k = min(batch, features)

    # Binary search for maximum feasible rank
    low, high = 1, max_k
    best_k = None

    while low <= high:
        mid = (low + high) // 2
        breakdown = svd_stored_bytes(batch, features, mid, dtype, store_sigma, include_meta)

        if breakdown.total_bytes <= budget_bytes:
            best_k = mid
            low = mid + 1
        else:
            high = mid - 1

    return best_k


# =============================================================================
# TT STORAGE MODEL
# =============================================================================


def tt_stored_elements(dims: Sequence[int], tt_ranks: Sequence[int]) -> int:
    """
    Calculate stored elements for TT decomposition.

    TT cores G_k have shape (r_{k-1}, n_k, r_k).
    Total elements = sum_k(r_{k-1} * n_k * r_k)

    Args:
        dims: Tensor dimensions (n_1, ..., n_d)
        tt_ranks: TT ranks (r_0, r_1, ..., r_d) with r_0 = r_d = 1

    Returns:
        Number of floating-point elements stored
    """
    if len(tt_ranks) != len(dims) + 1:
        raise ValueError(
            f"TT ranks length {len(tt_ranks)} must be dims length + 1 ({len(dims) + 1})"
        )

    total = 0
    for i, n in enumerate(dims):
        r_left = tt_ranks[i]
        r_right = tt_ranks[i + 1]
        total += r_left * n * r_right

    return total


def tt_stored_bytes(
    dims: Sequence[int], tt_ranks: Sequence[int], dtype: torch.dtype, include_meta: bool = True
) -> StorageBreakdown:
    """
    Calculate stored bytes for TT decomposition.

    Args:
        dims: Tensor dimensions
        tt_ranks: TT ranks
        dtype: Element dtype
        include_meta: Whether to include metadata bytes

    Returns:
        StorageBreakdown with detailed accounting
    """
    bpe = bytes_per_elem(dtype)
    total_elems = tt_stored_elements(dims, tt_ranks)
    payload_bytes = total_elems * bpe

    # Metadata: shape + ranks + extra
    mb = (
        meta_bytes(shape_len=len(dims), ranks_len=len(tt_ranks), extra_fields=2)
        if include_meta
        else 0
    )

    # Core-by-core breakdown
    core_sizes = []
    for i, n in enumerate(dims):
        r_left = tt_ranks[i]
        r_right = tt_ranks[i + 1]
        core_sizes.append(r_left * n * r_right)

    return StorageBreakdown(
        total_bytes=payload_bytes + mb,
        payload_bytes=payload_bytes,
        meta_bytes=mb,
        details={
            "core_sizes": core_sizes,
            "core_bytes": [s * bpe for s in core_sizes],
            "stored_elements": total_elems,
            "dims": list(dims),
            "tt_ranks": list(tt_ranks),
        },
    )


def tt_uniform_ranks(dims: Sequence[int], rank: int) -> List[int]:
    """
    Generate uniform TT ranks with boundary constraints.

    Args:
        dims: Tensor dimensions
        rank: Internal rank (capped by dimension constraints)

    Returns:
        TT rank list [1, r, r, ..., r, 1]
    """
    d = len(dims)
    ranks = [1]  # r_0 = 1

    for i in range(d - 1):
        # Rank can't exceed product of left dims or right dims
        max_left = 1
        for j in range(i + 1):
            max_left *= dims[j]
        max_right = 1
        for j in range(i + 1, d):
            max_right *= dims[j]

        r_i = min(rank, max_left, max_right)
        ranks.append(r_i)

    ranks.append(1)  # r_d = 1
    return ranks


def tt_rank_from_budget(
    dims: Sequence[int], budget_bytes: int, dtype: torch.dtype, include_meta: bool = True
) -> Optional[Tuple[int, List[int]]]:
    """
    Find maximum uniform TT rank that fits budget.

    Args:
        dims: Tensor dimensions
        budget_bytes: Byte budget
        dtype: Element dtype
        include_meta: Whether to include metadata

    Returns:
        Tuple of (uniform_rank, tt_ranks list) or None if infeasible
    """
    # Binary search on uniform rank
    low, high = 1, max(dims)
    best_rank = None
    best_tt_ranks = None

    while low <= high:
        mid = (low + high) // 2
        tt_ranks = tt_uniform_ranks(dims, mid)
        breakdown = tt_stored_bytes(dims, tt_ranks, dtype, include_meta)

        if breakdown.total_bytes <= budget_bytes:
            best_rank = mid
            best_tt_ranks = tt_ranks
            low = mid + 1
        else:
            high = mid - 1

    if best_rank is None:
        return None

    return best_rank, best_tt_ranks


# =============================================================================
# TUCKER STORAGE MODEL
# =============================================================================


def tucker_stored_elements(dims: Sequence[int], ranks: Sequence[int]) -> int:
    """
    Calculate stored elements for Tucker decomposition.

    Tucker storage: core (prod(ranks)) + factor matrices (sum(n_i * r_i))

    Args:
        dims: Tensor dimensions (n_1, ..., n_d)
        ranks: Tucker ranks (r_1, ..., r_d)

    Returns:
        Number of floating-point elements stored
    """
    if len(dims) != len(ranks):
        raise ValueError(f"Ranks length {len(ranks)} must match dims length {len(dims)}")

    # Core: prod(ranks)
    core_elems = 1
    for r in ranks:
        core_elems *= r

    # Factor matrices: sum(n_i * r_i)
    factors_elems = sum(n * r for n, r in zip(dims, ranks))

    return core_elems + factors_elems


def tucker_stored_bytes(
    dims: Sequence[int], ranks: Sequence[int], dtype: torch.dtype, include_meta: bool = True
) -> StorageBreakdown:
    """
    Calculate stored bytes for Tucker decomposition.

    Args:
        dims: Tensor dimensions
        ranks: Tucker ranks
        dtype: Element dtype
        include_meta: Whether to include metadata

    Returns:
        StorageBreakdown with detailed accounting
    """
    if len(dims) != len(ranks):
        raise ValueError(f"Ranks length {len(ranks)} must match dims length {len(dims)}")

    bpe = bytes_per_elem(dtype)

    # Core
    core_elems = 1
    for r in ranks:
        core_elems *= r

    # Factor matrices
    factor_elems = [n * r for n, r in zip(dims, ranks)]
    factors_total = sum(factor_elems)

    total_elems = core_elems + factors_total
    payload_bytes = total_elems * bpe

    # Metadata: shape + ranks + extra
    mb = (
        meta_bytes(shape_len=len(dims), ranks_len=len(ranks), extra_fields=2) if include_meta else 0
    )

    return StorageBreakdown(
        total_bytes=payload_bytes + mb,
        payload_bytes=payload_bytes,
        meta_bytes=mb,
        details={
            "core_elements": core_elems,
            "core_bytes": core_elems * bpe,
            "factor_elements": factor_elems,
            "factor_bytes": [e * bpe for e in factor_elems],
            "stored_elements": total_elems,
            "dims": list(dims),
            "ranks": list(ranks),
        },
    )


def tucker_uniform_ranks(dims: Sequence[int], rank: int) -> List[int]:
    """
    Generate uniform Tucker ranks, capped by dimension sizes.

    Args:
        dims: Tensor dimensions
        rank: Target rank for all modes

    Returns:
        Tucker rank list [min(r, n_i) for each dim]
    """
    return [min(rank, n) for n in dims]


def tucker_proportional_ranks(dims: Sequence[int], alpha: float, min_rank: int = 1) -> List[int]:
    """
    Generate proportional Tucker ranks: r_i = ceil(alpha * n_i).

    This policy sets ranks proportional to dimension sizes, which
    is often better balanced than uniform ranks for varying dimensions.

    Args:
        dims: Tensor dimensions
        alpha: Proportion factor (0 < alpha <= 1)
        min_rank: Minimum rank per mode (default: 1)

    Returns:
        Tucker rank list [max(min_rank, ceil(alpha * n_i)) for each dim]
    """
    import math

    return [max(min_rank, min(n, math.ceil(alpha * n))) for n in dims]


def tucker_proportional_ranks_from_budget(
    dims: Sequence[int], budget_bytes: int, dtype: torch.dtype, include_meta: bool = True
) -> Optional[Tuple[float, List[int]]]:
    """
    Find maximum alpha for proportional Tucker ranks that fits budget.

    Binary search on alpha in [0, 1] to maximize ranks while staying
    within budget.

    Args:
        dims: Tensor dimensions
        budget_bytes: Byte budget
        dtype: Element dtype
        include_meta: Whether to include metadata

    Returns:
        Tuple of (alpha, ranks list) or None if infeasible
    """
    # Check if even minimal ranks fit
    min_ranks = [1] * len(dims)
    min_breakdown = tucker_stored_bytes(dims, min_ranks, dtype, include_meta)
    if min_breakdown.total_bytes > budget_bytes:
        return None

    # Binary search on alpha (100 steps for 0.01 precision)
    best_alpha = 0.0
    best_ranks = min_ranks

    for alpha_int in range(1, 101):
        alpha = alpha_int / 100.0
        ranks = tucker_proportional_ranks(dims, alpha)
        breakdown = tucker_stored_bytes(dims, ranks, dtype, include_meta)

        if breakdown.total_bytes <= budget_bytes:
            best_alpha = alpha
            best_ranks = ranks
        else:
            break

    return best_alpha, best_ranks


def tucker_rank_from_budget(
    dims: Sequence[int], budget_bytes: int, dtype: torch.dtype, include_meta: bool = True
) -> Optional[Tuple[int, List[int]]]:
    """
    Find maximum uniform Tucker rank that fits budget.

    Note: Core size grows as r^d, so this can become infeasible quickly.

    Args:
        dims: Tensor dimensions
        budget_bytes: Byte budget
        dtype: Element dtype
        include_meta: Whether to include metadata

    Returns:
        Tuple of (uniform_rank, ranks list) or None if infeasible
    """
    # Binary search on uniform rank
    low, high = 1, min(dims)
    best_rank = None
    best_ranks = None

    while low <= high:
        mid = (low + high) // 2
        ranks = tucker_uniform_ranks(dims, mid)
        breakdown = tucker_stored_bytes(dims, ranks, dtype, include_meta)

        if breakdown.total_bytes <= budget_bytes:
            best_rank = mid
            best_ranks = ranks
            low = mid + 1
        else:
            high = mid - 1

    if best_rank is None:
        return None

    return best_rank, best_ranks


# =============================================================================
# PLANNING FUNCTIONS
# =============================================================================


def plan_svd(
    batch: int,
    features: int,
    dtype: torch.dtype,
    budget_bytes: Optional[int] = None,
    target_ratio: Optional[float] = None,
    store_sigma: SigmaStorage = SigmaStorage.VECTOR,
    include_meta: bool = True,
) -> PlanResult:
    """
    Plan SVD compression under budget or ratio constraint.

    Exactly one of budget_bytes or target_ratio must be specified.

    Args:
        batch: Batch dimension (B)
        features: Feature dimension (N)
        dtype: Element dtype
        budget_bytes: Byte budget constraint
        target_ratio: Target compression ratio constraint
        store_sigma: Sigma storage mode
        include_meta: Whether to include metadata

    Returns:
        PlanResult with feasibility, chosen rank, and breakdown
    """
    shape = (batch, features)
    orig_bytes = original_bytes(shape, dtype)

    # Compute rho_max
    rho_max = svd_rho_max(batch, features, store_sigma)

    # Convert target_ratio to budget if needed
    if target_ratio is not None and budget_bytes is None:
        if target_ratio > rho_max:
            return PlanResult(
                feasible=False,
                reason=FeasibilityReason.TARGET_RATIO_EXCEEDS_RHO_MAX,
                original_bytes=orig_bytes,
                budget_bytes=None,
                rho_max=rho_max,
                shape=shape,
                method="svd",
            )
        budget_bytes = int(math.floor(orig_bytes / target_ratio))

    if budget_bytes is None:
        return PlanResult(
            feasible=False,
            reason=FeasibilityReason.INVALID_PARAMETERS,
            original_bytes=orig_bytes,
            shape=shape,
            method="svd",
        )

    # Check if even k=1 fits
    min_breakdown = svd_stored_bytes(batch, features, 1, dtype, store_sigma, include_meta)
    if min_breakdown.total_bytes > budget_bytes:
        return PlanResult(
            feasible=False,
            reason=FeasibilityReason.BUDGET_TOO_LOW_FOR_MIN_PARAMS,
            original_bytes=orig_bytes,
            budget_bytes=budget_bytes,
            stored_bytes=min_breakdown.total_bytes,
            rho_max=rho_max,
            k=1,
            breakdown=min_breakdown,
            shape=shape,
            method="svd",
        )

    # Find maximum rank within budget
    k = svd_rank_from_budget(batch, features, budget_bytes, dtype, store_sigma, include_meta)

    if k is None:
        return PlanResult(
            feasible=False,
            reason=FeasibilityReason.BUDGET_TOO_LOW_FOR_MIN_PARAMS,
            original_bytes=orig_bytes,
            budget_bytes=budget_bytes,
            rho_max=rho_max,
            shape=shape,
            method="svd",
        )

    breakdown = svd_stored_bytes(batch, features, k, dtype, store_sigma, include_meta)
    achieved_ratio = orig_bytes / breakdown.total_bytes

    return PlanResult(
        feasible=True,
        reason=FeasibilityReason.FEASIBLE,
        original_bytes=orig_bytes,
        budget_bytes=budget_bytes,
        stored_bytes=breakdown.total_bytes,
        achieved_ratio=achieved_ratio,
        k=k,
        rho_max=rho_max,
        breakdown=breakdown,
        shape=shape,
        method="svd",
    )


def plan_tt(
    dims: Sequence[int],
    dtype: torch.dtype,
    budget_bytes: Optional[int] = None,
    target_ratio: Optional[float] = None,
    explicit_ranks: Optional[Sequence[int]] = None,
    include_meta: bool = True,
) -> PlanResult:
    """
    Plan TT compression under budget or ratio constraint.

    Supports:
    - Uniform rank selection (budget_bytes or target_ratio)
    - Explicit ranks (explicit_ranks)

    Args:
        dims: Tensor dimensions
        dtype: Element dtype
        budget_bytes: Byte budget
        target_ratio: Target compression ratio
        explicit_ranks: Explicit TT ranks (if provided, just check feasibility)
        include_meta: Whether to include metadata

    Returns:
        PlanResult with feasibility, chosen ranks, and breakdown
    """
    dims = tuple(dims)
    orig_bytes = original_bytes(dims, dtype)

    # Handle explicit ranks
    if explicit_ranks is not None:
        explicit_ranks = list(explicit_ranks)
        breakdown = tt_stored_bytes(dims, explicit_ranks, dtype, include_meta)
        achieved_ratio = orig_bytes / breakdown.total_bytes

        feasible = True
        reason = FeasibilityReason.FEASIBLE

        if budget_bytes is not None and breakdown.total_bytes > budget_bytes:
            feasible = False
            reason = FeasibilityReason.BUDGET_TOO_LOW_FOR_MIN_PARAMS

        return PlanResult(
            feasible=feasible,
            reason=reason,
            original_bytes=orig_bytes,
            budget_bytes=budget_bytes,
            stored_bytes=breakdown.total_bytes,
            achieved_ratio=achieved_ratio,
            ranks=explicit_ranks,
            breakdown=breakdown,
            shape=dims,
            method="tt",
        )

    # Convert target_ratio to budget
    if target_ratio is not None and budget_bytes is None:
        budget_bytes = int(math.floor(orig_bytes / target_ratio))

    if budget_bytes is None:
        return PlanResult(
            feasible=False,
            reason=FeasibilityReason.INVALID_PARAMETERS,
            original_bytes=orig_bytes,
            shape=dims,
            method="tt",
        )

    # Check if minimal ranks (all 1s) fit
    min_ranks = [1] * (len(dims) + 1)
    min_breakdown = tt_stored_bytes(dims, min_ranks, dtype, include_meta)

    if min_breakdown.total_bytes > budget_bytes:
        return PlanResult(
            feasible=False,
            reason=FeasibilityReason.BUDGET_TOO_LOW_FOR_MIN_PARAMS,
            original_bytes=orig_bytes,
            budget_bytes=budget_bytes,
            stored_bytes=min_breakdown.total_bytes,
            ranks=min_ranks,
            breakdown=min_breakdown,
            shape=dims,
            method="tt",
        )

    # Find maximum uniform rank
    result = tt_rank_from_budget(dims, budget_bytes, dtype, include_meta)

    if result is None:
        return PlanResult(
            feasible=False,
            reason=FeasibilityReason.BUDGET_TOO_LOW_FOR_MIN_PARAMS,
            original_bytes=orig_bytes,
            budget_bytes=budget_bytes,
            shape=dims,
            method="tt",
        )

    uniform_rank, tt_ranks = result
    breakdown = tt_stored_bytes(dims, tt_ranks, dtype, include_meta)
    achieved_ratio = orig_bytes / breakdown.total_bytes

    return PlanResult(
        feasible=True,
        reason=FeasibilityReason.FEASIBLE,
        original_bytes=orig_bytes,
        budget_bytes=budget_bytes,
        stored_bytes=breakdown.total_bytes,
        achieved_ratio=achieved_ratio,
        ranks=tt_ranks,
        breakdown=breakdown,
        shape=dims,
        method="tt",
    )


def plan_tucker(
    dims: Sequence[int],
    dtype: torch.dtype,
    budget_bytes: Optional[int] = None,
    target_ratio: Optional[float] = None,
    explicit_ranks: Optional[Sequence[int]] = None,
    include_meta: bool = True,
) -> PlanResult:
    """
    Plan Tucker compression under budget or ratio constraint.

    Warning: Tucker core grows as r^d, making it infeasible for high-order tensors.

    Args:
        dims: Tensor dimensions
        dtype: Element dtype
        budget_bytes: Byte budget
        target_ratio: Target compression ratio
        explicit_ranks: Explicit Tucker ranks
        include_meta: Whether to include metadata

    Returns:
        PlanResult with feasibility, chosen ranks, and breakdown
    """
    dims = tuple(dims)
    orig_bytes = original_bytes(dims, dtype)

    # Warn about inefficient Tucker for degenerate dimensions
    if min(dims) == 1:
        import warnings

        warnings.warn(f"Tucker may be inefficient for shapes with dimension=1: {dims}")

    # Handle explicit ranks
    if explicit_ranks is not None:
        explicit_ranks = list(explicit_ranks)

        # Validate ranks
        for i, (n, r) in enumerate(zip(dims, explicit_ranks)):
            if r < 1 or r > n:
                return PlanResult(
                    feasible=False,
                    reason=FeasibilityReason.INVALID_PARAMETERS,
                    original_bytes=orig_bytes,
                    ranks=explicit_ranks,
                    shape=dims,
                    method="tucker",
                )

        breakdown = tucker_stored_bytes(dims, explicit_ranks, dtype, include_meta)
        achieved_ratio = orig_bytes / breakdown.total_bytes

        feasible = True
        reason = FeasibilityReason.FEASIBLE

        if budget_bytes is not None and breakdown.total_bytes > budget_bytes:
            feasible = False
            reason = FeasibilityReason.BUDGET_TOO_LOW_FOR_MIN_PARAMS

        return PlanResult(
            feasible=feasible,
            reason=reason,
            original_bytes=orig_bytes,
            budget_bytes=budget_bytes,
            stored_bytes=breakdown.total_bytes,
            achieved_ratio=achieved_ratio,
            ranks=explicit_ranks,
            breakdown=breakdown,
            shape=dims,
            method="tucker",
        )

    # Convert target_ratio to budget
    if target_ratio is not None and budget_bytes is None:
        budget_bytes = int(math.floor(orig_bytes / target_ratio))

    if budget_bytes is None:
        return PlanResult(
            feasible=False,
            reason=FeasibilityReason.INVALID_PARAMETERS,
            original_bytes=orig_bytes,
            shape=dims,
            method="tucker",
        )

    # Check if minimal ranks (all 1s) fit
    min_ranks = [1] * len(dims)
    min_breakdown = tucker_stored_bytes(dims, min_ranks, dtype, include_meta)

    if min_breakdown.total_bytes > budget_bytes:
        return PlanResult(
            feasible=False,
            reason=FeasibilityReason.BUDGET_TOO_LOW_FOR_MIN_PARAMS,
            original_bytes=orig_bytes,
            budget_bytes=budget_bytes,
            stored_bytes=min_breakdown.total_bytes,
            ranks=min_ranks,
            breakdown=min_breakdown,
            shape=dims,
            method="tucker",
        )

    # Find maximum uniform rank
    result = tucker_rank_from_budget(dims, budget_bytes, dtype, include_meta)

    if result is None:
        return PlanResult(
            feasible=False,
            reason=FeasibilityReason.BUDGET_TOO_LOW_FOR_MIN_PARAMS,
            original_bytes=orig_bytes,
            budget_bytes=budget_bytes,
            shape=dims,
            method="tucker",
        )

    uniform_rank, tucker_ranks = result
    breakdown = tucker_stored_bytes(dims, tucker_ranks, dtype, include_meta)
    achieved_ratio = orig_bytes / breakdown.total_bytes

    return PlanResult(
        feasible=True,
        reason=FeasibilityReason.FEASIBLE,
        original_bytes=orig_bytes,
        budget_bytes=budget_bytes,
        stored_bytes=breakdown.total_bytes,
        achieved_ratio=achieved_ratio,
        ranks=tucker_ranks,
        breakdown=breakdown,
        shape=dims,
        method="tucker",
    )


# =============================================================================
# STAC-R STORAGE MODEL
# =============================================================================


def stacr_stored_elements(
    batch: int,
    features: int,
    rank: int,
    residual_budget: float = 0.0,
) -> int:
    """
    Calculate stored elements for STAC-R compression.

    STAC-R storage (per batch):
    - Coefficients: batch x rank
    - Residual values (sparse): batch x features x residual_budget
    - Residual indices: batch x features x residual_budget (int64)

    Note: Basis (features x rank) is persistent across batches and not counted
    in per-batch storage.

    Args:
        batch: Batch dimension (B)
        features: Feature dimension (N)
        rank: Subspace rank (R)
        residual_budget: Fraction of activation stored as residual (0 for STAC)

    Returns:
        Number of float-equivalent elements stored per batch (indices counted as 2 floats)
    """
    # Coefficients: B x R
    coeff_elems = batch * rank

    # Residual (sparse): values + indices
    residual_count = int(batch * features * residual_budget)
    # Values are floats, indices are int64 (~2 float32 equivalents)
    residual_elems = residual_count * 3

    return coeff_elems + residual_elems


def stacr_stored_bytes(
    batch: int,
    features: int,
    rank: int,
    residual_budget: float = 0.0,
    dtype: torch.dtype = torch.float32,
    coeff_bits: Optional[int] = None,
    residual_bits: Optional[int] = None,
    basis_dtype: Optional[torch.dtype] = None,
    include_meta: bool = True,
    include_basis: bool = False,
) -> StorageBreakdown:
    """
    Estimate a nominal STAC-R payload with conservative int64 residual indices.

    This analytical model excludes quantization scales, selector chunk/block
    rounding, data-dependent threshold counts, CSR/delta index compression and
    backward workspaces. It is not an allocator measurement or an exact tensor
    inventory. Basis storage is reported separately by default.

    Args:
        batch: Batch dimension (B)
        features: Feature dimension (N)
        rank: Subspace rank (R)
        residual_budget: Fraction of activation stored as residual (0 for STAC)
        dtype: Element dtype
        include_meta: Whether to include metadata bytes
        include_basis: Whether to include basis in storage (usually False since persistent)

    Returns:
        StorageBreakdown with detailed byte accounting
    """
    bpe = bytes_per_elem(dtype)
    basis_dtype = basis_dtype or dtype
    basis_bpe = bytes_per_elem(basis_dtype)
    coeff_bits = coeff_bits or (bpe * 8)
    residual_bits = residual_bits or (bpe * 8)

    def _bits_to_bytes(count: int, bits: int) -> int:
        return int(math.ceil(count * bits / 8.0))

    # Coefficients: B x R
    coeff_elems = batch * rank
    coeff_bytes = _bits_to_bytes(coeff_elems, coeff_bits)

    # Residual (sparse)
    residual_count = int(batch * features * residual_budget)
    residual_value_bytes = _bits_to_bytes(residual_count, residual_bits)
    residual_index_bytes = residual_count * 8  # int64 indices
    residual_bytes = residual_value_bytes + residual_index_bytes

    # Basis (optional - usually persistent across batches)
    basis_bytes = 0
    if include_basis:
        basis_bytes = features * rank * basis_bpe

    payload_bytes = coeff_bytes + residual_bytes + basis_bytes

    # Metadata: shape (2), rank (1), residual_count (1), extra (2)
    mb = meta_bytes(shape_len=2, ranks_len=1, extra_fields=3) if include_meta else 0

    return StorageBreakdown(
        total_bytes=payload_bytes + mb,
        payload_bytes=payload_bytes,
        meta_bytes=mb,
        persistent_cache_bytes=features * rank * basis_bpe if not include_basis else 0,
        details={
            "accounting_model": "nominal_payload_int64_indices",
            "includes_quantization_scales": False,
            "includes_selector_rounding": False,
            "includes_workspace": False,
            "coefficients_bytes": coeff_bytes,
            "coefficients_bits": coeff_bits,
            "residual_value_bytes": residual_value_bytes,
            "residual_index_bytes": residual_index_bytes,
            "residual_total_bytes": residual_bytes,
            "residual_bits": residual_bits,
            "basis_bytes": basis_bytes,
            "basis_bytes_persistent": features * rank * basis_bpe,
            "stored_elements": coeff_elems + residual_count * 3,
            "rank": rank,
            "residual_budget": residual_budget,
            "residual_count": residual_count,
        },
    )


def stacr_ckpt_stored_bytes(
    batch_tokens: int,
    hidden_size: int,
    rank: int,
    residual_budget: float = 0.0,
    dtype: torch.dtype = torch.float32,
    coeff_bits: Optional[int] = None,
    residual_bits: Optional[int] = None,
    basis_dtype: Optional[torch.dtype] = None,
    include_meta: bool = True,
    include_basis: bool = False,
) -> StorageBreakdown:
    """
    Calculate stored bytes for C1 checkpoint-boundary compression.

    This is a semantic wrapper over ``stacr_stored_bytes`` that makes the
    checkpoint-boundary surface explicit at call sites.
    """
    return stacr_stored_bytes(
        batch_tokens,
        hidden_size,
        rank,
        residual_budget=residual_budget,
        dtype=dtype,
        coeff_bits=coeff_bits,
        residual_bits=residual_bits,
        basis_dtype=basis_dtype,
        include_meta=include_meta,
        include_basis=include_basis,
    )


def stacr_rho_max(
    batch: int,
    features: int,
    residual_budget: float = 0.0,
    dtype: torch.dtype = torch.float32,
    coeff_bits: Optional[int] = None,
    residual_bits: Optional[int] = None,
) -> float:
    """
    Calculate maximum achievable compression ratio for STAC-R.

    The ceiling is achieved at rank=1:
        rho_max = BN / (B*1 + residual_elements)

    For STAC (no residual): rho_max = N (coefficient storage dominates)
    For STAC-R: rho_max depends on residual_budget

    Args:
        batch: Batch dimension (B)
        features: Feature dimension (N)
        residual_budget: Fraction of activation stored as residual

    Returns:
        Maximum compression ratio (original_elements / stored_elements at rank=1)
    """
    original = batch * features * bytes_per_elem(dtype)
    coeff_bits = coeff_bits or (bytes_per_elem(dtype) * 8)
    residual_bits = residual_bits or (bytes_per_elem(dtype) * 8)

    # At rank=1: coefficients = B, residual = B*N*budget
    coeff_at_r1 = batch
    residual_at_r1 = int(batch * features * residual_budget)

    coeff_bytes = int(math.ceil(coeff_at_r1 * coeff_bits / 8.0))
    residual_value_bytes = int(math.ceil(residual_at_r1 * residual_bits / 8.0))
    residual_index_bytes = residual_at_r1 * 8
    total_at_r1 = coeff_bytes + residual_value_bytes + residual_index_bytes

    return original / max(total_at_r1, 1)


def stacr_rank_from_budget(
    batch: int,
    features: int,
    budget_bytes: int,
    residual_budget: float = 0.0,
    dtype: torch.dtype = torch.float32,
    coeff_bits: Optional[int] = None,
    residual_bits: Optional[int] = None,
    basis_dtype: Optional[torch.dtype] = None,
    include_meta: bool = True,
) -> Optional[int]:
    """
    Calculate maximum STAC-R rank that fits within budget.

    Args:
        batch: Batch dimension
        features: Feature dimension
        budget_bytes: Byte budget
        residual_budget: Fraction of activation stored as residual
        dtype: Element dtype
        include_meta: Whether to account for metadata

    Returns:
        Maximum feasible rank, or None if even rank=1 exceeds budget
    """
    max_rank = features  # Rank can't exceed feature dimension

    # Binary search for maximum feasible rank
    low, high = 1, max_rank
    best_rank = None

    while low <= high:
        mid = (low + high) // 2
        breakdown = stacr_stored_bytes(
            batch,
            features,
            mid,
            residual_budget,
            dtype,
            coeff_bits,
            residual_bits,
            basis_dtype,
            include_meta,
        )

        if breakdown.total_bytes <= budget_bytes:
            best_rank = mid
            low = mid + 1
        else:
            high = mid - 1

    return best_rank


def plan_stacr(
    batch: int,
    features: int,
    dtype: torch.dtype,
    budget_bytes: Optional[int] = None,
    target_ratio: Optional[float] = None,
    residual_budget: float = 0.1,
    coeff_bits: Optional[int] = None,
    residual_bits: Optional[int] = None,
    basis_dtype: Optional[torch.dtype] = None,
    include_meta: bool = True,
) -> PlanResult:
    """
    Plan STAC-R compression under budget or ratio constraint.

    Exactly one of budget_bytes or target_ratio must be specified.

    Args:
        batch: Batch dimension (B)
        features: Feature dimension (N)
        dtype: Element dtype
        budget_bytes: Byte budget constraint
        target_ratio: Target compression ratio constraint
        residual_budget: Fraction of activation stored as residual (0 for STAC)
        include_meta: Whether to include metadata

    Returns:
        PlanResult with feasibility, chosen rank, and breakdown
    """
    shape = (batch, features)
    orig_bytes = original_bytes(shape, dtype)

    # Compute rho_max for STAC-R
    rho_max = stacr_rho_max(batch, features, residual_budget, dtype, coeff_bits, residual_bits)

    # Convert target_ratio to budget if needed
    if target_ratio is not None and budget_bytes is None:
        if target_ratio > rho_max:
            return PlanResult(
                feasible=False,
                reason=FeasibilityReason.TARGET_RATIO_EXCEEDS_RHO_MAX,
                original_bytes=orig_bytes,
                budget_bytes=None,
                rho_max=rho_max,
                shape=shape,
                method="stacr",
            )
        budget_bytes = int(math.floor(orig_bytes / target_ratio))

    if budget_bytes is None:
        return PlanResult(
            feasible=False,
            reason=FeasibilityReason.INVALID_PARAMETERS,
            original_bytes=orig_bytes,
            shape=shape,
            method="stacr",
        )

    # Check if even rank=1 fits
    min_breakdown = stacr_stored_bytes(
        batch,
        features,
        1,
        residual_budget,
        dtype,
        coeff_bits,
        residual_bits,
        basis_dtype,
        include_meta,
    )
    if min_breakdown.total_bytes > budget_bytes:
        return PlanResult(
            feasible=False,
            reason=FeasibilityReason.BUDGET_TOO_LOW_FOR_MIN_PARAMS,
            original_bytes=orig_bytes,
            budget_bytes=budget_bytes,
            stored_bytes=min_breakdown.total_bytes,
            rho_max=rho_max,
            k=1,
            breakdown=min_breakdown,
            shape=shape,
            method="stacr",
        )

    # Find maximum rank within budget
    rank = stacr_rank_from_budget(
        batch,
        features,
        budget_bytes,
        residual_budget,
        dtype,
        coeff_bits,
        residual_bits,
        basis_dtype,
        include_meta,
    )

    if rank is None:
        return PlanResult(
            feasible=False,
            reason=FeasibilityReason.BUDGET_TOO_LOW_FOR_MIN_PARAMS,
            original_bytes=orig_bytes,
            budget_bytes=budget_bytes,
            rho_max=rho_max,
            shape=shape,
            method="stacr",
        )

    breakdown = stacr_stored_bytes(
        batch,
        features,
        rank,
        residual_budget,
        dtype,
        coeff_bits,
        residual_bits,
        basis_dtype,
        include_meta,
    )
    achieved_ratio = orig_bytes / breakdown.total_bytes

    return PlanResult(
        feasible=True,
        reason=FeasibilityReason.FEASIBLE,
        original_bytes=orig_bytes,
        budget_bytes=budget_bytes,
        stored_bytes=breakdown.total_bytes,
        achieved_ratio=achieved_ratio,
        k=rank,  # Using k for rank (consistent with SVD)
        rho_max=rho_max,
        breakdown=breakdown,
        shape=shape,
        method="stacr",
    )


# =============================================================================
# COMPARISON UTILITIES
# =============================================================================


def compare_methods_at_budget(
    shape: Tuple[int, ...], dtype: torch.dtype, budget_bytes: int, include_meta: bool = True
) -> Dict[str, PlanResult]:
    """
    Compare all compression methods at a given budget.

    For 2D shapes: compares SVD, TT (tensorized), Tucker (tensorized)
    For higher-order: compares TT and Tucker

    Args:
        shape: Tensor shape
        dtype: Element dtype
        budget_bytes: Byte budget
        include_meta: Whether to include metadata

    Returns:
        Dictionary of method name -> PlanResult
    """
    results = {}

    if len(shape) == 2:
        # SVD on matrix
        results["svd"] = plan_svd(
            shape[0], shape[1], dtype, budget_bytes=budget_bytes, include_meta=include_meta
        )

    # TT and Tucker on full shape
    results["tt"] = plan_tt(shape, dtype, budget_bytes=budget_bytes, include_meta=include_meta)

    results["tucker"] = plan_tucker(
        shape, dtype, budget_bytes=budget_bytes, include_meta=include_meta
    )

    return results


def format_plan_result(result: PlanResult) -> str:
    """Format PlanResult for human-readable logging."""
    lines = [
        f"Method: {result.method}",
        f"Shape: {result.shape}",
        f"Feasible: {result.feasible}",
        f"Reason: {result.reason.value}",
        f"Original bytes: {result.original_bytes:,}",
    ]

    if result.budget_bytes is not None:
        lines.append(f"Budget bytes: {result.budget_bytes:,}")

    if result.stored_bytes is not None:
        lines.append(f"Stored bytes: {result.stored_bytes:,}")

    if result.achieved_ratio is not None:
        lines.append(f"Achieved ratio: {result.achieved_ratio:.2f}x")

    if result.k is not None:
        lines.append(f"SVD rank k: {result.k}")

    if result.ranks is not None:
        lines.append(f"Ranks: {result.ranks}")

    if result.rho_max is not None:
        lines.append(f"rho_max (SVD ceiling): {result.rho_max:.2f}x")

    return "\n".join(lines)


# =============================================================================
# VALIDATION HELPERS
# =============================================================================


def validate_tt_ranks(dims: Sequence[int], tt_ranks: Sequence[int]) -> Tuple[bool, str]:
    """
    Validate TT ranks for given dimensions.

    Checks:
    - Length is len(dims) + 1
    - Boundary ranks are 1
    - Internal ranks are positive

    Returns:
        (valid, message)
    """
    if len(tt_ranks) != len(dims) + 1:
        return False, f"TT ranks length {len(tt_ranks)} must be {len(dims) + 1}"

    if tt_ranks[0] != 1:
        return False, f"First TT rank must be 1, got {tt_ranks[0]}"

    if tt_ranks[-1] != 1:
        return False, f"Last TT rank must be 1, got {tt_ranks[-1]}"

    for i, r in enumerate(tt_ranks):
        if r < 1:
            return False, f"TT rank at position {i} must be >= 1, got {r}"

    return True, "valid"


def validate_tucker_ranks(dims: Sequence[int], ranks: Sequence[int]) -> Tuple[bool, str]:
    """
    Validate Tucker ranks for given dimensions.

    Checks:
    - Length matches dims
    - Each rank is in [1, n_i]

    Returns:
        (valid, message)
    """
    if len(ranks) != len(dims):
        return False, f"Tucker ranks length {len(ranks)} must match dims {len(dims)}"

    for i, (n, r) in enumerate(zip(dims, ranks)):
        if r < 1 or r > n:
            return False, f"Tucker rank {r} at mode {i} must be in [1, {n}]"

    return True, "valid"


# =============================================================================
# PER-LAYER MEMORY TABLE
# =============================================================================


@dataclass
class LayerMemoryPlan:
    """Memory plan for a single layer."""

    layer_name: str
    shape: Tuple[int, ...]
    original_bytes: int
    budget_bytes: int
    method: str
    plan_result: PlanResult


def print_memory_table(layer_plans: List[LayerMemoryPlan], file=None) -> str:
    """
    Print a formatted per-layer memory accounting table.

    Shows feasibility and storage estimates before running GPU experiments.

    Args:
        layer_plans: List of LayerMemoryPlan for each layer
        file: Optional file to write to (default: stdout)

    Returns:
        The formatted table as a string
    """
    # Column headers
    headers = [
        "Layer",
        "Shape",
        "Original",
        "Budget",
        "Method",
        "Params",
        "Stored",
        "Ratio",
        "Feasible",
        "Reason",
    ]

    # Format each row
    rows = []
    for lp in layer_plans:
        pr = lp.plan_result

        # Format params (k or ranks)
        if pr.k is not None:
            params = f"k={pr.k}"
        elif pr.ranks is not None:
            if len(pr.ranks) <= 5:
                params = str(pr.ranks)
            else:
                params = f"[{pr.ranks[0]}..{pr.ranks[-1]}]"
        else:
            params = "N/A"

        # Format stored bytes
        stored = f"{pr.stored_bytes:,}" if pr.stored_bytes else "N/A"

        # Format ratio
        ratio = f"{pr.achieved_ratio:.1f}x" if pr.achieved_ratio else "N/A"

        # Format reason
        reason = pr.reason.value if not pr.feasible else ""

        row = [
            lp.layer_name[:20],
            str(lp.shape),
            f"{lp.original_bytes:,}",
            f"{lp.budget_bytes:,}",
            lp.method.upper(),
            params,
            stored,
            ratio,
            "Yes" if pr.feasible else "No",
            reason[:30] if reason else "",
        ]
        rows.append(row)

    # Calculate column widths
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    # Build the table
    lines = []

    # Header
    header_line = " | ".join(h.ljust(w) for h, w in zip(headers, widths))
    lines.append(header_line)
    lines.append("-" * len(header_line))

    # Data rows
    for row in rows:
        lines.append(" | ".join(cell.ljust(w) for cell, w in zip(row, widths)))

    # Summary
    lines.append("-" * len(header_line))
    total_original = sum(lp.original_bytes for lp in layer_plans)
    total_stored = sum(lp.plan_result.stored_bytes or 0 for lp in layer_plans)
    feasible_count = sum(1 for lp in layer_plans if lp.plan_result.feasible)

    if total_stored > 0:
        overall_ratio = total_original / total_stored
        lines.append(
            f"Total: {total_original:,} original bytes -> {total_stored:,} stored bytes = {overall_ratio:.1f}x compression"
        )
    lines.append(f"Feasible: {feasible_count}/{len(layer_plans)} layers")

    table_str = "\n".join(lines)

    if file is not None:
        print(table_str, file=file)
    else:
        print(table_str)

    return table_str


def plan_layers_for_debug(
    layer_configs: List[Dict[str, Any]],
    global_budget_bytes: Optional[int] = None,
    target_ratio: Optional[float] = None,
    dtype: torch.dtype = torch.float32,
    methods: List[str] = None,
) -> List[LayerMemoryPlan]:
    """
    Plan memory for multiple layers and return structured plans.

    Args:
        layer_configs: List of dicts with keys:
            - "name": layer name
            - "shape": tuple of dimensions
            - "method": "svd", "tt", or "tucker" (optional if methods provided)
            - "budget_bytes": per-layer budget (optional)
        global_budget_bytes: Default budget if not specified per-layer
        target_ratio: Target compression ratio (alternative to budget)
        dtype: Data type for byte calculations
        methods: If provided, plan all methods for each layer

    Returns:
        List of LayerMemoryPlan objects
    """
    if methods is None:
        methods = ["svd"]

    plans = []

    for config in layer_configs:
        name = config.get("name", "layer")
        shape = tuple(config["shape"])
        layer_budget = config.get("budget_bytes", global_budget_bytes)

        orig = original_bytes(shape, dtype)

        # Compute budget from ratio if needed
        if layer_budget is None and target_ratio is not None:
            layer_budget = int(orig / target_ratio)

        if layer_budget is None:
            layer_budget = orig  # No compression

        # Plan each method
        for method in config.get("methods") or methods:
            if method == "svd" and len(shape) == 2:
                result = plan_svd(shape[0], shape[1], dtype, budget_bytes=layer_budget)
            elif method == "tt":
                result = plan_tt(shape, dtype, budget_bytes=layer_budget)
            elif method == "tucker":
                result = plan_tucker(shape, dtype, budget_bytes=layer_budget)
            else:
                continue

            plans.append(
                LayerMemoryPlan(
                    layer_name=name,
                    shape=shape,
                    original_bytes=orig,
                    budget_bytes=layer_budget,
                    method=method,
                    plan_result=result,
                )
            )

    return plans


def debug_memory_accounting(
    layer_configs: List[Dict[str, Any]],
    budget_bytes: Optional[int] = None,
    target_ratio: Optional[float] = None,
    dtype: torch.dtype = torch.float32,
    methods: List[str] = None,
    file=None,
) -> str:
    """
    One-call debug function: plan all layers and print table.

    Example usage:
        debug_memory_accounting([
            {"name": "fc1", "shape": (64, 512)},
            {"name": "fc2", "shape": (64, 256)},
        ], target_ratio=50.0, methods=["svd", "tt"])

    Args:
        layer_configs: Layer configuration dicts
        budget_bytes: Global byte budget
        target_ratio: Target compression ratio
        dtype: Data type
        methods: Methods to try per layer
        file: Output file (default: stdout)

    Returns:
        Formatted table string
    """
    plans = plan_layers_for_debug(
        layer_configs,
        global_budget_bytes=budget_bytes,
        target_ratio=target_ratio,
        dtype=dtype,
        methods=methods,
    )
    return print_memory_table(plans, file=file)
