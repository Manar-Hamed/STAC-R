from __future__ import annotations

import math

import torch


def dtype_bytes(dtype: torch.dtype) -> int:
    return torch.empty((), dtype=dtype).element_size()


def dense_bytes(shape: tuple[int, ...], dtype: torch.dtype) -> int:
    n = 1
    for dim in shape:
        if isinstance(dim, bool) or int(dim) != dim or dim < 0:
            raise ValueError("Shape dimensions must be nonnegative integers")
        n *= int(dim)
    return n * dtype_bytes(dtype)


def stacr_stored_bytes(
    shape: tuple[int, ...],
    rank: int,
    dtype: torch.dtype = torch.float16,
    residual_budget: float = 0.005,
    index_bytes: int = 4,
    include_basis: bool = False,
    *,
    coeff_bits: int | None = None,
    residual_bits: int | None = None,
    basis_dtype: torch.dtype | None = None,
    include_scales: bool = True,
) -> int:
    """Estimate factor payload, excluding parameters and temporary workspace.

    Unspecified bit widths keep the activation dtype. Explicit 4/8-bit storage
    includes a four-byte per-tensor scale. Explicit residual_bits=0 describes
    the unquantized sparse channel (which also stores a unit scale). Basis
    storage is counted once when requested; block selection can round its
    residual count up to whole blocks and needs its own measured count.
    """
    dense_bytes(shape, dtype)  # Validate dimensions even for empty payloads.
    if isinstance(rank, bool) or int(rank) != rank or rank < 1:
        raise ValueError("rank must be a positive integer")
    if not math.isfinite(residual_budget) or not 0 <= residual_budget <= 1:
        raise ValueError("residual_budget must be between 0 and 1")
    if index_bytes not in (2, 4, 8):
        raise ValueError("index_bytes must be 2, 4, or 8")
    if coeff_bits not in (None, 0, 4, 8) or residual_bits not in (None, 0, 4, 8):
        raise ValueError("Bit widths must be None, 0, 4, or 8")
    if len(shape) == 0:
        return dense_bytes(shape, dtype)
    features = int(shape[-1])
    if features == 0:
        return 0
    rows = 1
    for dim in shape[:-1]:
        rows *= int(dim)
    rank = min(int(rank), features)
    value_bytes = dtype_bytes(dtype)
    coeff_count = rows * rank
    coeff = math.ceil(coeff_count * coeff_bits / 8) if coeff_bits else coeff_count * value_bytes
    if include_scales and coeff_bits and coeff_count:
        coeff += 4
    residual_n = int(round(float(residual_budget) * rows * features))
    residual_values = (
        math.ceil(residual_n * residual_bits / 8) if residual_bits else residual_n * value_bytes
    )
    residual = residual_n * index_bytes + residual_values
    if include_scales and residual_bits is not None and residual_n:
        residual += 4
    basis = features * rank * dtype_bytes(basis_dtype or dtype) if include_basis else 0
    return int(coeff + residual + basis)


def saved_surface_predictor(saved_bytes: float, baseline_peak_bytes: float) -> float:
    """Return SSP = positive replaceable saved-surface bytes / baseline peak."""
    if not math.isfinite(baseline_peak_bytes) or baseline_peak_bytes <= 0:
        raise ValueError("baseline_peak_bytes must be finite and positive")
    if not math.isfinite(saved_bytes):
        raise ValueError("saved_bytes must be finite")
    return max(0.0, float(saved_bytes)) / float(baseline_peak_bytes)
