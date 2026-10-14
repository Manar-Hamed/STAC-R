"""
Sparse index compression utilities for STAC-R

Provides efficient index compression methods for sparse tensor storage,
critical for memory-efficient activation compression during training.

Key insight for STACR: Residual indices are stored as flat 1D indices where
flat_idx = row * N + col. By decomposing into (row, col) and using CSR format:
- Row pointers: (B_flat + 1) int32 elements
- Column indices: (nnz) int16 elements when N < 32768

Storage depends on matrix dimensions and the number of retained entries.

Example (BERT-base: B=32, L=128, N=3072, budget=0.01):
- Current: 125,829 × 4 bytes = 503 KB
- CSR: 4,097 × 4 + 125,829 × 2 = 268 KB (47% savings)

Classes:
    - CompressedIndices: Container for compressed index data
    - DeltaEncodedIndices: Delta encoding with variable-length integers
    - BlockOffsetEncoding: Block-sparse encoding for structured patterns
    - IndexCompressionStats: Performance monitoring

Functions:
    - compress_indices: Compress flat 1D indices to CSR/delta format
    - decompress_indices: Decompress back to flat 1D format
    - csr_memory_savings: Calculate CSR compression metrics
    - compress_activation_to_csr: Convert dense activation to CSR
    - analyze_delta_compression_potential: Evaluate delta encoding benefit
"""

import torch
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Tuple, Optional, List, Union, Any


# =============================================================================
# Index Compression Mode Enum
# =============================================================================


class IndexCompressionMode(Enum):
    """Index compression modes for STACR residual indices."""

    NONE = "none"  # No compression (baseline)
    CSR = "csr"  # CSR-like format with row pointers + column indices
    DELTA = "delta"  # Delta encoding on sorted indices


# =============================================================================
# Compressed Indices Container
# =============================================================================


@dataclass
class CompressedIndices:
    """Container for compressed index data.

    Used by compress_indices() and decompress_indices() to store
    compressed residual indices in memory-efficient format.

    Attributes:
        mode: Compression mode used
        row_ptrs: (B_flat + 1,) cumulative element counts per row (CSR mode)
        col_indices: Column indices (can use int16 if N < 32768)
        delta_base: Base value for delta decoding (delta mode)
        sort_indices: Indices to unsort after decompression (preserves original order)
        nnz: Number of non-zero elements
        B_flat: Batch dimension (rows)
        N: Feature dimension (columns)
        original_dtype: Original index dtype before compression
    """

    mode: IndexCompressionMode
    row_ptrs: Optional[torch.Tensor] = None
    col_indices: Optional[torch.Tensor] = None
    delta_base: Optional[torch.Tensor] = None
    sort_indices: Optional[torch.Tensor] = None  # For restoring original order
    nnz: int = 0
    B_flat: int = 0
    N: int = 0
    original_dtype: torch.dtype = torch.int32

    def memory_bytes(self) -> int:
        """Calculate total memory usage of compressed representation."""
        total = 0
        if self.row_ptrs is not None:
            total += self.row_ptrs.numel() * self.row_ptrs.element_size()
        if self.col_indices is not None:
            total += self.col_indices.numel() * self.col_indices.element_size()
        if self.delta_base is not None:
            total += self.delta_base.numel() * self.delta_base.element_size()
        if self.sort_indices is not None:
            total += self.sort_indices.numel() * self.sort_indices.element_size()
        return total

    def to(self, device: torch.device) -> "CompressedIndices":
        """Move all tensors to specified device."""
        return CompressedIndices(
            mode=self.mode,
            row_ptrs=self.row_ptrs.to(device) if self.row_ptrs is not None else None,
            col_indices=self.col_indices.to(device) if self.col_indices is not None else None,
            delta_base=self.delta_base.to(device) if self.delta_base is not None else None,
            sort_indices=self.sort_indices.to(device) if self.sort_indices is not None else None,
            nnz=self.nnz,
            B_flat=self.B_flat,
            N=self.N,
            original_dtype=self.original_dtype,
        )


# =============================================================================
# Core Compression/Decompression Functions
# =============================================================================


def _choose_col_dtype(N: int) -> torch.dtype:
    """Choose optimal dtype for column indices based on feature dimension.

    Args:
        N: Feature dimension (maximum column index + 1)

    Returns:
        torch.int16 if N < 32768, else torch.int32
    """
    if N < 32768:
        return torch.int16
    elif N <= 2**31:
        return torch.int32
    return torch.int64


def compress_indices(
    flat_indices: torch.Tensor,
    B_flat: int,
    N: int,
    mode: Union[str, IndexCompressionMode] = "csr",
    preserve_order: bool = True,
) -> CompressedIndices:
    """Compress flat 1D indices into memory-efficient format.

    This is the main entry point for index compression in STACR.

    Args:
        flat_indices: (nnz,) flat indices where flat_idx = row * N + col
        B_flat: Number of rows (batch dimension)
        N: Number of columns (feature dimension)
        mode: Compression mode ("none", "csr", "delta")
        preserve_order: If True, stores sort indices to restore original order

    Returns:
        CompressedIndices containing the compressed representation

    Example:
        >>> # BERT-base style indices
        >>> indices = torch.randint(0, 4096 * 3072, (125829,), dtype=torch.int32)
        >>> compressed = compress_indices(indices, B_flat=4096, N=3072, mode="csr")
        >>> print(f"Compression: {indices.numel() * 4 / compressed.memory_bytes():.2f}×")
    """
    if isinstance(mode, str):
        mode = IndexCompressionMode(mode)

    device = flat_indices.device
    original_dtype = flat_indices.dtype
    nnz = flat_indices.numel()

    # Handle empty case
    if nnz == 0:
        return CompressedIndices(
            mode=mode,
            row_ptrs=torch.zeros(B_flat + 1, dtype=torch.int32, device=device)
            if mode == IndexCompressionMode.CSR
            else None,
            col_indices=torch.tensor([], dtype=_choose_col_dtype(N), device=device),
            nnz=0,
            B_flat=B_flat,
            N=N,
            original_dtype=original_dtype,
        )

    if mode == IndexCompressionMode.NONE:
        # No compression - just store metadata
        return CompressedIndices(
            mode=mode,
            col_indices=flat_indices,  # Store as-is
            nnz=nnz,
            B_flat=B_flat,
            N=N,
            original_dtype=original_dtype,
        )

    elif mode == IndexCompressionMode.CSR:
        # Decompose flat indices into (row, col)
        flat_long = flat_indices.long()
        rows = torch.div(flat_long, N, rounding_mode="floor")
        cols = flat_long % N

        # Sort by row for CSR format
        sort_idx = torch.argsort(rows, stable=True)
        sorted_rows = rows[sort_idx]
        sorted_cols = cols[sort_idx]

        # Store inverse sort indices if we need to preserve order
        unsort_idx = None
        if preserve_order:
            unsort_idx = torch.argsort(sort_idx)

        # Compute row pointers using bincount
        # Note: Keep as long to avoid int32 overflow for large workloads
        row_counts = torch.bincount(sorted_rows.long(), minlength=B_flat)
        pointer_dtype = torch.int32 if nnz < 2**31 else torch.int64
        row_ptrs = torch.zeros(B_flat + 1, dtype=pointer_dtype, device=device)
        torch.cumsum(row_counts, dim=0, out=row_ptrs[1:])

        # Store column indices with optimal dtype
        col_dtype = _choose_col_dtype(N)
        col_indices = sorted_cols.to(col_dtype)

        return CompressedIndices(
            mode=mode,
            row_ptrs=row_ptrs,
            col_indices=col_indices,
            sort_indices=unsort_idx,
            nnz=nnz,
            B_flat=B_flat,
            N=N,
            original_dtype=original_dtype,
        )

    elif mode == IndexCompressionMode.DELTA:
        # Sort indices
        sorted_indices, sort_idx = torch.sort(flat_indices)

        # Store inverse sort indices if we need to preserve order
        unsort_idx = None
        if preserve_order:
            unsort_idx = torch.argsort(sort_idx)

        # Compute deltas
        base_dtype = torch.int32 if B_flat * N <= 2**31 else torch.int64
        base = sorted_indices[0:1].to(base_dtype)
        if nnz > 1:
            deltas = sorted_indices[1:].long() - sorted_indices[:-1].long()

            # Choose dtype based on max delta
            # int16 range: -32768 to 32767 (signed), we only use positive
            max_delta = deltas.max().item()
            if max_delta <= 32767:  # Explicit signed int16 max
                delta_dtype = torch.int16
            elif max_delta < 2**31:
                delta_dtype = torch.int32
            else:
                delta_dtype = torch.int64

            deltas = deltas.to(delta_dtype)
        else:
            deltas = torch.tensor([], dtype=torch.int16, device=device)

        return CompressedIndices(
            mode=mode,
            col_indices=deltas,  # Store deltas in col_indices field
            delta_base=base,
            sort_indices=unsort_idx,
            nnz=nnz,
            B_flat=B_flat,
            N=N,
            original_dtype=original_dtype,
        )

    else:
        raise ValueError(f"Unknown compression mode: {mode}")


def decompress_indices(
    compressed: CompressedIndices,
    restore_order: bool = True,
) -> torch.Tensor:
    """Decompress indices back to flat 1D format.

    Args:
        compressed: CompressedIndices from compress_indices()
        restore_order: If True and sort_indices available, restore original order

    Returns:
        (nnz,) flat indices where flat_idx = row * N + col
    """
    mode = compressed.mode

    if compressed.col_indices is not None:
        device = compressed.col_indices.device
    elif compressed.delta_base is not None:
        device = compressed.delta_base.device
    else:
        device = torch.device("cpu")

    if compressed.nnz == 0:
        return torch.tensor([], dtype=compressed.original_dtype, device=device)

    if mode == IndexCompressionMode.NONE:
        return compressed.col_indices.to(compressed.original_dtype)

    elif mode == IndexCompressionMode.CSR:
        row_ptrs = compressed.row_ptrs
        col_indices = compressed.col_indices
        N = compressed.N
        nnz = compressed.nnz
        B_flat = compressed.B_flat

        # Compute row counts from row pointers
        row_counts = row_ptrs[1:] - row_ptrs[:-1]  # (B_flat,)

        # Validate consistency to catch corrupted state (e.g., from offloading)
        expected_nnz = row_counts.sum().item()
        if expected_nnz != nnz:
            raise ValueError(f"CSR row pointers sum to {expected_nnz} but nnz={nnz}")
        if col_indices.numel() != nnz:
            raise ValueError(f"CSR col_indices has {col_indices.numel()} elements but nnz={nnz}")

        # Create row indices by repeating each row index by its count
        row_indices = torch.repeat_interleave(
            torch.arange(B_flat, device=device, dtype=torch.int64), row_counts.long()
        )

        # Compute flat indices
        flat_indices = row_indices * N + col_indices.long()
        flat_indices = flat_indices.to(compressed.original_dtype)

        # Restore original order if requested
        if restore_order and compressed.sort_indices is not None:
            flat_indices = flat_indices[compressed.sort_indices]

        return flat_indices

    elif mode == IndexCompressionMode.DELTA:
        deltas = compressed.col_indices
        base = compressed.delta_base
        nnz = compressed.nnz

        if nnz == 1:
            result = base.to(compressed.original_dtype)
        else:
            # Reconstruct by cumulative sum of deltas
            result = torch.zeros(nnz, dtype=torch.int64, device=device)
            result[0] = base[0]
            if nnz > 1:
                result[1:] = deltas.long()
                torch.cumsum(result, dim=0, out=result)

            result = result.to(compressed.original_dtype)

        # Restore original order if requested
        if restore_order and compressed.sort_indices is not None:
            result = result[compressed.sort_indices]

        return result

    else:
        raise ValueError(f"Unknown compression mode: {mode}")


def estimate_index_compression(
    B_flat: int,
    N: int,
    nnz: int,
    mode: Union[str, IndexCompressionMode] = "csr",
    original_dtype: torch.dtype = torch.int32,
    preserve_order: bool = True,
) -> Dict[str, Any]:
    """Estimate compression ratio without actual data.

    Useful for planning memory budgets before running experiments.

    Args:
        B_flat: Number of rows
        N: Number of columns
        nnz: Number of non-zero elements
        mode: Compression mode
        original_dtype: Original index dtype
        preserve_order: Whether order preservation is enabled

    Returns:
        Dict with memory estimates and compression ratio
    """
    if isinstance(mode, str):
        mode = IndexCompressionMode(mode)

    elem_size = 4 if original_dtype in (torch.int32, torch.long) else 2
    original_bytes = nnz * elem_size

    if mode == IndexCompressionMode.NONE:
        compressed_bytes = original_bytes

    elif mode == IndexCompressionMode.CSR:
        row_ptr_bytes = (B_flat + 1) * 4  # int32
        col_bytes = nnz * (2 if N < 32768 else 4)  # int16 or int32
        sort_bytes = nnz * 4 if preserve_order else 0  # int32 for unsort indices
        compressed_bytes = row_ptr_bytes + col_bytes + sort_bytes

    elif mode == IndexCompressionMode.DELTA:
        base_bytes = 4  # int32
        # Assume deltas fit in int16 (common case for clustered indices)
        delta_bytes = max(0, nnz - 1) * 2
        sort_bytes = nnz * 4 if preserve_order else 0
        compressed_bytes = base_bytes + delta_bytes + sort_bytes

    else:
        compressed_bytes = original_bytes

    ratio = original_bytes / compressed_bytes if compressed_bytes > 0 else float("inf")

    return {
        "original_bytes": original_bytes,
        "compressed_bytes": compressed_bytes,
        "compression_ratio": ratio,
        "savings_percent": (1 - compressed_bytes / original_bytes) * 100
        if original_bytes > 0
        else 0,
        "savings_bytes": original_bytes - compressed_bytes,
        "mode": mode.value,
        "B_flat": B_flat,
        "N": N,
        "nnz": nnz,
    }


# =============================================================================
# Delta and CSR helper APIs
# =============================================================================


def csr_memory_savings(
    batch_size: int,
    features: int,
    nnz: int,
    dtype: torch.dtype = torch.float32,
    index_dtype: torch.dtype = torch.int32,
) -> Dict[str, float]:
    """
    Calculate memory savings of CSR format vs dense storage.

    Args:
        batch_size: Number of rows
        features: Number of columns
        nnz: Number of non-zero elements
        dtype: Data type for values
        index_dtype: Data type for indices (int32 or int64)

    Returns:
        dict with memory breakdown:
            - dense_bytes: Dense storage size
            - csr_bytes: CSR storage size
            - crow_bytes: Row pointer storage
            - col_bytes: Column index storage
            - values_bytes: Value storage
            - savings_bytes: Memory saved
            - sparsity: Sparsity ratio
            - compression_ratio: Compression ratio
    """
    # Dense storage
    dense_bytes = batch_size * features * _dtype_bytes(dtype)

    # CSR storage
    crow_bytes = (batch_size + 1) * _dtype_bytes(index_dtype)
    col_bytes = nnz * _dtype_bytes(index_dtype)
    values_bytes = nnz * _dtype_bytes(dtype)
    csr_bytes = crow_bytes + col_bytes + values_bytes

    # Metrics
    sparsity = 1.0 - (nnz / (batch_size * features)) if batch_size * features > 0 else 0.0
    compression_ratio = dense_bytes / csr_bytes if csr_bytes > 0 else float("inf")

    return {
        "dense_bytes": dense_bytes,
        "csr_bytes": csr_bytes,
        "crow_bytes": crow_bytes,
        "col_bytes": col_bytes,
        "values_bytes": values_bytes,
        "savings_bytes": dense_bytes - csr_bytes,
        "sparsity": sparsity,
        "compression_ratio": compression_ratio,
    }


def compress_activation_to_csr(
    activation: torch.Tensor, threshold: float = 0.0, index_dtype: torch.dtype = torch.int32
) -> torch.Tensor:
    """
    Compress activation tensor to CSR format by thresholding.

    Args:
        activation: Dense tensor of shape (B, N)
        threshold: Absolute threshold for keeping values
        index_dtype: Data type for indices (int32 or int64)

    Returns:
        CSR sparse tensor

    Example:
        >>> activation = torch.randn(64, 512)
        >>> csr = compress_activation_to_csr(activation, threshold=0.5)
        >>> reconstructed = csr.to_dense()
    """
    if activation.ndim != 2:
        raise ValueError(f"Expected 2D tensor, got shape {activation.shape}")

    B, N = activation.shape
    device = activation.device

    # Create mask for significant values
    mask = torch.abs(activation) > threshold

    # Get non-zero positions
    row_indices, col_indices = mask.nonzero(as_tuple=True)
    values = activation[mask]

    # Build crow_indices (cumulative count of nnz per row)
    crow_indices = torch.zeros(B + 1, dtype=index_dtype, device=device)
    for i in range(B):
        crow_indices[i + 1] = (row_indices <= i).sum()

    # Create CSR tensor
    csr_activation = torch.sparse_csr_tensor(
        crow_indices,
        col_indices.to(index_dtype),
        values,
        size=(B, N),
        dtype=activation.dtype,
        device=device,
    )

    return csr_activation


def analyze_delta_compression_potential(indices: torch.Tensor) -> Dict[str, float]:
    """
    Analyze potential compression from delta encoding.

    Args:
        indices: Sorted 1D tensor of indices

    Returns:
        dict with analysis:
            - num_indices: Number of indices
            - avg_delta: Average delta value
            - max_delta: Maximum delta
            - min_delta: Minimum delta
            - original_bytes: Original storage size
            - delta_bytes: Estimated delta-encoded size
            - compression_ratio: Compression ratio
            - delta_histogram: Distribution of delta sizes
    """
    if len(indices) == 0:
        return {
            "num_indices": 0,
            "avg_delta": 0,
            "max_delta": 0,
            "min_delta": 0,
            "original_bytes": 0,
            "delta_bytes": 0,
            "compression_ratio": 1.0,
            "delta_histogram": {},
        }

    # Calculate deltas
    deltas = torch.diff(indices, prepend=torch.tensor([0], device=indices.device))

    # Analyze distribution
    avg_delta = deltas.float().mean().item()
    max_delta = deltas.max().item()
    min_delta = deltas.min().item()

    # Estimate variable-length encoding size
    def bits_needed_varint(delta: int) -> int:
        """Variable-length integer encoding (protobuf-style)."""
        if delta < (1 << 7):
            return 8
        elif delta < (1 << 14):
            return 16
        elif delta < (1 << 21):
            return 24
        else:
            return 32

    bits_per_delta = torch.tensor([bits_needed_varint(d.item()) for d in deltas])
    total_bits_delta = bits_per_delta.sum().item()
    total_bits_original = len(indices) * 32  # int32

    compression_ratio = total_bits_original / total_bits_delta if total_bits_delta > 0 else 1.0

    # Distribution histogram
    delta_histogram = {
        "< 128 (1 byte)": (deltas < 128).sum().item(),
        "128-16383 (2 bytes)": ((deltas >= 128) & (deltas < 16384)).sum().item(),
        ">= 16384 (4 bytes)": (deltas >= 16384).sum().item(),
    }

    return {
        "num_indices": len(indices),
        "avg_delta": avg_delta,
        "max_delta": max_delta,
        "min_delta": min_delta,
        "original_bytes": total_bits_original // 8,
        "delta_bytes": total_bits_delta // 8,
        "compression_ratio": compression_ratio,
        "delta_histogram": delta_histogram,
    }


class DeltaEncodedIndices:
    """
    Delta-encoded index storage for PyTorch tensors.

    Stores indices as: base_value + cumsum(deltas)
    Uses variable-length encoding for deltas.

    Attributes:
        base: First index value
        deltas: Delta-encoded differences
        num_indices: Total number of indices
        device: Torch device
        dtype: Data type for deltas

    Example:
        >>> indices = torch.tensor([10, 12, 15, 18, 22])
        >>> encoded = DeltaEncodedIndices(indices)
        >>> reconstructed = encoded.decode()
        >>> assert torch.equal(indices, reconstructed)
    """

    def __init__(self, indices: torch.Tensor, dtype: torch.dtype = torch.int32):
        """
        Encode indices with delta encoding.

        Args:
            indices: Sorted 1D tensor of indices
            dtype: Data type for deltas
        """
        if indices.ndim != 1:
            raise ValueError(f"Indices must be 1D, got shape {indices.shape}")

        # Check sorted
        if len(indices) > 1 and not (torch.diff(indices) >= 0).all():
            raise ValueError("Indices must be sorted")

        self.device = indices.device
        self.dtype = dtype
        self.num_indices = len(indices)

        if len(indices) == 0:
            self.base = 0
            self.deltas = torch.tensor([], dtype=dtype, device=self.device)
        else:
            self.base = indices[0].item()
            deltas = torch.diff(indices, prepend=torch.tensor([self.base], device=self.device))
            self.deltas = deltas[1:].to(dtype)  # Skip first delta (0)

    def decode(self) -> torch.Tensor:
        """
        Reconstruct original indices from delta encoding.

        Returns:
            Reconstructed indices tensor
        """
        if self.num_indices == 0:
            return torch.tensor([], dtype=self.dtype, device=self.device)

        # Reconstruct: indices[i] = base + sum(deltas[0:i])
        cumsum = torch.cat(
            [
                torch.tensor([self.base], dtype=self.dtype, device=self.device),
                torch.cumsum(self.deltas, dim=0) + self.base,
            ]
        )

        return cumsum

    def memory_bytes(self) -> int:
        """Calculate memory usage in bytes."""
        base_bytes = 4  # int32
        deltas_bytes = self.deltas.element_size() * self.deltas.numel()
        overhead_bytes = 12  # metadata
        return base_bytes + deltas_bytes + overhead_bytes

    def __repr__(self) -> str:
        return (
            f"DeltaEncodedIndices(num_indices={self.num_indices}, "
            f"base={self.base}, dtype={self.dtype}, "
            f"memory={self.memory_bytes()} bytes)"
        )


class BlockOffsetEncoding:
    """
    Block-offset encoding for structured sparse tensors.

    Stores sparse data as:
      - block_indices: which blocks are active
      - block_values: values for each active block
      - row_ptr: row pointers to blocks (CSR-style)

    Effective for structured sparsity patterns (e.g., block-sparse attention).

    Attributes:
        B, N: Original tensor shape
        block_size: Size of square blocks
        block_indices: Flat indices of active blocks
        block_values: Values for each active block
        row_ptr: Row pointers (CSR-style)

    Example:
        >>> tensor = torch.randn(128, 512)
        >>> # Create block sparsity
        >>> tensor[:64, :256] = 0  # Zero out some blocks
        >>> encoded = BlockOffsetEncoding(tensor, block_size=16)
        >>> reconstructed = encoded.to_dense()
    """

    def __init__(self, tensor: torch.Tensor, block_size: int):
        """
        Create block-offset encoding from dense or sparse tensor.

        Args:
            tensor: 2D tensor (B, N)
            block_size: Size of square blocks
        """
        if tensor.ndim != 2:
            raise ValueError(f"Tensor must be 2D, got shape {tensor.shape}")

        self.B, self.N = tensor.shape
        self.block_size = block_size
        self.dtype = tensor.dtype
        self.device = tensor.device

        # Calculate block grid
        self.blocks_per_row = (self.N + block_size - 1) // block_size
        self.blocks_per_col = (self.B + block_size - 1) // block_size

        # Extract non-zero blocks
        block_indices = []
        block_values = []
        row_ptr = [0]

        for block_row in range(self.blocks_per_col):
            row_block_count = 0
            for block_col in range(self.blocks_per_row):
                # Extract block
                row_start = block_row * block_size
                row_end = min(row_start + block_size, self.B)
                col_start = block_col * block_size
                col_end = min(col_start + block_size, self.N)

                block = tensor[row_start:row_end, col_start:col_end]

                # Check if block has significant values
                if block.abs().max() > 1e-6:  # Non-zero threshold
                    # Pad block to full block_size if needed
                    if block.shape != (block_size, block_size):
                        padded = torch.zeros(
                            block_size, block_size, dtype=self.dtype, device=self.device
                        )
                        padded[: block.shape[0], : block.shape[1]] = block
                        block = padded

                    block_indices.append(block_row * self.blocks_per_row + block_col)
                    block_values.append(block)
                    row_block_count += 1

            row_ptr.append(row_ptr[-1] + row_block_count)

        self.block_indices = torch.tensor(block_indices, dtype=torch.int32, device=self.device)
        self.block_values = (
            torch.stack(block_values)
            if block_values
            else torch.tensor([], device=self.device, dtype=self.dtype)
        )
        self.row_ptr = torch.tensor(row_ptr, dtype=torch.int32, device=self.device)

    def to_dense(self) -> torch.Tensor:
        """
        Reconstruct dense tensor from block encoding.

        Returns:
            Reconstructed dense tensor
        """
        result = torch.zeros(self.B, self.N, dtype=self.dtype, device=self.device)

        for i, block_idx in enumerate(self.block_indices):
            block_row = block_idx // self.blocks_per_row
            block_col = block_idx % self.blocks_per_row

            row_start = block_row * self.block_size
            row_end = min(row_start + self.block_size, self.B)
            col_start = block_col * self.block_size
            col_end = min(col_start + self.block_size, self.N)

            # Copy block (handle padding)
            block = self.block_values[i]
            result[row_start:row_end, col_start:col_end] = block[
                : row_end - row_start, : col_end - col_start
            ]

        return result

    def memory_bytes(self) -> int:
        """Calculate memory usage in bytes."""
        indices_bytes = self.block_indices.element_size() * self.block_indices.numel()
        values_bytes = self.block_values.element_size() * self.block_values.numel()
        row_ptr_bytes = self.row_ptr.element_size() * self.row_ptr.numel()
        return indices_bytes + values_bytes + row_ptr_bytes

    def compression_ratio(self) -> float:
        """Calculate compression ratio vs dense storage."""
        original_bytes = self.B * self.N * _dtype_bytes(self.dtype)
        compressed_bytes = self.memory_bytes()
        return original_bytes / compressed_bytes if compressed_bytes > 0 else float("inf")

    def __repr__(self) -> str:
        return (
            f"BlockOffsetEncoding(shape=({self.B}, {self.N}), "
            f"block_size={self.block_size}, "
            f"num_blocks={len(self.block_indices)}, "
            f"compression={self.compression_ratio():.2f}×, "
            f"memory={self.memory_bytes() / 1024:.2f} KB)"
        )


class IndexCompressionStats:
    """
    Track index compression effectiveness during training.

    Useful for monitoring compression overhead and memory savings.

    Example:
        >>> stats = IndexCompressionStats()
        >>> for batch in dataloader:
        ...     indices, compressed_bytes, recon_time = compress_batch(batch)
        ...     stats.record(indices, compressed_bytes, recon_time)
        >>> stats.report()
    """

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset all statistics."""
        self.total_indices = 0
        self.total_bytes_original = 0
        self.total_bytes_compressed = 0
        self.reconstruction_times: List[float] = []

    def record(self, indices: torch.Tensor, compressed_bytes: int, reconstruction_time: float):
        """
        Record statistics for one compression operation.

        Args:
            indices: Original indices
            compressed_bytes: Compressed size in bytes
            reconstruction_time: Time to reconstruct (seconds)
        """
        self.total_indices += len(indices)
        self.total_bytes_original += len(indices) * 4  # int32
        self.total_bytes_compressed += compressed_bytes
        self.reconstruction_times.append(reconstruction_time)

    def average_compression_ratio(self) -> float:
        """Calculate average compression ratio."""
        if self.total_bytes_compressed == 0:
            return 1.0
        return self.total_bytes_original / self.total_bytes_compressed

    def average_reconstruction_time(self) -> float:
        """Calculate average reconstruction time in milliseconds."""
        if not self.reconstruction_times:
            return 0.0
        return sum(self.reconstruction_times) / len(self.reconstruction_times) * 1000  # ms

    def report(self) -> Dict[str, float]:
        """
        Generate summary report.

        Returns:
            dict with statistics
        """
        stats = {
            "total_indices": self.total_indices,
            "original_KB": self.total_bytes_original / 1024,
            "compressed_KB": self.total_bytes_compressed / 1024,
            "avg_compression_ratio": self.average_compression_ratio(),
            "avg_reconstruction_ms": self.average_reconstruction_time(),
            "num_operations": len(self.reconstruction_times),
        }

        print("\nIndex Compression Summary:")
        print(f"  Total indices: {stats['total_indices']:,}")
        print(f"  Original size: {stats['original_KB']:.2f} KB")
        print(f"  Compressed size: {stats['compressed_KB']:.2f} KB")
        print(f"  Avg compression ratio: {stats['avg_compression_ratio']:.2f}×")
        print(f"  Avg reconstruction time: {stats['avg_reconstruction_ms']:.3f} ms")
        print(f"  Operations tracked: {stats['num_operations']}")

        return stats


# Helper functions


def _dtype_bytes(dtype: torch.dtype) -> int:
    """Get number of bytes for a dtype."""
    if dtype == torch.float32 or dtype == torch.int32:
        return 4
    elif dtype == torch.float64 or dtype == torch.int64:
        return 8
    elif dtype == torch.float16 or dtype == torch.bfloat16 or dtype == torch.int16:
        return 2
    elif dtype == torch.int8 or dtype == torch.uint8:
        return 1
    else:
        # Fallback: create dummy tensor
        return torch.zeros(1, dtype=dtype).element_size()


def index_dtype_recommendation(max_index: int, target: str = "memory") -> torch.dtype:
    """
    Recommend index dtype based on maximum index value.

    Args:
        max_index: Maximum index value needed
        target: 'memory' (minimize memory) or 'speed' (maximize MKL performance)

    Returns:
        Recommended torch dtype
    """
    int8_max = 2**7 - 1  # 127
    int16_max = 2**15 - 1  # 32,767
    int32_max = 2**31 - 1  # ~2.1 billion

    if target == "memory":
        if max_index <= int8_max:
            return torch.int8
        elif max_index <= int16_max:
            return torch.int16
        else:
            return torch.int32
    else:  # target == 'speed'
        # MKL/cuSPARSE prefer int32
        return torch.int32


def should_use_delta_encoding(indices: torch.Tensor, threshold_ratio: float = 1.5) -> bool:
    """
    Decide whether delta encoding is worthwhile.

    Args:
        indices: Sorted tensor of indices
        threshold_ratio: Minimum compression ratio to use delta encoding

    Returns:
        True if delta encoding is recommended
    """
    analysis = analyze_delta_compression_potential(indices)
    ratio = analysis["compression_ratio"]

    # Decision criteria
    if ratio >= threshold_ratio:
        return True
    elif analysis["avg_delta"] < 128:
        # Most deltas fit in 1 byte
        return True
    elif len(indices) > 0 and analysis["delta_histogram"]["< 128 (1 byte)"] / len(indices) > 0.8:
        # >80% of deltas fit in 1 byte
        return True
    else:
        return False
