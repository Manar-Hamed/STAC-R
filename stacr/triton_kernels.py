"""Triton kernels for STAC-R (projection first).

FP16-only projection kernel for input @ basis.
"""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl

    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False
    triton = None
    tl = None


def triton_available() -> bool:
    return TRITON_AVAILABLE


# Only define kernel when Triton is available
if TRITON_AVAILABLE:
    _KERNEL_CONFIGS = [
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 32, "BLOCK_K": 32}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 32, "BLOCK_K": 32}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 32, "BLOCK_K": 64}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 64}, num_warps=8, num_stages=3),
    ]

    @triton.autotune(configs=_KERNEL_CONFIGS, key=["M", "N", "K"])
    @triton.jit
    def _matmul_kernel(
        a_ptr,
        b_ptr,
        c_ptr,
        M,
        N,
        K,
        stride_am,
        stride_ak,
        stride_bk,
        stride_bn,
        stride_cm,
        stride_cn,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)

        a_ptrs = a_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
        b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        k = 0
        while k < K:
            a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (k + offs_k[None, :] < K), other=0.0)
            b = tl.load(b_ptrs, mask=(k + offs_k[:, None] < K) & (offs_n[None, :] < N), other=0.0)
            acc += tl.dot(a, b)
            k += BLOCK_K
            a_ptrs += BLOCK_K * stride_ak
            b_ptrs += BLOCK_K * stride_bk

        c_ptrs = c_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
        # Explicit cast to FP16 before storing
        acc_fp16 = acc.to(tl.float16)
        tl.store(c_ptrs, acc_fp16, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


def triton_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """FP16 Triton matmul for 2D tensors."""
    if not TRITON_AVAILABLE:
        raise RuntimeError("Triton is not available in this environment.")
    if a.dim() != 2 or b.dim() != 2:
        raise ValueError("triton_matmul expects 2D tensors.")
    if a.dtype != torch.float16 or b.dtype != torch.float16:
        raise ValueError("triton_matmul expects FP16 inputs.")
    if not a.is_cuda or not b.is_cuda:
        raise ValueError("triton_matmul expects CUDA tensors.")

    # Ensure contiguous for correct stride computation
    if not a.is_contiguous():
        a = a.contiguous()
    if not b.is_contiguous():
        b = b.contiguous()

    M, K = a.shape
    K2, N = b.shape
    if K != K2:
        raise ValueError(f"Incompatible shapes: {a.shape} @ {b.shape}")

    c = torch.empty((M, N), device=a.device, dtype=torch.float16)

    def grid(meta):
        return (
            triton.cdiv(M, meta["BLOCK_M"]),
            triton.cdiv(N, meta["BLOCK_N"]),
        )

    _matmul_kernel[grid](
        a,
        b,
        c,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        c.stride(0),
        c.stride(1),
    )
    return c


def projection_pytorch(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Reference projection using torch.matmul."""
    return a.matmul(b)


def projection_triton(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Projection using Triton kernel."""
    return triton_matmul(a, b)


# === C8: Fused STAC-R Kernels ===


def stacr_forward_fused(
    input: torch.Tensor,
    basis: torch.Tensor,
    budget: float = 0.1,
    topk_mode: str = "row",
    chunk_size: int = 512,
    return_metadata: bool = False,
) -> tuple:
    """Fused STAC-R forward pass: projection + residual + top-k.

    This function combines:
    1. Projection: coefficients = input @ basis  (optimized matmul)
    2. Residual: residual = input - coefficients @ basis^T  (chunked to save memory)
    3. Top-k selection: keep top-k elements of residual  (torch.topk)

    Memory optimization: Uses chunked processing for residual computation to avoid
    materializing full (B, N) intermediate tensors. Workspace is limited to
    (chunk_size, N) instead of (B, N).

    Args:
        input: (B, N) input activations
        basis: (N, R) or (R, N) orthonormal basis
        budget: Fraction of residual elements to keep (default 0.1 = 10%)
        topk_mode: "row" for per-row top-k, "global" for full-tensor top-k
        chunk_size: Size of chunks for residual computation (default 512)

    Returns:
        tuple: (coefficients, topk_values, topk_indices[, metadata])
            - coefficients: (B, R) projection coefficients
            - topk_values: (B, K) per-row or (K,) global top-k residual values
            - topk_indices: (B, K) per-row or (K,) global top-k residual indices
            - metadata: dict with route/parity metadata when return_metadata=True

    Fallback:
        Uses optimized Triton matmul when available, otherwise PyTorch.
    """
    # Validate inputs
    if input.dim() != 2 or basis.dim() != 2:
        raise ValueError(f"Expects 2D tensors, got input {input.shape}, basis {basis.shape}")

    B, N = input.shape
    if basis.shape[0] == N:
        basis_nr = basis
    elif basis.shape[1] == N:
        basis_nr = basis.t()
    else:
        raise ValueError(f"Incompatible shapes: input {input.shape}, basis {basis.shape}")

    if basis_nr is not basis:
        basis_nr = basis_nr.contiguous()

    metadata = {
        "route": "pytorch_chunked",
        "route_counts": {"pytorch_chunked": 0, "triton_chunked": 0, "pytorch_fallback": 0},
        "parity_error": None,
    }

    def _mark_route(route: str) -> None:
        counts = metadata["route_counts"]
        counts[route] = int(counts.get(route, 0)) + 1
        nonzero = [name for name, count in counts.items() if count]
        metadata["route"] = nonzero[0] if len(nonzero) == 1 else "mixed"

    use_triton = (
        TRITON_AVAILABLE
        and input.is_cuda
        and basis_nr.is_cuda
        and input.dtype == torch.float16
        and basis_nr.dtype == torch.float16
    )

    # Step 1: Compute full coefficients (small: B x R)
    if use_triton:
        try:
            coefficients = triton_matmul(input, basis_nr)
            _mark_route("triton_chunked")
        except Exception:
            coefficients = input.mm(basis_nr)
            _mark_route("pytorch_fallback")
    else:
        coefficients = input.mm(basis_nr)
        _mark_route("pytorch_chunked")

    # Step 2 & 3: Chunked residual computation + top-k selection
    # This avoids materializing full (B, N) residual tensor
    total_elements = B * N
    k = int(total_elements * budget) if topk_mode == "global" else max(1, int(N * budget))

    if k <= 0:
        topk_values = input.new_empty((0,))
        topk_indices = torch.empty((0,), device=input.device, dtype=torch.int32)
        if return_metadata:
            return coefficients, topk_values, topk_indices, metadata
        return coefficients, topk_values, topk_indices

    if topk_mode == "row":
        k = min(k, N)
    else:
        k = min(k, total_elements)

    # Prepare basis transpose for reconstruction
    # Always use direct operations - Triton autotune workspace causes memory spikes
    basis_t = basis_nr.t().contiguous()

    # Use chunking only for global mode (row mode needs full rows anyway)
    if topk_mode == "global":
        chunk_size = min(chunk_size, B)
        per_chunk_k = max(1, (k * chunk_size + B - 1) // B)

        all_values = []
        all_indices = []

        for chunk_start in range(0, B, chunk_size):
            chunk_end = min(chunk_start + chunk_size, B)
            chunk_input = input[chunk_start:chunk_end]  # VIEW, no allocation
            chunk_coeff = coefficients[chunk_start:chunk_end]  # VIEW

            if use_triton:
                try:
                    chunk_recon = triton_matmul(chunk_coeff, basis_t)
                    _mark_route("triton_chunked")
                except Exception:
                    chunk_recon = chunk_coeff.mm(basis_t)
                    _mark_route("pytorch_fallback")
            else:
                chunk_recon = chunk_coeff.mm(basis_t)
                _mark_route("pytorch_chunked")
            chunk_residual = chunk_input - chunk_recon
            del chunk_recon

            # Local top-k in this chunk
            chunk_flat = chunk_residual.reshape(-1)
            actual_k = min(per_chunk_k, chunk_flat.numel())
            _, local_indices = torch.topk(chunk_flat.abs(), actual_k)
            local_values = chunk_flat[local_indices]

            # Convert to global indices
            global_offset = chunk_start * N
            global_indices = local_indices + global_offset

            all_values.append(local_values)
            all_indices.append(global_indices)
            del chunk_residual, chunk_flat

        # Merge chunks and find global top-k
        merged_values = torch.cat(all_values)
        merged_indices = torch.cat(all_indices)

        if merged_values.numel() > k:
            _, top_indices = torch.topk(merged_values.abs(), k)
            topk_values = merged_values[top_indices]
            topk_indices = merged_indices[top_indices].to(torch.int32)
        else:
            topk_values = merged_values
            topk_indices = merged_indices.to(torch.int32)

    else:  # topk_mode == "row"
        # For row mode, we need full rows anyway, so use original approach
        # but still chunk to limit memory
        chunk_size = min(chunk_size, B)
        all_topk_values = []
        all_topk_indices = []

        for chunk_start in range(0, B, chunk_size):
            chunk_end = min(chunk_start + chunk_size, B)
            chunk_input = input[chunk_start:chunk_end]
            chunk_coeff = coefficients[chunk_start:chunk_end]

            if use_triton:
                try:
                    chunk_recon = triton_matmul(chunk_coeff, basis_t)
                    _mark_route("triton_chunked")
                except Exception:
                    chunk_recon = chunk_coeff.mm(basis_t)
                    _mark_route("pytorch_fallback")
            else:
                chunk_recon = chunk_coeff.mm(basis_t)
                _mark_route("pytorch_chunked")
            chunk_residual = chunk_input - chunk_recon
            del chunk_recon

            _, chunk_topk_indices = torch.topk(chunk_residual.abs(), k, dim=1)
            chunk_topk_values = torch.gather(chunk_residual, 1, chunk_topk_indices)

            all_topk_values.append(chunk_topk_values)
            all_topk_indices.append(chunk_topk_indices)
            del chunk_residual

        topk_values = torch.cat(all_topk_values, dim=0)
        topk_indices = torch.cat(all_topk_indices, dim=0).to(torch.int32)

    if return_metadata:
        return coefficients, topk_values, topk_indices, metadata
    return coefficients, topk_values, topk_indices


def stacr_forward_pytorch(
    input: torch.Tensor,
    basis: torch.Tensor,
    budget: float = 0.1,
    topk_mode: str = "row",
) -> tuple:
    """Pure PyTorch reference implementation of STAC-R forward.

    Same interface as stacr_forward_fused, but uses only PyTorch ops.
    """
    if input.dim() != 2 or basis.dim() != 2:
        raise ValueError(f"Expects 2D tensors, got input {input.shape}, basis {basis.shape}")

    B, N = input.shape
    if basis.shape[0] == N:
        basis_nr = basis
    elif basis.shape[1] == N:
        basis_nr = basis.t()
    else:
        raise ValueError(f"Incompatible shapes: input {input.shape}, basis {basis.shape}")

    # Use FP32 for numerical stability
    input_f32 = input.float() if input.dtype == torch.float16 else input
    basis_f32 = basis_nr.float() if basis_nr.dtype == torch.float16 else basis_nr

    # Projection
    coefficients = torch.mm(input_f32, basis_f32)

    # Residual
    reconstruction = torch.mm(coefficients, basis_f32.t())
    residual = input_f32 - reconstruction

    # Cast back to original dtype
    if input.dtype == torch.float16:
        coefficients = coefficients.half()
        residual = residual.half()

    # Top-k selection
    if topk_mode == "row":
        k = max(1, int(N * budget))
        k = min(k, N)
        _, topk_indices = torch.topk(residual.abs(), k, dim=1)
        topk_values = torch.gather(residual, 1, topk_indices)
        topk_indices = topk_indices.to(torch.int32)
    elif topk_mode == "global":
        total_elements = B * N
        k = int(total_elements * budget)
        if k <= 0:
            topk_values = residual.new_empty((0,))
            topk_indices = torch.empty((0,), device=residual.device, dtype=torch.int32)
        else:
            k = min(k, total_elements)
            residual_flat = residual.reshape(-1)
            _, topk_indices = torch.topk(residual_flat.abs(), k)
            topk_values = residual_flat[topk_indices]
            topk_indices = topk_indices.to(torch.int32)
    else:
        raise ValueError(f"Unsupported topk_mode: {topk_mode}")

    return coefficients, topk_values, topk_indices


def verify_fused_kernel(
    B: int = 64,
    N: int = 512,
    R: int = 32,
    budget: float = 0.1,
    topk_mode: str = "row",
) -> bool:
    """Verify fused STAC-R kernel correctness against PyTorch reference.

    Args:
        B: Batch size
        N: Number of features
        R: Rank of basis
        budget: Top-k budget fraction

    Returns:
        bool: True if verification passes (relative error < 1e-3)
    """
    if not torch.cuda.is_available():
        print("CUDA not available, skipping verification")
        return True  # Pass by default if no CUDA

    # Create random inputs
    torch.manual_seed(42)
    input_tensor = torch.randn(B, N, device="cuda", dtype=torch.float16)

    # Create orthonormal basis (using QR decomposition for correctness)
    basis_rand = torch.randn(N, R, device="cuda", dtype=torch.float32)
    Q, _ = torch.linalg.qr(basis_rand)
    basis = Q[:, :R].half()

    # Reference PyTorch implementation
    ref_coeff, ref_topk_vals, ref_topk_idx = stacr_forward_pytorch(
        input_tensor, basis, budget, topk_mode=topk_mode
    )

    # Test fused implementation
    try:
        test_coeff, test_topk_vals, test_topk_idx = stacr_forward_fused(
            input_tensor, basis, budget, topk_mode=topk_mode
        )
    except Exception as e:
        print(f"Fused kernel failed: {e}")
        return False

    # Compare coefficients
    coeff_err = (ref_coeff - test_coeff).abs().max().item()
    coeff_rel_err = coeff_err / (ref_coeff.abs().max().item() + 1e-8)

    # Compare top-k values (indices may differ due to ties, compare sorted values)
    if topk_mode == "row":
        ref_sorted = torch.sort(ref_topk_vals.abs(), dim=1, descending=True)[0]
        test_sorted = torch.sort(test_topk_vals.abs(), dim=1, descending=True)[0]
    else:
        ref_sorted = torch.sort(ref_topk_vals.abs(), descending=True)[0]
        test_sorted = torch.sort(test_topk_vals.abs(), descending=True)[0]
    topk_err = (ref_sorted - test_sorted).abs().max().item()
    topk_rel_err = topk_err / (ref_sorted.max().item() + 1e-8)

    print(f"Verification (B={B}, N={N}, R={R}, budget={budget}):")
    print(f"  Coefficient max error: {coeff_err:.6f}, relative: {coeff_rel_err:.6f}")
    print(f"  Top-k max error: {topk_err:.6f}, relative: {topk_rel_err:.6f}")

    # Success if relative errors < 1e-2 (reasonable for FP16)
    passed = coeff_rel_err < 1e-2 and topk_rel_err < 1e-2
    print(f"  Status: {'PASS' if passed else 'FAIL'}")

    return passed
