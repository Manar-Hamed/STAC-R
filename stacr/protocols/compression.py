import torch
import tensorly as tl
from tensorly.decomposition import tucker, tensor_train
from tensorly import tt_to_tensor, tucker_to_tensor
from stacr.protocols.reshape import Reshaper
from stacr.protocols.profiling import CudaProfile
import warnings

tl.set_backend("pytorch")


class Compressor:
    """
    Unified compressor for SVD, Tucker, and Tensor Train.
    Provides analysis helpers and factor-only utilities for autograd.
    """

    @staticmethod
    def _normalize_reshape_shape(matrix, reshape_shape):
        reshape_shape = tuple(reshape_shape)
        if matrix.dim() > 2 and matrix.shape[1:] == reshape_shape:
            return reshape_shape
        prod = 1
        for d in reshape_shape:
            if d <= 0:
                raise ValueError(f"Reshape dimensions must be positive, got {reshape_shape}")
            prod *= d
        feature_dim = matrix.shape[-1]
        full_flat = matrix.numel() // matrix.shape[0]
        if prod not in (feature_dim, full_flat):
            raise ValueError(
                f"Tensorization shape product {prod} must equal feature dim {feature_dim} or full tensor size {full_flat}. Got reshape_shape={reshape_shape}, matrix.shape={matrix.shape}"
            )
        return reshape_shape

    @staticmethod
    def _validate_inputs(matrix, reshape_shape=None, rank=None, variance_threshold=None):
        if matrix.numel() == 0:
            raise ValueError("Input matrix cannot be empty.")
        if matrix.shape[0] == 0:
            raise ValueError("Input matrix must have at least one sample.")
        if torch.count_nonzero(matrix) == 0:
            warnings.warn(
                "Input matrix contains only zeros; compression may be degenerate.", UserWarning
            )
        if variance_threshold is not None and (not 0 < variance_threshold <= 1):
            raise ValueError(f"variance_threshold must be in (0, 1], got {variance_threshold}")
        if reshape_shape is not None:
            prod = 1
            for d in reshape_shape:
                if d <= 0:
                    raise ValueError(f"Reshape dimensions must be positive, got {reshape_shape}")
                prod *= d
            feature_dim = matrix.shape[-1]
            full_flat = matrix.numel() // matrix.shape[0]
            if prod not in (feature_dim, full_flat):
                raise ValueError(
                    f"Reshape shape product {prod} must equal feature dim {feature_dim} or full tensor size {full_flat}"
                )
        if rank is not None:
            max_rank = min(matrix.shape[0], matrix.shape[-1])
            if rank > max_rank or rank <= 0:
                raise ValueError(f"Rank must be in [1, {max_rank}], got {rank}")

    @staticmethod
    def _ensure_same_device(tensors, context: str):
        devices = {t.device for t in tensors if torch.is_tensor(t)}
        if len(devices) > 1:
            raise ValueError(f"{context} requires all tensors on the same device, got {devices}")
        return devices.pop() if devices else None

    @staticmethod
    def _prepare_decomposition_input(tensor):
        """
        Prepare tensor for decomposition by upcasting fp16/bfloat16 to fp32.

        SVD (torch.linalg.svd) and tensorly decompositions don't support half precision.
        This helper centralizes the dtype conversion for all decomposition paths.

        Args:
            tensor: Input tensor (any dtype)

        Returns:
            tensor: Same tensor if fp32/fp64, otherwise upcast to fp32
        """
        if tensor.dtype in (torch.float16, torch.bfloat16):
            return tensor.float()
        return tensor

    @staticmethod
    def _decomposition_context(device):
        """
        Return a context manager that disables autocast for decomposition operations.

        This prevents AMP from casting intermediate operations back to fp16 during
        SVD/TT/Tucker decomposition, which would cause "not implemented for 'Half'" errors.

        Args:
            device: torch.device or None

        Returns:
            Context manager (torch.autocast with enabled=False on CUDA, nullcontext on CPU)
        """
        from contextlib import nullcontext

        if device is not None and device.type == "cuda":
            return torch.autocast(device_type="cuda", enabled=False)
        return nullcontext()

    @staticmethod
    def compress_svd(
        matrix, variance_threshold=0.9, profile: bool = False, use_lowrank: bool = False
    ):
        """SVD compression with variance-based rank selection.

        Note: use_lowrank=False by default here because variance_threshold needs
        full singular values. For fixed-rank compression, use compress_svd_factors
        which defaults to use_lowrank=True for memory efficiency.
        """
        Compressor._validate_inputs(matrix, variance_threshold=variance_threshold)
        matrix = Compressor._prepare_decomposition_input(matrix)
        with Compressor._decomposition_context(matrix.device):
            if profile:
                with CudaProfile("decompose"):
                    (U, S, Vh) = torch.linalg.svd(matrix, full_matrices=False)
            else:
                (U, S, Vh) = torch.linalg.svd(matrix, full_matrices=False)
        rank = Compressor.get_rank_from_variance(S, variance_threshold)
        U_k = U[:, :rank]
        S_k = S[:rank]
        Vh_k = Vh[:rank, :]
        recon = torch.matmul(U_k * S_k.unsqueeze(0), Vh_k)
        compressed_mem = U_k.numel() + S_k.numel() + Vh_k.numel()
        return (recon, compressed_mem, rank)

    @staticmethod
    def compress_svd_factors(matrix, rank, profile: bool = False, use_lowrank: bool = False):
        """Compress matrix using SVD, returning truncated factors.

        Args:
            matrix: Input matrix to decompose
            rank: Target rank for truncation
            profile: Whether to profile the decomposition
            use_lowrank: If True, use randomized SVD (torch.svd_lowrank) which has
                O(rank) workspace. If False (default), use full SVD which has
                O(min(m,n)^2) workspace but higher accuracy.
                Enable use_lowrank for memory-constrained scenarios (OOM testing).
        """
        Compressor._validate_inputs(matrix, rank=rank)
        matrix = Compressor._prepare_decomposition_input(matrix)
        with Compressor._decomposition_context(matrix.device):
            if use_lowrank:
                q = min(rank + 4, min(matrix.shape))
                if profile:
                    with CudaProfile("decompose"):
                        (U, S, V) = torch.svd_lowrank(matrix, q=q, niter=2)
                else:
                    (U, S, V) = torch.svd_lowrank(matrix, q=q, niter=2)
                Vh = V[:, :rank].transpose(-2, -1)
                return (U[:, :rank], S[:rank], Vh)
            else:
                if profile:
                    with CudaProfile("decompose"):
                        (U, S, Vh) = torch.linalg.svd(matrix, full_matrices=False)
                else:
                    (U, S, Vh) = torch.linalg.svd(matrix, full_matrices=False)
                return (U[:, :rank], S[:rank], Vh[:rank, :])

    @staticmethod
    def decompress_svd(factors, profile: bool = False):
        (U_k, s_k, Vh_k) = factors
        Compressor._ensure_same_device((U_k, s_k, Vh_k), "SVD reconstruction")
        if profile:
            with CudaProfile("reconstruct"):
                return torch.matmul(U_k * s_k.unsqueeze(0), Vh_k)
        return torch.matmul(U_k * s_k.unsqueeze(0), Vh_k)

    @staticmethod
    def compress_tucker(matrix, reshape_shape, variance_threshold=0.9):
        Compressor._validate_inputs(
            matrix, reshape_shape=reshape_shape, variance_threshold=variance_threshold
        )
        reshape_shape = Compressor._normalize_reshape_shape(matrix, reshape_shape)
        matrix = Compressor._prepare_decomposition_input(matrix)
        tensor = matrix.reshape((matrix.shape[0], *reshape_shape))
        with Compressor._decomposition_context(tensor.device):
            ranks = [tensor.shape[0]]
            for mode in range(1, len(tensor.shape)):
                unfolded = tl.unfold(tensor, mode)
                (_, S, _) = torch.linalg.svd(unfolded, full_matrices=False)
                ranks.append(Compressor.get_rank_from_variance(S, variance_threshold))
            try:
                (core, factors) = tucker(tensor, rank=ranks, init="svd")
            except Exception as exc:
                raise ValueError(
                    f"Tucker decomposition failed for ranks {ranks} and shape {tensor.shape}: {exc}"
                ) from exc
        recon_tensor = tucker_to_tensor((core, factors))
        recon = recon_tensor.reshape(matrix.shape)
        factors_mem = sum((f.numel() for f in factors))
        compressed_mem = core.numel() + factors_mem
        return (recon, compressed_mem, ranks)

    @staticmethod
    def compress_tucker_factors(matrix, reshape_shape, ranks, profile: bool = False):
        Compressor._validate_inputs(matrix, reshape_shape=reshape_shape)
        reshape_shape = Compressor._normalize_reshape_shape(matrix, reshape_shape)
        matrix = Compressor._prepare_decomposition_input(matrix)
        tensor = matrix.reshape((matrix.shape[0], *reshape_shape))
        dims = tensor.shape
        if len(ranks) != len(dims):
            raise ValueError(f"Tucker ranks length {len(ranks)} must equal tensor dims {len(dims)}")
        for dim, r in zip(dims, ranks):
            if r < 1 or r > dim:
                raise ValueError(
                    f"Tucker rank {r} must be between 1 and corresponding dimension {dim}"
                )
        with Compressor._decomposition_context(tensor.device):
            if profile:
                with CudaProfile("decompose"):
                    return tucker(tensor, rank=ranks, init="svd")
            try:
                return tucker(tensor, rank=ranks, init="svd")
            except Exception as exc:
                raise ValueError(
                    f"Tucker decomposition failed for ranks {ranks} and shape {tensor.shape}: {exc}"
                ) from exc

    @staticmethod
    def decompress_tucker(compressed_data, profile: bool = False):
        (core, factors) = compressed_data
        Compressor._ensure_same_device((core, *factors), "Tucker reconstruction")
        if profile:
            with CudaProfile("reconstruct"):
                return tucker_to_tensor((core, factors))
        return tucker_to_tensor((core, factors))

    @staticmethod
    def compress_tt(matrix, reshape_shape, rank_preset=10):
        Compressor._validate_inputs(matrix, reshape_shape=reshape_shape, rank=rank_preset)
        reshape_shape = Compressor._normalize_reshape_shape(matrix, reshape_shape)
        matrix = Compressor._prepare_decomposition_input(matrix)
        tensor = matrix.reshape((matrix.shape[0], *reshape_shape))
        with Compressor._decomposition_context(tensor.device):
            try:
                factors = tensor_train(tensor, rank=rank_preset)
            except Exception as e:
                raise RuntimeError(
                    f"Tensor Train compression failed: {e}\n  Input matrix shape: {matrix.shape}\n  Reshape shape: {reshape_shape}\n  Tensor shape: {tensor.shape}\n  Rank: {rank_preset}\n  Dtype: {tensor.dtype}\n  Device: {tensor.device}"
                ) from e
        recon_tensor = tt_to_tensor(factors)
        recon = recon_tensor.reshape(matrix.shape)
        compressed_mem = sum((f.numel() for f in factors))
        return (recon, compressed_mem, rank_preset)

    @staticmethod
    def compress_tt_factors(matrix, reshape_shape, rank, profile: bool = False):
        Compressor._validate_inputs(matrix, reshape_shape=reshape_shape)
        reshape_shape = Compressor._normalize_reshape_shape(matrix, reshape_shape)
        matrix = Compressor._prepare_decomposition_input(matrix)
        tensor = matrix.reshape((matrix.shape[0], *reshape_shape))
        if isinstance(rank, (list, tuple)):
            if len(rank) != tensor.dim() + 1:
                raise ValueError(
                    f"TT rank length {len(rank)} must equal tensor order + 1 ({tensor.dim() + 1})"
                )
            if rank[0] != 1 or rank[-1] != 1:
                raise ValueError(f"TT boundary ranks must be 1, got {rank[0]} and {rank[-1]}")
            for r in rank:
                if not isinstance(r, int) or r < 1:
                    raise ValueError(f"TT ranks must be positive integers, got {rank}")
        else:
            Compressor._validate_inputs(matrix, rank=rank)
        with Compressor._decomposition_context(tensor.device):
            if profile:
                with CudaProfile("decompose"):
                    return tensor_train(tensor, rank=rank)
            return tensor_train(tensor, rank=rank)

    @staticmethod
    def decompress_tt(factors, profile: bool = False):
        Compressor._ensure_same_device(factors, "TT reconstruction")
        if profile:
            with CudaProfile("reconstruct"):
                return tt_to_tensor(factors)
        return tt_to_tensor(factors)

    @staticmethod
    def _quantize_tensor_int8(tensor):
        """
        Quantize a tensor to int8 using symmetric per-tensor quantization.

        Args:
            tensor: Float tensor to quantize

        Returns:
            (quantized_tensor, scale): int8 tensor and float32 scale
        """
        if tensor.numel() == 0:
            return (
                tensor.to(torch.int8),
                torch.tensor(1.0, dtype=torch.float32, device=tensor.device),
            )
        abs_max = tensor.abs().max()
        if abs_max == 0:
            scale = torch.tensor(1.0, dtype=torch.float32, device=tensor.device)
        else:
            scale = abs_max / 127.0
        quantized = (tensor / scale).round().clamp(-128, 127).to(torch.int8)
        return (quantized, scale)

    @staticmethod
    def _dequantize_tensor_int8(quantized, scale):
        """
        Dequantize an int8 tensor back to float32.

        Args:
            quantized: int8 tensor
            scale: float32 scale factor

        Returns:
            Float32 tensor (quantized * scale)
        """
        return quantized.float() * scale

    @staticmethod
    def _quantize_tensor_int4(tensor):
        """
        Quantize a tensor to signed 4-bit values packed into uint8.

        Returns:
            packed: uint8 tensor containing two 4-bit values per byte
            scale: float32 scale factor
            orig_shape: original tensor shape (tuple)
            orig_numel: original number of elements (int)
        """
        if tensor.numel() == 0:
            packed = tensor.new_empty((0,), dtype=torch.uint8)
            scale = torch.tensor(1.0, dtype=torch.float32, device=tensor.device)
            return (packed, scale, tuple(tensor.shape), 0)
        abs_max = tensor.abs().max()
        if abs_max == 0:
            scale = torch.tensor(1.0, dtype=torch.float32, device=tensor.device)
        else:
            scale = abs_max / 7.0
        q = (tensor / scale).round().clamp(-8, 7).to(torch.int8)
        q_flat = q.reshape(-1)
        orig_numel = int(q_flat.numel())
        if orig_numel % 2 == 1:
            pad = torch.zeros(1, dtype=torch.int8, device=q_flat.device)
            q_flat = torch.cat([q_flat, pad], dim=0)
        q_u = (q_flat + 8).to(torch.uint8)
        lo = q_u[0::2]
        hi = q_u[1::2]
        packed = hi << 4 | lo
        return (packed, scale, tuple(tensor.shape), orig_numel)

    @staticmethod
    def _dequantize_tensor_int4(packed, scale, orig_shape, orig_numel):
        """
        Dequantize packed 4-bit values back to float32.
        """
        if packed.numel() == 0:
            return torch.empty(orig_shape, dtype=torch.float32, device=packed.device)
        packed_flat = packed.reshape(-1)
        lo = packed_flat & 15
        hi = packed_flat >> 4 & 15
        q_u = torch.stack((lo, hi), dim=1).reshape(-1)
        q = q_u.to(torch.int16) - 8
        if orig_numel is not None:
            q = q[:orig_numel]
        q = q.to(torch.float32) * scale
        return q.reshape(orig_shape)

    @staticmethod
    def quantize_svd_factors(factors, bits=8):
        """
        Quantize SVD factors (U, S, Vh) to reduce storage.

        For SVD, we quantize U and Vh to int8, but keep S in float32
        because singular values need high precision and are small.

        Args:
            factors: (U_k, s_k, Vh_k) tuple
            bits: Quantization bits (8 or 4, default 8)

        Returns:
            Quantized factors dict with metadata for dequantization
        """
        (U_k, s_k, Vh_k) = factors
        if bits == 8:
            (U_q, U_scale) = Compressor._quantize_tensor_int8(U_k)
            (Vh_q, Vh_scale) = Compressor._quantize_tensor_int8(Vh_k)
            return {
                "method": "svd",
                "bits": 8,
                "U_q": U_q,
                "U_scale": U_scale,
                "s_k": s_k,
                "Vh_q": Vh_q,
                "Vh_scale": Vh_scale,
            }
        if bits == 4:
            (U_q, U_scale, U_shape, U_numel) = Compressor._quantize_tensor_int4(U_k)
            (Vh_q, Vh_scale, Vh_shape, Vh_numel) = Compressor._quantize_tensor_int4(Vh_k)
            return {
                "method": "svd",
                "bits": 4,
                "U_q": U_q,
                "U_scale": U_scale,
                "U_shape": U_shape,
                "U_numel": U_numel,
                "s_k": s_k,
                "Vh_q": Vh_q,
                "Vh_scale": Vh_scale,
                "Vh_shape": Vh_shape,
                "Vh_numel": Vh_numel,
            }
        raise ValueError(f"Unsupported quantization bits: {bits}. Use 8 or 4.")

    @staticmethod
    def dequantize_svd_factors(quantized_factors):
        """
        Dequantize SVD factors back to float32.

        Args:
            quantized_factors: Dict from quantize_svd_factors

        Returns:
            (U_k, s_k, Vh_k) tuple in float32
        """
        required_keys = {"bits", "U_q", "U_scale", "s_k", "Vh_q", "Vh_scale", "method"}
        missing = required_keys - set(quantized_factors.keys())
        if missing:
            raise ValueError(f"Missing required keys in quantized_factors: {missing}")
        if quantized_factors["method"] != "svd":
            raise ValueError(f"Expected method='svd', got '{quantized_factors['method']}'")
        if quantized_factors["bits"] == 8:
            U_k = Compressor._dequantize_tensor_int8(
                quantized_factors["U_q"], quantized_factors["U_scale"]
            )
            Vh_k = Compressor._dequantize_tensor_int8(
                quantized_factors["Vh_q"], quantized_factors["Vh_scale"]
            )
            s_k = quantized_factors["s_k"]
            return (U_k, s_k, Vh_k)
        if quantized_factors["bits"] == 4:
            U_shape = quantized_factors.get("U_shape")
            Vh_shape = quantized_factors.get("Vh_shape")
            U_numel = quantized_factors.get("U_numel")
            Vh_numel = quantized_factors.get("Vh_numel")
            if U_shape is None or Vh_shape is None:
                raise ValueError("Missing U_shape/Vh_shape for int4 dequantization")
            U_k = Compressor._dequantize_tensor_int4(
                quantized_factors["U_q"], quantized_factors["U_scale"], tuple(U_shape), U_numel
            )
            Vh_k = Compressor._dequantize_tensor_int4(
                quantized_factors["Vh_q"], quantized_factors["Vh_scale"], tuple(Vh_shape), Vh_numel
            )
            s_k = quantized_factors["s_k"]
            return (U_k, s_k, Vh_k)
        raise ValueError(f"Unsupported bits: {quantized_factors['bits']}")

    @staticmethod
    def quantize_tt_factors(factors, bits=8):
        """
        Quantize TT cores to reduce storage.

        Each TT core is quantized separately with its own scale.

        Args:
            factors: List of TT cores
            bits: Quantization bits (8, default)

        Returns:
            Quantized factors dict with metadata
        """
        if bits == 8:
            quantized_cores = []
            scales = []
            for core in factors:
                (core_q, scale) = Compressor._quantize_tensor_int8(core)
                quantized_cores.append(core_q)
                scales.append(scale)
            return {"method": "tt", "bits": 8, "cores": quantized_cores, "scales": scales}
        if bits == 4:
            quantized_cores = []
            scales = []
            shapes = []
            numels = []
            for core in factors:
                (core_q, scale, shape, numel) = Compressor._quantize_tensor_int4(core)
                quantized_cores.append(core_q)
                scales.append(scale)
                shapes.append(shape)
                numels.append(numel)
            return {
                "method": "tt",
                "bits": 4,
                "cores": quantized_cores,
                "scales": scales,
                "core_shapes": shapes,
                "core_numels": numels,
            }
        raise ValueError(f"Unsupported quantization bits: {bits}. Use 8 or 4.")

    @staticmethod
    def dequantize_tt_factors(quantized_factors):
        """
        Dequantize TT cores back to float32.

        Args:
            quantized_factors: Dict from quantize_tt_factors

        Returns:
            List of TT cores in float32
        """
        required_keys = {"bits", "cores", "scales", "method"}
        missing = required_keys - set(quantized_factors.keys())
        if missing:
            raise ValueError(f"Missing required keys in quantized_factors: {missing}")
        if quantized_factors["method"] != "tt":
            raise ValueError(f"Expected method='tt', got '{quantized_factors['method']}'")
        if quantized_factors["bits"] == 8:
            cores = []
            for core_q, scale in zip(quantized_factors["cores"], quantized_factors["scales"]):
                cores.append(Compressor._dequantize_tensor_int8(core_q, scale))
            return cores
        if quantized_factors["bits"] == 4:
            shapes = quantized_factors.get("core_shapes")
            numels = quantized_factors.get("core_numels")
            if shapes is None or numels is None:
                raise ValueError("Missing core_shapes/core_numels for int4 dequantization")
            cores = []
            for core_q, scale, shape, numel in zip(
                quantized_factors["cores"], quantized_factors["scales"], shapes, numels
            ):
                cores.append(Compressor._dequantize_tensor_int4(core_q, scale, tuple(shape), numel))
            return cores
        raise ValueError(f"Unsupported bits: {quantized_factors['bits']}")

    @staticmethod
    def quantize_tucker_factors(factors, bits=8):
        """
        Quantize Tucker factors (core + factor matrices) to reduce storage.

        Args:
            factors: (core, [factor_matrices]) tuple
            bits: Quantization bits (8, default)

        Returns:
            Quantized factors dict with metadata
        """
        (core, factor_matrices) = factors
        if bits == 8:
            (core_q, core_scale) = Compressor._quantize_tensor_int8(core)
            quantized_matrices = []
            matrix_scales = []
            for matrix in factor_matrices:
                (mat_q, mat_scale) = Compressor._quantize_tensor_int8(matrix)
                quantized_matrices.append(mat_q)
                matrix_scales.append(mat_scale)
            return {
                "method": "tucker",
                "bits": 8,
                "core_q": core_q,
                "core_scale": core_scale,
                "matrices": quantized_matrices,
                "matrix_scales": matrix_scales,
            }
        if bits == 4:
            (core_q, core_scale, core_shape, core_numel) = Compressor._quantize_tensor_int4(core)
            quantized_matrices = []
            matrix_scales = []
            matrix_shapes = []
            matrix_numels = []
            for matrix in factor_matrices:
                (mat_q, mat_scale, mat_shape, mat_numel) = Compressor._quantize_tensor_int4(matrix)
                quantized_matrices.append(mat_q)
                matrix_scales.append(mat_scale)
                matrix_shapes.append(mat_shape)
                matrix_numels.append(mat_numel)
            return {
                "method": "tucker",
                "bits": 4,
                "core_q": core_q,
                "core_scale": core_scale,
                "core_shape": core_shape,
                "core_numel": core_numel,
                "matrices": quantized_matrices,
                "matrix_scales": matrix_scales,
                "matrix_shapes": matrix_shapes,
                "matrix_numels": matrix_numels,
            }
        raise ValueError(f"Unsupported quantization bits: {bits}. Use 8 or 4.")

    @staticmethod
    def dequantize_tucker_factors(quantized_factors):
        """
        Dequantize Tucker factors back to float32.

        Args:
            quantized_factors: Dict from quantize_tucker_factors

        Returns:
            (core, [factor_matrices]) tuple in float32
        """
        required_keys = {"bits", "core_q", "core_scale", "matrices", "matrix_scales", "method"}
        missing = required_keys - set(quantized_factors.keys())
        if missing:
            raise ValueError(f"Missing required keys in quantized_factors: {missing}")
        if quantized_factors["method"] != "tucker":
            raise ValueError(f"Expected method='tucker', got '{quantized_factors['method']}'")
        if quantized_factors["bits"] == 8:
            core = Compressor._dequantize_tensor_int8(
                quantized_factors["core_q"], quantized_factors["core_scale"]
            )
            factor_matrices = []
            for mat_q, scale in zip(
                quantized_factors["matrices"], quantized_factors["matrix_scales"]
            ):
                factor_matrices.append(Compressor._dequantize_tensor_int8(mat_q, scale))
            return (core, factor_matrices)
        if quantized_factors["bits"] == 4:
            core_shape = quantized_factors.get("core_shape")
            core_numel = quantized_factors.get("core_numel")
            matrix_shapes = quantized_factors.get("matrix_shapes")
            matrix_numels = quantized_factors.get("matrix_numels")
            if core_shape is None or matrix_shapes is None or matrix_numels is None:
                raise ValueError("Missing shapes for int4 dequantization")
            core = Compressor._dequantize_tensor_int4(
                quantized_factors["core_q"],
                quantized_factors["core_scale"],
                tuple(core_shape),
                core_numel,
            )
            factor_matrices = []
            for mat_q, scale, shape, numel in zip(
                quantized_factors["matrices"],
                quantized_factors["matrix_scales"],
                matrix_shapes,
                matrix_numels,
            ):
                factor_matrices.append(
                    Compressor._dequantize_tensor_int4(mat_q, scale, tuple(shape), numel)
                )
            return (core, factor_matrices)
        raise ValueError(f"Unsupported bits: {quantized_factors['bits']}")

    @staticmethod
    def get_quantized_storage_bytes(quantized_factors):
        """
        Calculate storage bytes for quantized factors.

        Args:
            quantized_factors: Dict from quantize_*_factors

        Returns:
            Total bytes for quantized storage (including scales)
        """
        method = quantized_factors["method"]
        if method == "svd":
            U_q = quantized_factors["U_q"]
            Vh_q = quantized_factors["Vh_q"]
            U_bytes = U_q.numel() * U_q.element_size()
            Vh_bytes = Vh_q.numel() * Vh_q.element_size()
            s_bytes = quantized_factors["s_k"].numel() * quantized_factors["s_k"].element_size()
            scale_bytes = (
                quantized_factors["U_scale"].numel() * quantized_factors["U_scale"].element_size()
                + quantized_factors["Vh_scale"].numel()
                * quantized_factors["Vh_scale"].element_size()
            )
            return U_bytes + Vh_bytes + s_bytes + scale_bytes
        elif method == "tt":
            core_bytes = sum((c.numel() * c.element_size() for c in quantized_factors["cores"]))
            scale_bytes = sum((s.numel() * s.element_size() for s in quantized_factors["scales"]))
            return core_bytes + scale_bytes
        elif method == "tucker":
            core_q = quantized_factors["core_q"]
            core_bytes = core_q.numel() * core_q.element_size()
            matrix_bytes = sum(
                (m.numel() * m.element_size() for m in quantized_factors["matrices"])
            )
            scale_bytes = quantized_factors["core_scale"].numel() * quantized_factors[
                "core_scale"
            ].element_size() + sum(
                (s.numel() * s.element_size() for s in quantized_factors["matrix_scales"])
            )
            return core_bytes + matrix_bytes + scale_bytes
        else:
            raise ValueError(f"Unknown method: {method}")

    @staticmethod
    def get_rank_from_variance(eigenvalues, threshold):
        if eigenvalues.numel() == 0:
            raise ValueError("eigenvalues must be non-empty")
        eigenvalues_sq = eigenvalues**2
        total_energy = torch.sum(eigenvalues_sq)
        if total_energy <= 1e-12:
            warnings.warn(
                "Total energy is near zero; returning rank=1 to avoid division by zero.",
                UserWarning,
            )
            return 1
        cumulative_energy = torch.cumsum(eigenvalues_sq, dim=0)
        explained = cumulative_energy / total_energy
        if explained[-1] < threshold:
            return len(eigenvalues)
        k = torch.searchsorted(explained, threshold).item() + 1
        return k

    @staticmethod
    def get_memory_usage(obj):
        if torch.is_tensor(obj):
            return obj.numel()
        elif isinstance(obj, (list, tuple)):
            return sum((x.numel() for x in obj))
        return 0
