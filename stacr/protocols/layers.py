import torch
import torch.nn as nn
from stacr.protocols.compression import Compressor
from stacr.protocols.reshape import Reshaper


class CompressedFunction(torch.autograd.Function):
    """
    Custom Autograd function.
    Forward: computes output, optionally compresses activations, stores only
    factors + weights/bias (never the dense activation).
    Backward: reconstructs activation from factors (if needed) to compute grad_weight.
    """

    @staticmethod
    def forward(ctx, input_tensor, weight, bias, config):
        method = (config or {}).get("method", "none")
        profiler = (config or {}).get("phase_profiler")
        quant_bits = (config or {}).get("factor_quantization_bits")
        if isinstance(quant_bits, str):
            try:
                quant_bits = int(quant_bits)
            except ValueError:
                quant_bits = None
        if quant_bits == 0:
            quant_bits = None
        in_features = weight.shape[1]
        leading_shape = input_tensor.shape[:-1]
        input_2d = input_tensor.reshape(-1, in_features)
        output_2d = input_2d.mm(weight.t())
        if bias is not None:
            output_2d = output_2d + bias.unsqueeze(0)
        output = output_2d.reshape(*leading_shape, weight.shape[0])
        ctx.method = method
        ctx.leading_shape = leading_shape
        ctx.config = config or {}
        saved = []
        factors_to_save = []
        debug_dump_path = ctx.config.get("debug_dump_ctx_path")
        ctx.has_bias = bias is not None
        if method == "tt":
            reshape_shape = ctx.config["reshape_shape"]
            rank = ctx.config["rank"]
            if profiler and profiler.enabled:
                with profiler.phase("compress"):
                    factors = Compressor.compress_tt_factors(input_2d, reshape_shape, rank)
            else:
                factors = Compressor.compress_tt_factors(input_2d, reshape_shape, rank)
            if quant_bits:
                quant = Compressor.quantize_tt_factors(factors, bits=quant_bits)
                factors_to_save = []
                for core_q, scale in zip(quant["cores"], quant["scales"]):
                    factors_to_save.extend([core_q, scale])
                ctx.factors_meta = {
                    "type": "tt",
                    "tensorization": tuple(reshape_shape),
                    "rank": rank,
                    "quantized": True,
                    "quant_bits": quant_bits,
                    "num_cores": len(quant["cores"]),
                    "core_shapes": quant.get("core_shapes"),
                    "core_numels": quant.get("core_numels"),
                }
            else:
                factors_to_save = list(factors)
                ctx.factors_meta = {
                    "type": "tt",
                    "tensorization": tuple(reshape_shape),
                    "rank": rank,
                    "quantized": False,
                }
        elif method == "tucker":
            reshape_shape = ctx.config["reshape_shape"]
            ranks = ctx.config["ranks"]
            if profiler and profiler.enabled:
                with profiler.phase("compress"):
                    (core, factors) = Compressor.compress_tucker_factors(
                        input_2d, reshape_shape, ranks
                    )
            else:
                (core, factors) = Compressor.compress_tucker_factors(input_2d, reshape_shape, ranks)
            if quant_bits:
                quant = Compressor.quantize_tucker_factors((core, factors), bits=quant_bits)
                factors_to_save = [quant["core_q"], quant["core_scale"]]
                for mat_q, mat_scale in zip(quant["matrices"], quant["matrix_scales"]):
                    factors_to_save.extend([mat_q, mat_scale])
                ctx.factors_meta = {
                    "type": "tucker",
                    "tensorization": tuple(reshape_shape),
                    "ranks": tuple(ranks),
                    "quantized": True,
                    "quant_bits": quant_bits,
                    "num_matrices": len(quant["matrices"]),
                    "core_shape": quant.get("core_shape"),
                    "core_numel": quant.get("core_numel"),
                    "matrix_shapes": quant.get("matrix_shapes"),
                    "matrix_numels": quant.get("matrix_numels"),
                }
            else:
                factors_to_save = [core] + list(factors)
                ctx.factors_meta = {
                    "type": "tucker",
                    "tensorization": tuple(reshape_shape),
                    "ranks": tuple(ranks),
                    "quantized": False,
                }
        elif method == "svd":
            rank = ctx.config["rank"]
            use_lowrank = ctx.config.get("use_lowrank_svd", False)
            if profiler and profiler.enabled:
                with profiler.phase("compress"):
                    (U_k, s_k, Vh_k) = Compressor.compress_svd_factors(
                        input_2d, rank, use_lowrank=use_lowrank
                    )
            else:
                (U_k, s_k, Vh_k) = Compressor.compress_svd_factors(
                    input_2d, rank, use_lowrank=use_lowrank
                )
            if quant_bits:
                quant = Compressor.quantize_svd_factors((U_k, s_k, Vh_k), bits=quant_bits)
                factors_to_save = [
                    quant["U_q"],
                    quant["U_scale"],
                    quant["s_k"],
                    quant["Vh_q"],
                    quant["Vh_scale"],
                ]
                ctx.factors_meta = {
                    "type": "svd",
                    "rank": rank,
                    "quantized": True,
                    "quant_bits": quant_bits,
                    "U_shape": quant.get("U_shape"),
                    "U_numel": quant.get("U_numel"),
                    "Vh_shape": quant.get("Vh_shape"),
                    "Vh_numel": quant.get("Vh_numel"),
                }
            else:
                factors_to_save = [U_k, s_k, Vh_k]
                ctx.factors_meta = {"type": "svd", "rank": rank, "quantized": False}
        else:
            saved.append(input_2d)
            ctx.factors_meta = {"type": "none"}
        saved.append(weight)
        if bias is not None:
            saved.append(bias)
        ctx.save_for_backward(*saved, *factors_to_save)
        ctx.n_saved_input = len(saved)
        ctx.n_factors = len(factors_to_save)
        if debug_dump_path:
            CompressedFunction._debug_dump(
                debug_dump_path, factors_to_save, ctx.factors_meta, input_2d
            )
        return output

    @staticmethod
    def backward(ctx, grad_output):
        saved = ctx.saved_tensors
        profiler = (ctx.config or {}).get("phase_profiler")
        input_saved = saved[0] if ctx.method == "none" else None
        weight = saved[ctx.n_saved_input - (2 if ctx.has_bias else 1)]
        bias = saved[ctx.n_saved_input - 1] if ctx.has_bias else None
        factor_tensors = saved[ctx.n_saved_input :]
        quant_bits = ctx.factors_meta.get("quant_bits") if hasattr(ctx, "factors_meta") else None
        grad_output_2d = grad_output.reshape(-1, grad_output.shape[-1])
        grad_input = grad_weight = grad_bias = None
        if ctx.needs_input_grad[0]:
            grad_input_2d = grad_output_2d.mm(weight)
            grad_input = grad_input_2d.reshape(*ctx.leading_shape, weight.shape[1])
        if ctx.needs_input_grad[1]:
            if ctx.method == "tt":
                if quant_bits:
                    num_cores = ctx.factors_meta.get("num_cores", 0)
                    cores = []
                    scales = []
                    for idx in range(num_cores):
                        cores.append(factor_tensors[2 * idx])
                        scales.append(factor_tensors[2 * idx + 1])
                    quant = {
                        "method": "tt",
                        "bits": quant_bits,
                        "cores": cores,
                        "scales": scales,
                        "core_shapes": ctx.factors_meta.get("core_shapes"),
                        "core_numels": ctx.factors_meta.get("core_numels"),
                    }
                    factors = Compressor.dequantize_tt_factors(quant)
                else:
                    factors = factor_tensors
                if profiler and profiler.enabled:
                    with profiler.phase("reconstruct"):
                        recon = Compressor.decompress_tt(factors)
                else:
                    recon = Compressor.decompress_tt(factors)
            elif ctx.method == "tucker":
                if quant_bits:
                    core_q = factor_tensors[0]
                    core_scale = factor_tensors[1]
                    matrices = []
                    scales = []
                    idx = 2
                    while idx < len(factor_tensors):
                        matrices.append(factor_tensors[idx])
                        scales.append(factor_tensors[idx + 1])
                        idx += 2
                    quant = {
                        "method": "tucker",
                        "bits": quant_bits,
                        "core_q": core_q,
                        "core_scale": core_scale,
                        "core_shape": ctx.factors_meta.get("core_shape"),
                        "core_numel": ctx.factors_meta.get("core_numel"),
                        "matrices": matrices,
                        "matrix_scales": scales,
                        "matrix_shapes": ctx.factors_meta.get("matrix_shapes"),
                        "matrix_numels": ctx.factors_meta.get("matrix_numels"),
                    }
                    (core, factors) = Compressor.dequantize_tucker_factors(quant)
                else:
                    core = factor_tensors[0]
                    factors = factor_tensors[1:]
                if profiler and profiler.enabled:
                    with profiler.phase("reconstruct"):
                        recon = Compressor.decompress_tucker((core, factors))
                else:
                    recon = Compressor.decompress_tucker((core, factors))
            elif ctx.method == "svd":
                if quant_bits:
                    (U_q, U_scale, s_k, Vh_q, Vh_scale) = factor_tensors[:5]
                    quant = {
                        "method": "svd",
                        "bits": quant_bits,
                        "U_q": U_q,
                        "U_scale": U_scale,
                        "s_k": s_k,
                        "Vh_q": Vh_q,
                        "Vh_scale": Vh_scale,
                        "U_shape": ctx.factors_meta.get("U_shape"),
                        "U_numel": ctx.factors_meta.get("U_numel"),
                        "Vh_shape": ctx.factors_meta.get("Vh_shape"),
                        "Vh_numel": ctx.factors_meta.get("Vh_numel"),
                    }
                    (U_k, s_k, Vh_k) = Compressor.dequantize_svd_factors(quant)
                else:
                    (U_k, s_k, Vh_k) = factor_tensors
                factor_native = ctx.config.get("factor_native_backward", False)
                if factor_native:
                    if profiler and profiler.enabled:
                        with profiler.phase("factor_native_grad"):
                            GtU = grad_output_2d.t().mm(U_k)
                            GtU_S = GtU * s_k.unsqueeze(0)
                            grad_weight = GtU_S.mm(Vh_k)
                    else:
                        GtU = grad_output_2d.t().mm(U_k)
                        GtU_S = GtU * s_k.unsqueeze(0)
                        grad_weight = GtU_S.mm(Vh_k)
                    recon = None
                elif profiler and profiler.enabled:
                    with profiler.phase("reconstruct"):
                        recon = (U_k * s_k.unsqueeze(0)).mm(Vh_k)
                else:
                    recon = (U_k * s_k.unsqueeze(0)).mm(Vh_k)
            else:
                recon = input_saved
            if ctx.method != "svd" or not ctx.config.get("factor_native_backward", False):
                input_recon_2d = Reshaper.to_matrix(recon)
                grad_weight = grad_output_2d.t().mm(input_recon_2d)
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output_2d.sum(0)
        return (grad_input, grad_weight, grad_bias, None)

    @staticmethod
    def _debug_dump(path, factors, meta, input_tensor):
        import json
        from pathlib import Path

        payload = {
            "meta": meta,
            "factors": [
                {"shape": list(f.shape), "numel": f.numel(), "dtype": str(f.dtype)} for f in factors
            ],
            "dense_activation_detected": any(
                (
                    f.numel() == input_tensor.numel() and f.shape == input_tensor.shape
                    for f in factors
                )
            ),
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a") as f:
            f.write(json.dumps(payload) + "\n")


class CompressedLinear(nn.Module):
    """
    Drop-in replacement for nn.Linear that uses compression.
    """

    def __init__(self, in_features, out_features, compression_config, bias=True):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.config = compression_config
        self._validate_config(in_features, compression_config)

    def forward(self, x):
        return CompressedFunction.apply(x, self.linear.weight, self.linear.bias, self.config)

    @staticmethod
    def _validate_config(in_features: int, config):
        if config is None:
            return
        if not isinstance(config, dict):
            raise TypeError("compression_config must be a dict")
        method = config.get("method", "none")
        if method == "none":
            return
        if method == "svd":
            if "rank" not in config:
                raise ValueError("SVD compression_config requires 'rank'")
            if config["rank"] <= 0:
                raise ValueError("SVD rank must be positive")
            quant_bits = config.get("factor_quantization_bits")
            if quant_bits is not None and quant_bits not in (4, 8):
                raise ValueError("factor_quantization_bits must be 4, 8, or None")
        elif method == "tt":
            for key in ("rank", "reshape_shape"):
                if key not in config:
                    raise ValueError(f"TT compression_config requires '{key}'")
            if len(config["reshape_shape"]) == 0:
                raise ValueError("TT reshape_shape must be non-empty")
            tt_rank = config["rank"]
            if isinstance(tt_rank, (list, tuple)):
                if any((r <= 0 for r in tt_rank)):
                    raise ValueError("TT ranks must all be positive")
            elif tt_rank <= 0:
                raise ValueError("TT rank must be positive")
            quant_bits = config.get("factor_quantization_bits")
            if quant_bits is not None and quant_bits not in (4, 8):
                raise ValueError("factor_quantization_bits must be 4, 8, or None")
        elif method == "tucker":
            for key in ("ranks", "reshape_shape"):
                if key not in config:
                    raise ValueError(f"Tucker compression_config requires '{key}'")
            if len(config["reshape_shape"]) == 0:
                raise ValueError("Tucker reshape_shape must be non-empty")
            if any((r <= 0 for r in config["ranks"])):
                raise ValueError("Tucker ranks must be positive")
            quant_bits = config.get("factor_quantization_bits")
            if quant_bits is not None and quant_bits not in (4, 8):
                raise ValueError("factor_quantization_bits must be 4, 8, or None")
        else:
            raise ValueError(f"Unknown compression method '{method}'")
