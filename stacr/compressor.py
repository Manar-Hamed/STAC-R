from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import torch

from .oja import OjaBasis
from .residual import SparseResidual, select_topk_residual


@dataclass
class STACRConfig:
    rank: int = 8
    residual_budget: float = 0.005
    min_features: int = 128
    min_numel: int = 4096
    oja_lr: float = 0.01
    update_interval: int = 1
    warmup_steps: int = 0
    residual_mode: str = "topk"
    post_warmup_mode: str = "full"
    turbo_after_step: int | None = None
    basis_dtype: str = "float32"


@dataclass
class PackedActivation:
    shape: tuple[int, ...]
    dtype: torch.dtype
    coeff: torch.Tensor
    basis: torch.Tensor
    residual: SparseResidual | None


def _basis_dtype(name: str, activation_dtype: torch.dtype) -> torch.dtype:
    spec = str(name).lower()
    if spec in ("auto", "activation"):
        return activation_dtype
    if spec in ("float16", "fp16", "half"):
        return torch.float16
    if spec in ("bfloat16", "bf16"):
        return torch.bfloat16
    return torch.float32


class STACRCompressor:
    """Saved-tensor compressor with persistent Oja bases keyed by feature size.

    This is the storage-only path: it is useful for reconstruction-error telemetry and
    for wrapping graphs that are not built from linear modules. It is **not** the
    method evaluated in the paper. ``saved_tensors_hooks`` requires ``unpack`` to hand
    autograd a tensor of the original shape, so this path rebuilds a dense activation
    in backward and cannot express the factor-native contraction. For the measured
    method use ``stacr.surgery.apply_stacr``, which installs ``STACRLinear`` layers
    that contract the weight gradient directly from the stored factors.
    """

    def __init__(self, config: STACRConfig | None = None):
        self.config = config or STACRConfig()
        self.step = 0
        self._bases: dict[tuple[int, torch.device], OjaBasis] = {}

    def set_step(self, step: int) -> None:
        self.step = int(step)

    def advance(self) -> None:
        self.step += 1

    def _basis_for(self, tensor: torch.Tensor) -> OjaBasis:
        features = int(tensor.shape[-1])
        key = (features, tensor.device)
        if key not in self._bases:
            dtype = _basis_dtype(self.config.basis_dtype, tensor.dtype)
            self._bases[key] = OjaBasis(
                features, self.config.rank, dtype=dtype, device=tensor.device
            )
        return self._bases[key]

    @staticmethod
    def is_parameter(tensor: torch.Tensor) -> bool:
        """True for weights/biases rather than saved activations.

        Autograd saves parameters alongside activations, and a parameter is always a
        leaf that requires grad. Compressing one would make the input gradient
        ``G W`` approximate, which the method never does.
        """
        if isinstance(tensor, torch.nn.Parameter) or (tensor.requires_grad and tensor.is_leaf):
            return True
        # Linear autograd often saves a transposed view of the weight.
        base = getattr(tensor, "_base", None)
        return base is not None and STACRCompressor.is_parameter(base)

    def should_compress(self, tensor: torch.Tensor) -> bool:
        return (
            tensor.is_floating_point()
            and tensor.dim() >= 2
            and not self.is_parameter(tensor)
            and int(tensor.shape[-1]) >= int(self.config.min_features)
            and tensor.numel() >= int(self.config.min_numel)
        )

    def _effective_residual_mode(self) -> str:
        if self.config.turbo_after_step is not None and self.step >= int(
            self.config.turbo_after_step
        ):
            return "none"
        return self.config.residual_mode

    def pack(self, tensor: torch.Tensor) -> Any:
        if not self.should_compress(tensor) or self.step < int(self.config.warmup_steps):
            return tensor
        x = tensor.detach()
        x2d = x.reshape(-1, x.shape[-1])
        basis_state = self._basis_for(x)
        if self.step % max(1, int(self.config.update_interval)) == 0:
            basis_state.update(x2d, lr=float(self.config.oja_lr))
        basis = basis_state.shared(dtype=x.dtype)
        coeff = x2d.mm(basis).to(dtype=x.dtype)
        low_rank = coeff.mm(basis.t()).reshape_as(x)
        residual = None
        if self._effective_residual_mode() != "none":
            residual = select_topk_residual(x - low_rank, float(self.config.residual_budget))
        return PackedActivation(tuple(x.shape), x.dtype, coeff, basis, residual)

    def unpack(self, packed: Any) -> torch.Tensor:
        if not isinstance(packed, PackedActivation):
            return packed
        recon = packed.coeff.mm(packed.basis.t()).reshape(packed.shape)
        if packed.residual is not None:
            recon = recon + packed.residual.dense(device=recon.device, dtype=recon.dtype)
        return recon.to(dtype=packed.dtype)


@contextmanager
def stacr_saved_tensors(compressor: STACRCompressor):
    """Context manager for PyTorch saved-tensor hooks."""
    with torch.autograd.graph.saved_tensors_hooks(compressor.pack, compressor.unpack):
        yield compressor
