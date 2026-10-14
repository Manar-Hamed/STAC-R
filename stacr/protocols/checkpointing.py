"""Compress saved checkpoint boundaries with a low-rank codec.

The block's forward uses its input unchanged. Backward replays the block on a
reconstructed input, so both parameter and input gradients can be approximate.
Each packed boundary retains its own immutable basis version. Supported inputs
are registered decoder-block boundaries with residual_mode="none" and optional
8-bit coefficients. This is an opt-in checkpoint composition experiment.
"""

from __future__ import annotations
import copy
import contextlib
from dataclasses import dataclass, replace
import functools
import math
import threading
from typing import Any, Dict, Iterable, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.utils.checkpoint as torch_checkpoint
from stacr.protocols.baselines.amp import resolve_amp_dtype

_DTYPE_NAME_TO_TORCH = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def _torch_dtype_name(dtype: torch.dtype) -> str:
    for name, value in _DTYPE_NAME_TO_TORCH.items():
        if value == dtype:
            return name
    return str(dtype).replace("torch.", "")


def _resolve_ckpt_basis_dtype(mode: str, amp_mode: str, device: torch.device) -> torch.dtype:
    if mode != "auto":
        return _DTYPE_NAME_TO_TORCH[mode]
    (amp_dtype, _) = resolve_amp_dtype(amp_mode, device.type)
    if amp_dtype in (torch.float16, torch.bfloat16):
        return amp_dtype
    return torch.float32


def _default_ckpt_rank(model_family: str, hidden_size: int) -> int:
    explicit = {"gpt2": 96, "llama": 64, "bert": 64}.get(model_family)
    if explicit is not None:
        return min(hidden_size, explicit)
    return min(hidden_size, 64)


def _resolve_ckpt_refresh_energy_threshold(
    model_family: str, configured: float, mode: str
) -> float:
    if mode == "auto" and model_family == "llama":
        return 0.4
    return float(configured)


def _tensor_matches_target(candidate: torch.Tensor, target: torch.Tensor) -> bool:
    if not candidate.is_floating_point() or not target.is_floating_point():
        return False
    if candidate is target:
        return True
    if candidate.device != target.device:
        return False
    if tuple(candidate.shape) != tuple(target.shape):
        return False
    if tuple(candidate.stride()) != tuple(target.stride()):
        return False
    if candidate.storage_offset() != target.storage_offset():
        return False
    try:
        return candidate.untyped_storage().data_ptr() == target.untyped_storage().data_ptr()
    except RuntimeError:
        return False


def _safe_max_abs(tensor: Optional[torch.Tensor]) -> Optional[float]:
    if tensor is None:
        return None
    data = tensor.detach()
    if data.numel() == 0:
        return 0.0
    abs_data = data.abs()
    max_value = abs_data.amax()
    if torch.isfinite(max_value):
        return float(max_value.to(torch.float32).item())
    if torch.isinf(max_value):
        return float("inf")
    if torch.isnan(abs_data).all():
        return float("nan")
    finite_max = torch.nan_to_num(abs_data, nan=0.0).amax()
    if torch.isfinite(finite_max):
        return float(finite_max.to(torch.float32).item())
    if torch.isinf(finite_max):
        return float("inf")
    return float("nan")


@dataclass
class STACRCheckpointConfig:
    enabled: bool = False
    target: str = "decoder_block_input"
    rank: Optional[int] = None
    warmup_steps: int = 500
    residual_mode: str = "none"
    residual_budget: float = 0.005
    refresh_trigger: str = "residual_energy"
    refresh_period: int = 200
    refresh_energy_threshold_mode: str = "auto"
    refresh_energy_threshold: float = 0.15
    min_numel: int = 1 << 20
    basis_dtype: str = "auto"
    quantize_coefficients: bool = False
    quantize_bits: int = 8
    quantize_residuals: bool = True
    residual_quantize_bits: int = 4
    allow_inner_stacr: bool = False
    planner: str = "heuristic"


@dataclass(frozen=True)
class STACRCheckpointSpec:
    site_id: str
    module_type: str
    model_family: str
    hidden_size: int
    rank: int
    residual_mode: str
    residual_budget: float
    basis_dtype: torch.dtype
    quantize_coefficients: bool
    quantize_bits: int
    quantize_residuals: bool
    residual_quantize_bits: int
    warmup_steps: int
    refresh_trigger: str
    refresh_period: int
    refresh_energy_threshold: float
    target_arg_index: int
    min_numel: int


@dataclass
class CompressedBoundaryRecord:
    site_id: str
    original_shape: Tuple[int, ...]
    original_dtype: torch.dtype
    device: torch.device
    is_dense_fallback: bool
    dense_tensor: Optional[torch.Tensor] = None
    coeff: Optional[torch.Tensor] = None
    coeff_scale: Optional[torch.Tensor] = None
    coeff_quantized: bool = False
    basis: Optional[torch.Tensor] = None


@dataclass(frozen=True)
class CheckpointSiteCandidate:
    name: str
    module: nn.Module
    spec: STACRCheckpointSpec


@dataclass
class ResolvedCheckpointSite:
    module: nn.Module
    spec: STACRCheckpointSpec
    codec: "CheckpointBoundaryCodec"
    target_tensor: Optional[torch.Tensor]
    target_arg_index: int


class CheckpointBoundaryCodec(nn.Module):
    """Per-site persistent boundary codec."""

    def __init__(self, spec: STACRCheckpointSpec):
        super().__init__()
        self.spec = spec
        self.register_buffer(
            "basis", torch.zeros(spec.hidden_size, spec.rank, dtype=spec.basis_dtype)
        )
        self.register_buffer("_basis_initialized", torch.tensor(False, dtype=torch.bool))
        self.register_buffer("_step_counter", torch.tensor(0, dtype=torch.int64))
        self.register_buffer("_basis_refresh_count", torch.tensor(0, dtype=torch.int64))
        self.register_buffer("_basis_init_refresh_count", torch.tensor(0, dtype=torch.int64))
        self.register_buffer("_basis_warmup_refresh_count", torch.tensor(0, dtype=torch.int64))
        self.register_buffer("_basis_trigger_refresh_count", torch.tensor(0, dtype=torch.int64))
        self.register_buffer("_pack_count", torch.tensor(0, dtype=torch.int64))
        self.register_buffer("_unpack_count", torch.tensor(0, dtype=torch.int64))
        self.register_buffer("_fallback_count", torch.tensor(0, dtype=torch.int64))
        self.register_buffer("_compressed_pack_count", torch.tensor(0, dtype=torch.int64))
        self.register_buffer("_duplicate_pack_count", torch.tensor(0, dtype=torch.int64))
        self.register_buffer("_energy_ewma", torch.tensor(0.0, dtype=torch.float32))
        self.register_buffer("_last_energy", torch.tensor(0.0, dtype=torch.float32))
        self.register_buffer("_raw_bytes_total", torch.tensor(0, dtype=torch.int64))
        self.register_buffer("_stored_bytes_total", torch.tensor(0, dtype=torch.int64))
        self._fallback_reasons: Dict[str, int] = {}
        self._first_nonfinite_event: Optional[Dict[str, Any]] = None
        self._last_input_max_abs: Optional[float] = None
        self._last_coeff_max_abs: Optional[float] = None
        self._last_reconstruction_max_abs: Optional[float] = None
        self._last_residual_max_abs: Optional[float] = None
        self._last_basis_max_abs: Optional[float] = None

    @property
    def pack_count(self) -> int:
        return int(self._pack_count.item())

    @property
    def unpack_count(self) -> int:
        return int(self._unpack_count.item())

    @property
    def basis_refresh_count(self) -> int:
        return int(self._basis_refresh_count.item())

    @property
    def step_count(self) -> int:
        return int(self._step_counter.item())

    @property
    def basis_initialized(self) -> bool:
        return bool(self._basis_initialized.item())

    def _flatten(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor.reshape(-1, tensor.shape[-1])

    def _record_storage(self, *, raw_bytes: int, stored_bytes: int, compressed: bool) -> None:
        self._raw_bytes_total.add_(int(raw_bytes))
        self._stored_bytes_total.add_(int(stored_bytes))
        if compressed:
            self._compressed_pack_count.add_(1)

    def _ensure_device(self, device: torch.device) -> None:
        if self.basis.device != device:
            self.to(device=device)

    def _apply(self, fn):
        super()._apply(fn)
        return self

    def note_duplicate_pack(self, tensor: torch.Tensor) -> None:
        self._duplicate_pack_count.add_(1)
        self._record_fallback(reason="duplicate_target_save")
        dense_bytes = int(tensor.numel() * tensor.element_size())
        self._record_storage(raw_bytes=dense_bytes, stored_bytes=dense_bytes, compressed=False)

    def _record_fallback(self, *, reason: str) -> None:
        self._fallback_count.add_(1)
        self._fallback_reasons[reason] = int(self._fallback_reasons.get(reason, 0)) + 1

    def _record_first_nonfinite(
        self,
        *,
        reason: str,
        stage: str,
        input_tensor: Optional[torch.Tensor] = None,
        coeff_tensor: Optional[torch.Tensor] = None,
        reconstruction_tensor: Optional[torch.Tensor] = None,
        residual_tensor: Optional[torch.Tensor] = None,
        basis_tensor: Optional[torch.Tensor] = None,
    ) -> None:
        if self._first_nonfinite_event is not None:
            return
        self._first_nonfinite_event = {
            "step": self.step_count + 1,
            "reason": reason,
            "stage": stage,
            "input_max_abs": _safe_max_abs(input_tensor),
            "coeff_max_abs": _safe_max_abs(coeff_tensor),
            "reconstruction_max_abs": _safe_max_abs(reconstruction_tensor),
            "residual_max_abs": _safe_max_abs(residual_tensor),
            "basis_max_abs": _safe_max_abs(basis_tensor),
        }

    def _record_last_pack_observation(
        self,
        *,
        input_tensor: Optional[torch.Tensor] = None,
        coeff_tensor: Optional[torch.Tensor] = None,
        reconstruction_tensor: Optional[torch.Tensor] = None,
        residual_tensor: Optional[torch.Tensor] = None,
        basis_tensor: Optional[torch.Tensor] = None,
    ) -> None:
        self._last_input_max_abs = _safe_max_abs(input_tensor)
        self._last_coeff_max_abs = _safe_max_abs(coeff_tensor)
        self._last_reconstruction_max_abs = _safe_max_abs(reconstruction_tensor)
        self._last_residual_max_abs = _safe_max_abs(residual_tensor)
        self._last_basis_max_abs = _safe_max_abs(basis_tensor)

    def _dense_fallback(self, tensor: torch.Tensor, *, reason: str) -> CompressedBoundaryRecord:
        self._record_fallback(reason=reason)
        dense_bytes = int(tensor.numel() * tensor.element_size())
        self._record_storage(raw_bytes=dense_bytes, stored_bytes=dense_bytes, compressed=False)
        return CompressedBoundaryRecord(
            site_id=self.spec.site_id,
            original_shape=tuple(tensor.shape),
            original_dtype=tensor.dtype,
            device=tensor.device,
            is_dense_fallback=True,
            dense_tensor=tensor,
        )

    def _dense_passthrough(self, tensor: torch.Tensor) -> CompressedBoundaryRecord:
        return CompressedBoundaryRecord(
            site_id=self.spec.site_id,
            original_shape=tuple(tensor.shape),
            original_dtype=tensor.dtype,
            device=tensor.device,
            is_dense_fallback=True,
            dense_tensor=tensor,
        )

    def _compute_basis(self, flat: torch.Tensor) -> torch.Tensor:
        h = flat.shape[-1]
        rank = min(self.spec.rank, flat.shape[0], h)
        work = flat.detach().to(torch.float32)
        if rank <= 0:
            return torch.zeros(h, self.spec.rank, device=flat.device, dtype=self.spec.basis_dtype)
        try:
            (_, _, vh) = torch.linalg.svd(work, full_matrices=False)
            basis = vh[:rank].transpose(0, 1)
        except RuntimeError:
            (q, _) = torch.linalg.qr(work.transpose(0, 1), mode="reduced")
            basis = q[:, :rank]
        full_basis = torch.zeros(h, self.spec.rank, device=flat.device, dtype=torch.float32)
        full_basis[:, :rank] = basis[:, :rank]
        if rank < self.spec.rank:
            fill = min(self.spec.rank - rank, h - rank)
            if fill > 0:
                eye = torch.eye(h, device=flat.device, dtype=torch.float32)
                if rank > 0:
                    projected = eye - basis[:, :rank] @ (basis[:, :rank].transpose(0, 1) @ eye)
                else:
                    projected = eye
                (q_comp, _) = torch.linalg.qr(projected, mode="reduced")
                full_basis[:, rank : rank + fill] = q_comp[:, :fill]
        return full_basis.to(self.spec.basis_dtype)

    def _update_basis(self, flat: torch.Tensor, *, reason: str) -> None:
        new_basis = self._compute_basis(flat)
        # Packed records may still need the previous basis for backward.
        self.basis = new_basis.detach()
        self._basis_initialized.fill_(True)
        self._basis_refresh_count.add_(1)
        if reason == "warmup":
            self._basis_warmup_refresh_count.add_(1)
        elif reason == "trigger":
            self._basis_trigger_refresh_count.add_(1)
        else:
            self._basis_init_refresh_count.add_(1)

    def _should_refresh(self, energy: float) -> bool:
        trigger = self.spec.refresh_trigger
        if trigger == "off":
            return False
        if trigger == "period":
            step_after_warmup = max(self.step_count - self.spec.warmup_steps, 0)
            return (
                step_after_warmup > 0 and step_after_warmup % max(self.spec.refresh_period, 1) == 0
            )
        if trigger == "residual_energy":
            return energy > self.spec.refresh_energy_threshold
        if trigger == "hybrid":
            step_after_warmup = max(self.step_count - self.spec.warmup_steps, 0)
            periodic = (
                step_after_warmup > 0 and step_after_warmup % max(self.spec.refresh_period, 1) == 0
            )
            return periodic or energy > self.spec.refresh_energy_threshold
        return False

    def _quantize_coefficients(
        self, coeff: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], bool]:
        if not self.spec.quantize_coefficients:
            return (coeff.to(self.basis.dtype), None, False)
        if self.spec.quantize_bits != 8:
            raise ValueError(
                f"C1 checkpoint coefficient quantization supports only 8-bit coefficients; got {self.spec.quantize_bits}."
            )
        reduce_dims = tuple(range(max(coeff.dim() - 1, 0)))
        max_abs = coeff.detach().abs().to(torch.float32)
        if reduce_dims:
            max_abs = max_abs.amax(dim=reduce_dims, keepdim=True)
        safe_max = torch.where(
            torch.isfinite(max_abs) & (max_abs > 0), max_abs, torch.ones_like(max_abs)
        )
        scale = safe_max / 127.0
        quantized = torch.clamp(torch.round(coeff.to(torch.float32) / scale), -127, 127).to(
            torch.int8
        )
        return (quantized, scale, True)

    def _dequantize_coefficients(
        self, coeff: torch.Tensor, scale: Optional[torch.Tensor], quantized: bool
    ) -> torch.Tensor:
        if not quantized:
            return coeff
        assert scale is not None
        return coeff.to(torch.float32) * scale.to(torch.float32)

    def _packed_coeff_bytes(self, coeff: torch.Tensor, coeff_scale: Optional[torch.Tensor]) -> int:
        total = int(coeff.numel() * coeff.element_size())
        if coeff_scale is not None:
            total += int(coeff_scale.numel() * coeff_scale.element_size())
        return total

    def pack_tensor(self, tensor: torch.Tensor, *, training: bool) -> CompressedBoundaryRecord:
        if not training:
            return self._dense_passthrough(tensor)
        self._pack_count.add_(1)
        self._ensure_device(tensor.device)
        try:
            self._record_last_pack_observation(input_tensor=tensor)
            if self.spec.residual_mode != "none":
                return self._dense_fallback(tensor, reason="residual_mode_unsupported")
            if not torch.isfinite(tensor).all():
                self._record_first_nonfinite(
                    reason="input_nonfinite", stage="input", input_tensor=tensor
                )
                return self._dense_fallback(tensor, reason="input_nonfinite")
            flat = self._flatten(tensor)
            flat_fp32 = flat.to(torch.float32)
            if training and self.step_count < self.spec.warmup_steps:
                self._update_basis(flat, reason="warmup")
                self._record_last_pack_observation(input_tensor=tensor, basis_tensor=self.basis)
                return self._dense_fallback(tensor, reason="warmup_dense")
            if not self.basis_initialized:
                self._update_basis(flat, reason="init")
            if not torch.isfinite(self.basis).all():
                self._record_last_pack_observation(input_tensor=tensor, basis_tensor=self.basis)
                self._record_first_nonfinite(
                    reason="basis_nonfinite", stage="basis", basis_tensor=self.basis
                )
                return self._dense_fallback(tensor, reason="basis_nonfinite")
            basis_fp32 = self.basis.to(torch.float32)
            coeff_fp32 = flat_fp32 @ basis_fp32
            if not torch.isfinite(coeff_fp32).all():
                self._record_last_pack_observation(
                    input_tensor=tensor, coeff_tensor=coeff_fp32, basis_tensor=self.basis
                )
                self._record_first_nonfinite(
                    reason="projection_nonfinite",
                    stage="coefficients",
                    input_tensor=tensor,
                    coeff_tensor=coeff_fp32,
                )
                return self._dense_fallback(tensor, reason="projection_nonfinite")
            coeff_matrix = coeff_fp32.reshape(*tensor.shape[:-1], self.spec.rank)
            (stored_coeff, coeff_scale, coeff_quantized) = self._quantize_coefficients(coeff_matrix)
            if coeff_scale is not None and (not torch.isfinite(coeff_scale).all()):
                self._record_last_pack_observation(
                    input_tensor=tensor, coeff_tensor=coeff_matrix, basis_tensor=self.basis
                )
                self._record_first_nonfinite(
                    reason="coeff_scale_nonfinite",
                    stage="coefficients",
                    input_tensor=tensor,
                    coeff_tensor=coeff_matrix,
                )
                return self._dense_fallback(tensor, reason="coeff_scale_nonfinite")
            reconstructed = (coeff_fp32 @ basis_fp32.transpose(0, 1)).reshape_as(tensor)
            if not torch.isfinite(reconstructed).all():
                self._record_last_pack_observation(
                    input_tensor=tensor,
                    coeff_tensor=coeff_matrix,
                    reconstruction_tensor=reconstructed,
                    basis_tensor=self.basis,
                )
                self._record_first_nonfinite(
                    reason="reconstruction_nonfinite",
                    stage="reconstruction",
                    input_tensor=tensor,
                    coeff_tensor=coeff_matrix,
                    reconstruction_tensor=reconstructed,
                )
                return self._dense_fallback(tensor, reason="reconstruction_nonfinite")
            reconstructed_fp32 = reconstructed.to(torch.float32)
            residual_fp32 = tensor.to(torch.float32) - reconstructed_fp32
            self._record_last_pack_observation(
                input_tensor=tensor,
                coeff_tensor=coeff_matrix,
                reconstruction_tensor=reconstructed,
                residual_tensor=residual_fp32,
                basis_tensor=self.basis,
            )
            error = residual_fp32.pow(2).sum()
            denom = tensor.to(torch.float32).pow(2).sum().clamp_min(1e-12)
            energy = float((error / denom).item())
            if not math.isfinite(energy):
                self._record_first_nonfinite(
                    reason="energy_nonfinite",
                    stage="reconstruction",
                    input_tensor=tensor,
                    coeff_tensor=coeff_matrix,
                    reconstruction_tensor=reconstructed,
                )
                return self._dense_fallback(tensor, reason="energy_nonfinite")
            self._last_energy.fill_(energy)
            self._energy_ewma.mul_(0.9).add_(0.1 * energy)
            should_refresh = training and self._should_refresh(energy)
            if should_refresh:
                self._update_basis(flat, reason="trigger")
                return self._dense_fallback(tensor, reason="trigger_refresh_dense")
            raw_bytes = int(tensor.numel() * tensor.element_size())
            stored_bytes = self._packed_coeff_bytes(stored_coeff, coeff_scale)
            self._record_storage(raw_bytes=raw_bytes, stored_bytes=stored_bytes, compressed=True)
            return CompressedBoundaryRecord(
                site_id=self.spec.site_id,
                original_shape=tuple(tensor.shape),
                original_dtype=tensor.dtype,
                device=tensor.device,
                is_dense_fallback=False,
                coeff=stored_coeff,
                coeff_scale=coeff_scale,
                coeff_quantized=coeff_quantized,
                basis=self.basis,
            )
        finally:
            if training:
                self._step_counter.add_(1)

    def unpack_record(self, packed: CompressedBoundaryRecord) -> torch.Tensor:
        self._unpack_count.add_(1)
        self._ensure_device(packed.device)
        if packed.is_dense_fallback:
            assert packed.dense_tensor is not None
            return packed.dense_tensor
        assert packed.coeff is not None
        coeff = self._dequantize_coefficients(
            packed.coeff, packed.coeff_scale, packed.coeff_quantized
        )
        coeff = coeff.reshape(-1, self.spec.rank).to(torch.float32)
        if packed.basis is None:
            raise ValueError("Compressed checkpoint record is missing its saved basis")
        restored = coeff @ packed.basis.to(device=packed.device, dtype=torch.float32).transpose(
            0, 1
        )
        return restored.reshape(packed.original_shape).to(
            dtype=packed.original_dtype, device=packed.device
        )

    def stats(self) -> Dict[str, Any]:
        first_nonfinite = self._first_nonfinite_event or {}
        return {
            "site_id": self.spec.site_id,
            "module_type": self.spec.module_type,
            "hidden_size": self.spec.hidden_size,
            "rank": self.spec.rank,
            "pack_count": self.pack_count,
            "unpack_count": self.unpack_count,
            "compressed_pack_count": int(self._compressed_pack_count.item()),
            "duplicate_pack_count": int(self._duplicate_pack_count.item()),
            "fallback_count": int(self._fallback_count.item()),
            "basis_refresh_count": self.basis_refresh_count,
            "step_count": self.step_count,
            "basis_initialized": self.basis_initialized,
            "basis_dtype": _torch_dtype_name(self.basis.dtype),
            "energy_ewma": float(self._energy_ewma.item()),
            "last_energy": float(self._last_energy.item()),
            "warmup_steps": self.spec.warmup_steps,
            "refresh_energy_threshold": float(self.spec.refresh_energy_threshold),
            "raw_bytes_total": int(self._raw_bytes_total.item()),
            "stored_bytes_total": int(self._stored_bytes_total.item()),
            "basis_persistent_bytes": int(self.basis.numel() * self.basis.element_size()),
            "basis_init_refresh_count": int(self._basis_init_refresh_count.item()),
            "basis_warmup_refresh_count": int(self._basis_warmup_refresh_count.item()),
            "basis_trigger_refresh_count": int(self._basis_trigger_refresh_count.item()),
            "fallback_reasons": dict(self._fallback_reasons),
            "first_nonfinite_step": first_nonfinite.get("step"),
            "first_nonfinite_reason": first_nonfinite.get("reason"),
            "first_nonfinite_stage": first_nonfinite.get("stage"),
            "first_nonfinite_input_max_abs": first_nonfinite.get("input_max_abs"),
            "first_nonfinite_coeff_max_abs": first_nonfinite.get("coeff_max_abs"),
            "first_nonfinite_reconstruction_max_abs": first_nonfinite.get("reconstruction_max_abs"),
            "first_nonfinite_residual_max_abs": first_nonfinite.get("residual_max_abs"),
            "first_nonfinite_basis_max_abs": first_nonfinite.get("basis_max_abs"),
            "last_input_max_abs": self._last_input_max_abs,
            "last_coeff_max_abs": self._last_coeff_max_abs,
            "last_reconstruction_max_abs": self._last_reconstruction_max_abs,
            "last_residual_max_abs": self._last_residual_max_abs,
            "last_basis_max_abs": self._last_basis_max_abs,
        }


class CheckpointSiteRegistry:
    def __init__(self):
        self._lock = threading.Lock()
        self._specs_by_module_id: Dict[int, STACRCheckpointSpec] = {}
        self._codecs_by_site_id: Dict[str, CheckpointBoundaryCodec] = {}

    def clear(self) -> None:
        with self._lock:
            self._specs_by_module_id.clear()
            self._codecs_by_site_id.clear()

    def register(self, module: nn.Module, spec: STACRCheckpointSpec) -> None:
        with self._lock:
            self._specs_by_module_id[id(module)] = spec
            self._codecs_by_site_id[spec.site_id] = CheckpointBoundaryCodec(spec)

    def get(self, module: Optional[nn.Module]) -> Optional[STACRCheckpointSpec]:
        if module is None:
            return None
        with self._lock:
            return self._specs_by_module_id.get(id(module))

    def codec_for(self, site_id: str) -> Optional[CheckpointBoundaryCodec]:
        with self._lock:
            return self._codecs_by_site_id.get(site_id)

    def specs(self) -> Iterable[STACRCheckpointSpec]:
        with self._lock:
            return list(self._specs_by_module_id.values())

    def items(self) -> Dict[str, CheckpointBoundaryCodec]:
        with self._lock:
            return dict(self._codecs_by_site_id)

    def __len__(self) -> int:
        with self._lock:
            return len(self._specs_by_module_id)


_GLOBAL_C1_REGISTRY = CheckpointSiteRegistry()
_THREAD_LOCAL_RUNTIME = threading.local()


def get_global_checkpoint_site_registry() -> CheckpointSiteRegistry:
    return _GLOBAL_C1_REGISTRY


class STACRCheckpointRuntime:
    def __init__(self, registry: Optional[CheckpointSiteRegistry] = None):
        self.registry = registry if registry is not None else get_global_checkpoint_site_registry()
        self._active = False
        self.site_count_total = 0
        self.planner_mode = "off"
        self.planner_summary: Optional[Dict[str, Any]] = None
        self.planner_decisions: Optional[List[Dict[str, Any]]] = None

    @property
    def is_active(self) -> bool:
        return self._active

    def activate(self) -> None:
        self._active = True
        _THREAD_LOCAL_RUNTIME.runtime = self

    def deactivate(self, *, clear_registry: bool = False) -> None:
        self._active = False
        if getattr(_THREAD_LOCAL_RUNTIME, "runtime", None) is self:
            _THREAD_LOCAL_RUNTIME.runtime = None
        if clear_registry:
            self.registry.clear()

    def resolve_site(
        self, function: Any, args: Tuple[Any, ...]
    ) -> Optional[ResolvedCheckpointSite]:
        module = self._resolve_checkpoint_module(function)
        spec = self.registry.get(module)
        if spec is None:
            return None
        (target_tensor, target_arg_index) = self.identify_target_tensor(spec, args)
        if target_tensor is None or target_arg_index is None:
            return None
        codec = self.registry.codec_for(spec.site_id)
        if codec is None:
            return None
        return ResolvedCheckpointSite(
            module=module,
            spec=spec,
            codec=codec,
            target_tensor=target_tensor,
            target_arg_index=target_arg_index,
        )

    def _resolve_checkpoint_module(self, function: Any) -> Optional[nn.Module]:
        current = function
        while True:
            module = getattr(current, "__self__", None)
            if isinstance(module, nn.Module):
                return module
            if not isinstance(current, functools.partial):
                return None
            current = current.func

    def identify_target_tensor(
        self, spec: STACRCheckpointSpec, args: Tuple[Any, ...]
    ) -> Tuple[Optional[torch.Tensor], Optional[int]]:
        preferred = None
        preferred_index = None
        if 0 <= spec.target_arg_index < len(args):
            candidate = args[spec.target_arg_index]
            if self._is_target_candidate(candidate, spec):
                preferred = candidate
                preferred_index = spec.target_arg_index
        if preferred is not None:
            return (preferred, preferred_index)
        matches = [
            (idx, arg) for (idx, arg) in enumerate(args) if self._is_target_candidate(arg, spec)
        ]
        if len(matches) == 1:
            (idx, tensor) = matches[0]
            return (tensor, idx)
        return (None, None)

    def _is_target_candidate(self, value: Any, spec: STACRCheckpointSpec) -> bool:
        if not isinstance(value, torch.Tensor):
            return False
        if not value.is_floating_point():
            return False
        if not value.requires_grad:
            return False
        if value.dim() not in (2, 3):
            return False
        if value.shape[-1] != spec.hidden_size:
            return False
        return value.numel() >= spec.min_numel

    def stats(self) -> Dict[str, Any]:
        per_site = {site_id: codec.stats() for (site_id, codec) in self.registry.items().items()}
        engaged_sites = sum((1 for site in per_site.values() if int(site.get("pack_count", 0)) > 0))
        compressed_sites = sum(
            (1 for site in per_site.values() if int(site.get("compressed_pack_count", 0)) > 0)
        )
        return {
            "active": self.is_active,
            "site_count_total": self.site_count_total,
            "site_count_registered": len(self.registry),
            "site_count_active": engaged_sites,
            "site_count_engaged": engaged_sites,
            "site_count_compressed": compressed_sites,
            "planner_mode": self.planner_mode,
            "planner_summary": self.planner_summary,
            "planner_decisions": self.planner_decisions,
            "per_site": per_site,
        }


def get_active_c1_runtime() -> Optional[STACRCheckpointRuntime]:
    runtime = getattr(_THREAD_LOCAL_RUNTIME, "runtime", None)
    if runtime is None or not runtime.is_active:
        return None
    return runtime


def clear_active_c1_runtime(*, clear_registry: bool = False) -> None:
    runtime = get_active_c1_runtime()
    if runtime is not None:
        runtime.deactivate(clear_registry=clear_registry)
    elif clear_registry:
        get_global_checkpoint_site_registry().clear()


def _serialize_packed_record(
    record: CompressedBoundaryRecord,
) -> Tuple[Dict[str, Any], List[torch.Tensor]]:
    tensor_parts: List[torch.Tensor] = []
    if record.is_dense_fallback:
        assert record.dense_tensor is not None
        tensor_parts.append(record.dense_tensor)
    else:
        assert record.coeff is not None
        tensor_parts.append(record.coeff)
        if record.coeff_scale is not None:
            tensor_parts.append(record.coeff_scale)
        assert record.basis is not None
        tensor_parts.append(record.basis)
    meta = {
        "site_id": record.site_id,
        "original_shape": tuple(record.original_shape),
        "original_dtype": record.original_dtype,
        "device": record.device,
        "is_dense_fallback": record.is_dense_fallback,
        "coeff_quantized": record.coeff_quantized,
        "has_coeff_scale": record.coeff_scale is not None,
    }
    return (meta, tensor_parts)


def _deserialize_packed_record(
    meta: Dict[str, Any], tensor_parts: Tuple[torch.Tensor, ...]
) -> CompressedBoundaryRecord:
    if meta["is_dense_fallback"]:
        dense_tensor = tensor_parts[0]
        return CompressedBoundaryRecord(
            site_id=meta["site_id"],
            original_shape=tuple(meta["original_shape"]),
            original_dtype=meta["original_dtype"],
            device=meta["device"],
            is_dense_fallback=True,
            dense_tensor=dense_tensor,
        )
    coeff = tensor_parts[0]
    coeff_scale = tensor_parts[1] if meta.get("has_coeff_scale") else None
    return CompressedBoundaryRecord(
        site_id=meta["site_id"],
        original_shape=tuple(meta["original_shape"]),
        original_dtype=meta["original_dtype"],
        device=meta["device"],
        is_dense_fallback=False,
        coeff=coeff,
        coeff_scale=coeff_scale,
        coeff_quantized=bool(meta.get("coeff_quantized", False)),
        basis=tensor_parts[2 if meta.get("has_coeff_scale") else 1],
    )


def _checkpoint_supports_autocast(device_type: str, device_module: Any) -> bool:
    helper = getattr(torch_checkpoint, "_supports_autocast", None)
    if callable(helper):
        return bool(helper(device_type))
    amp = getattr(device_module, "amp", None)
    return bool(amp is not None and hasattr(amp, "autocast"))


def _autocast_context(device_type: str, kwargs: Dict[str, Any], device_module: Any):
    amp_root = getattr(torch, "amp", None)
    if amp_root is not None and hasattr(amp_root, "autocast"):
        return amp_root.autocast(device_type, **kwargs)
    if device_type == "cpu":
        return torch.cpu.amp.autocast(**kwargs)
    return device_module.amp.autocast(**kwargs)


def _is_cache_like(value: Any) -> bool:
    value_type = type(value)
    module_name = getattr(value_type, "__module__", "")
    class_name = getattr(value_type, "__name__", "")
    if module_name.startswith("transformers.cache_utils"):
        return True
    if callable(getattr(value, "to_legacy_cache", None)):
        return True
    return class_name.endswith("Cache")


def _snapshot_non_tensor_value(value: Any) -> Any:
    if not _is_cache_like(value):
        return value
    legacy_export = getattr(value, "to_legacy_cache", None)
    restore = getattr(type(value), "from_legacy_cache", None)
    if callable(legacy_export) and callable(restore):
        return restore(legacy_export())
    try:
        return copy.deepcopy(value)
    except Exception as exc:
        raise RuntimeError(
            f"C1 could not snapshot mutable checkpoint value of type {type(value).__name__}."
        ) from exc


def _encode_checkpoint_value(
    value: Any,
    *,
    resolved: ResolvedCheckpointSite,
    saved_parts: List[torch.Tensor],
    target_state: Dict[str, bool],
):
    if torch.is_tensor(value):
        if _tensor_matches_target(value, resolved.target_tensor):
            if target_state.get("seen", False):
                resolved.codec.note_duplicate_pack(value)
                start = len(saved_parts)
                saved_parts.append(value)
                return ("tensor", (start, 1))
            target_state["seen"] = True
            packed = resolved.codec.pack_tensor(value, training=bool(resolved.module.training))
            (meta, tensor_parts) = _serialize_packed_record(packed)
            start = len(saved_parts)
            saved_parts.extend(tensor_parts)
            return ("compressed", (start, len(tensor_parts), meta, bool(value.requires_grad)))
        start = len(saved_parts)
        saved_parts.append(value)
        return ("tensor", (start, 1))
    if isinstance(value, tuple):
        return (
            "tuple",
            tuple(
                (
                    _encode_checkpoint_value(
                        item, resolved=resolved, saved_parts=saved_parts, target_state=target_state
                    )
                    for item in value
                )
            ),
        )
    if isinstance(value, list):
        return (
            "list",
            [
                _encode_checkpoint_value(
                    item, resolved=resolved, saved_parts=saved_parts, target_state=target_state
                )
                for item in value
            ],
        )
    if isinstance(value, dict):
        return (
            "dict",
            [
                (
                    key,
                    _encode_checkpoint_value(
                        item, resolved=resolved, saved_parts=saved_parts, target_state=target_state
                    ),
                )
                for (key, item) in value.items()
            ],
        )
    return ("value", _snapshot_non_tensor_value(value))


def _decode_checkpoint_value(
    entry: Any, saved: Tuple[torch.Tensor, ...], resolved: ResolvedCheckpointSite
):
    (kind, payload) = entry
    if kind == "tensor":
        (start, count) = payload
        return saved[start : start + count][0]
    if kind == "compressed":
        (start, count, meta, requires_grad) = payload
        tensor_parts = tuple(saved[start : start + count])
        record = _deserialize_packed_record(meta, tensor_parts)
        restored = resolved.codec.unpack_record(record)
        restored.requires_grad_(bool(requires_grad))
        return restored
    if kind == "tuple":
        return tuple((_decode_checkpoint_value(item, saved, resolved) for item in payload))
    if kind == "list":
        return [_decode_checkpoint_value(item, saved, resolved) for item in payload]
    if kind == "dict":
        return {key: _decode_checkpoint_value(item, saved, resolved) for (key, item) in payload}
    return payload


def _detach_checkpoint_value(value: Any):
    if torch.is_tensor(value):
        detached = value.detach()
        detached.requires_grad_(value.requires_grad)
        return detached
    if isinstance(value, tuple):
        return tuple((_detach_checkpoint_value(item) for item in value))
    if isinstance(value, list):
        return [_detach_checkpoint_value(item) for item in value]
    if isinstance(value, dict):
        return {key: _detach_checkpoint_value(item) for (key, item) in value.items()}
    return value


def _flatten_checkpoint_values(
    values: Tuple[Any, ...],
) -> Tuple[Tuple[Any, ...], Tuple[torch.Tensor, ...]]:
    flat_tensors: List[torch.Tensor] = []

    def _encode(value: Any):
        if torch.is_tensor(value):
            index = len(flat_tensors)
            flat_tensors.append(value)
            return ("tensor", index)
        if isinstance(value, tuple):
            return ("tuple", tuple((_encode(item) for item in value)))
        if isinstance(value, list):
            return ("list", [_encode(item) for item in value])
        if isinstance(value, dict):
            return ("dict", [(key, _encode(item)) for (key, item) in value.items()])
        return ("value", value)

    tree = tuple((_encode(value) for value in values))
    return (tree, tuple(flat_tensors))


def _unflatten_checkpoint_values(
    tree: Tuple[Any, ...], flat_tensors: Tuple[torch.Tensor, ...]
) -> Tuple[Any, ...]:

    def _decode(entry: Any):
        (kind, payload) = entry
        if kind == "tensor":
            return flat_tensors[payload]
        if kind == "tuple":
            return tuple((_decode(item) for item in payload))
        if kind == "list":
            return [_decode(item) for item in payload]
        if kind == "dict":
            return {key: _decode(item) for (key, item) in payload}
        return payload

    return tuple((_decode(entry) for entry in tree))


def _collect_tensor_leaves(value: Any, leaves: List[torch.Tensor]) -> None:
    if torch.is_tensor(value):
        leaves.append(value)
        return
    if isinstance(value, tuple):
        for item in value:
            _collect_tensor_leaves(item, leaves)
        return
    if isinstance(value, list):
        for item in value:
            _collect_tensor_leaves(item, leaves)
        return
    if isinstance(value, dict):
        for item in value.values():
            _collect_tensor_leaves(item, leaves)


def _collect_detached_input_leaves(values: Tuple[Any, ...]) -> Tuple[torch.Tensor, ...]:
    leaves: List[torch.Tensor] = []
    for value in values:
        _collect_tensor_leaves(value, leaves)
    return tuple(leaves)


def _normalize_checkpoint_callable(function: Any) -> Tuple[Any, Tuple[Any, ...], Dict[str, Any]]:
    prefix_args: List[Any] = []
    bound_kwargs: Dict[str, Any] = {}
    current = function
    while isinstance(current, functools.partial):
        if current.keywords:
            bound_kwargs = {**current.keywords, **bound_kwargs}
        if current.args:
            prefix_args = list(current.args) + prefix_args
        current = current.func
    if current is not function:
        return (current, tuple(prefix_args), bound_kwargs)
    return (function, (), {})


class _STACRCheckpointFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        run_function,
        preserve_rng_state: bool,
        resolved: ResolvedCheckpointSite,
        input_tree: Tuple[Any, ...],
        *flat_args,
    ):
        torch_checkpoint.check_backward_validity(flat_args)
        ctx.run_function = run_function
        ctx.preserve_rng_state = preserve_rng_state
        # Site metadata is needed during backward, but retaining its target
        # tensor would keep the dense boundary alive alongside its coefficients.
        ctx.resolved = replace(resolved, target_tensor=None)
        ctx.device = torch_checkpoint._infer_device_type(*flat_args)
        (ctx.device_autocast_kwargs, ctx.cpu_autocast_kwargs) = (
            torch_checkpoint._get_autocast_kwargs(ctx.device)
        )
        (prefix_args, bound_kwargs, args) = _unflatten_checkpoint_values(input_tree, flat_args)
        if preserve_rng_state:
            ctx.fwd_cpu_state = torch.get_rng_state()
            ctx.had_device_in_fwd = False
            device_module = torch_checkpoint._get_device_module(ctx.device)
            if getattr(device_module, "_initialized", False):
                ctx.had_device_in_fwd = True
                (ctx.fwd_devices, ctx.fwd_device_states) = torch_checkpoint.get_device_states(
                    *flat_args
                )
        saved_parts: List[torch.Tensor] = []
        target_state = {"seen": False}
        ctx.input_entries = [
            _encode_checkpoint_value(
                arg, resolved=resolved, saved_parts=saved_parts, target_state=target_state
            )
            for arg in (prefix_args, bound_kwargs, args)
        ]
        ctx.save_for_backward(*saved_parts)
        with torch.no_grad():
            outputs = run_function(*prefix_args, *args, **bound_kwargs)
        return outputs

    @staticmethod
    def backward(ctx, *grad_outputs):
        if not torch.autograd._is_checkpoint_valid():
            raise RuntimeError(
                "Checkpointing is not compatible with .grad() or when an `inputs` parameter is passed to .backward(). Please use .backward() and do not pass its `inputs` argument."
            )
        saved = ctx.saved_tensors
        inputs = tuple(
            (_decode_checkpoint_value(entry, saved, ctx.resolved) for entry in ctx.input_entries)
        )
        (prefix_args, bound_kwargs, args) = inputs
        rng_devices = []
        if ctx.preserve_rng_state and ctx.had_device_in_fwd:
            rng_devices = ctx.fwd_devices
        with torch.random.fork_rng(
            devices=rng_devices, enabled=ctx.preserve_rng_state, device_type=ctx.device
        ):
            if ctx.preserve_rng_state:
                torch.set_rng_state(ctx.fwd_cpu_state)
                if ctx.had_device_in_fwd:
                    torch_checkpoint.set_device_states(ctx.fwd_devices, ctx.fwd_device_states)
            (detached_prefix_args, detached_bound_kwargs, detached_args) = (
                _detach_checkpoint_value(value) for value in inputs
            )
            device_module = torch_checkpoint._get_device_module(ctx.device)
            supports_device_autocast = _checkpoint_supports_autocast(ctx.device, device_module)
            device_autocast_ctx = (
                _autocast_context(ctx.device, ctx.device_autocast_kwargs, device_module)
                if supports_device_autocast
                else contextlib.nullcontext()
            )
            from stacr.protocols.baselines.checkpointing import get_global_recomputation_guard

            guard = get_global_recomputation_guard()
            with (
                torch.enable_grad(),
                guard.recomputation_context(),
                device_autocast_ctx,
                _autocast_context("cpu", ctx.cpu_autocast_kwargs, torch.cpu),
            ):
                outputs = ctx.run_function(
                    *detached_prefix_args, *detached_args, **detached_bound_kwargs
                )
        if isinstance(outputs, torch.Tensor):
            outputs = (outputs,)
        outputs_with_grad = []
        grad_outputs_with_grad = []
        for output, grad_output in zip(outputs, grad_outputs):
            if torch.is_tensor(output) and output.requires_grad:
                outputs_with_grad.append(output)
                grad_outputs_with_grad.append(grad_output)
        if len(outputs_with_grad) == 0:
            raise RuntimeError(
                "none of output has requires_grad=True, this checkpoint() is not necessary"
            )
        torch.autograd.backward(outputs_with_grad, grad_outputs_with_grad)
        grads = tuple(
            (
                leaf.grad
                for leaf in _collect_detached_input_leaves(
                    (detached_prefix_args, detached_bound_kwargs, detached_args)
                )
            )
        )
        return (None, None, None, None) + grads


def stacr_checkpoint(
    function: Any, *args, resolved: ResolvedCheckpointSite, preserve_rng_state: bool = True
):
    (base_function, prefix_args, bound_kwargs) = _normalize_checkpoint_callable(function)
    (input_tree, flat_args) = _flatten_checkpoint_values((prefix_args, bound_kwargs, args))
    return _STACRCheckpointFunction.apply(
        base_function, preserve_rng_state, resolved, input_tree, *flat_args
    )


def make_c1_saved_tensor_hooks(
    spec: STACRCheckpointSpec,
    target_tensor: torch.Tensor,
    module: nn.Module,
    codec: CheckpointBoundaryCodec,
):
    matched_once = False

    def pack_hook(tensor: torch.Tensor):
        nonlocal matched_once
        if not isinstance(tensor, torch.Tensor):
            return tensor
        if not _tensor_matches_target(tensor, target_tensor):
            return tensor
        if matched_once:
            codec.note_duplicate_pack(tensor)
            return tensor
        matched_once = True
        from stacr.protocols.baselines.checkpointing import get_global_recomputation_guard

        guard = get_global_recomputation_guard()
        training = bool(module.training and (not guard.is_recomputing))
        return codec.pack_tensor(tensor, training=training)

    def unpack_hook(packed: Any):
        if isinstance(packed, CompressedBoundaryRecord) and packed.site_id == spec.site_id:
            return codec.unpack_record(packed)
        return packed

    return (pack_hook, unpack_hook)


def build_stacr_checkpoint_config(config: Any) -> STACRCheckpointConfig:
    return STACRCheckpointConfig(
        enabled=bool(getattr(config, "stacr_compress_checkpoint", False)),
        target=getattr(config, "stacr_ckpt_target", "decoder_block_input"),
        rank=getattr(config, "stacr_ckpt_rank", None),
        warmup_steps=int(getattr(config, "stacr_ckpt_warmup_steps", 500)),
        residual_mode=getattr(config, "stacr_ckpt_residual_mode", "none"),
        residual_budget=float(getattr(config, "stacr_ckpt_residual_budget", 0.005)),
        refresh_trigger=getattr(config, "stacr_ckpt_refresh_trigger", "residual_energy"),
        refresh_period=int(getattr(config, "stacr_ckpt_refresh_period", 200)),
        refresh_energy_threshold_mode=str(
            getattr(config, "stacr_ckpt_refresh_energy_threshold_mode", "auto")
        ),
        refresh_energy_threshold=float(
            getattr(config, "stacr_ckpt_refresh_energy_threshold", 0.15)
        ),
        min_numel=int(getattr(config, "stacr_ckpt_min_numel", 1 << 20)),
        basis_dtype=getattr(config, "stacr_ckpt_basis_dtype", "auto"),
        quantize_coefficients=bool(getattr(config, "stacr_ckpt_quantize_coefficients", False)),
        quantize_bits=int(getattr(config, "stacr_ckpt_quantize_bits", 8)),
        quantize_residuals=bool(getattr(config, "stacr_ckpt_quantize_residuals", True)),
        residual_quantize_bits=int(getattr(config, "stacr_ckpt_residual_quantize_bits", 4)),
        allow_inner_stacr=bool(getattr(config, "stacr_ckpt_allow_inner_stacr", False)),
        planner=getattr(config, "stacr_ckpt_planner", "heuristic"),
    )


def collect_c1_site_candidates(model: nn.Module, config: Any) -> List[CheckpointSiteCandidate]:
    ckpt_cfg = build_stacr_checkpoint_config(config)
    family = _infer_model_family(model, getattr(config, "model", None))
    if family not in {"gpt2", "llama", "bert"}:
        raise ValueError("C1 v1 supports only {gpt2, llama, bert} model families.")
    hidden_size = _hidden_size_for_model(model)
    if hidden_size is None:
        raise ValueError("Unable to infer hidden size for C1 checkpoint sites.")
    basis_dtype = _resolve_ckpt_basis_dtype(
        ckpt_cfg.basis_dtype,
        getattr(config.baseline_cfg, "amp", "none"),
        next(model.parameters()).device,
    )
    block_names = {"gpt2": {"GPT2Block"}, "llama": {"LlamaDecoderLayer"}, "bert": {"BertLayer"}}[
        family
    ]
    rank = ckpt_cfg.rank if ckpt_cfg.rank is not None else _default_ckpt_rank(family, hidden_size)
    resolved_rank = min(hidden_size, int(rank))
    _validate_llama_lossy_refresh_policy(
        family=family,
        hidden_size=hidden_size,
        rank=resolved_rank,
        refresh_trigger=ckpt_cfg.refresh_trigger,
    )
    candidates: List[CheckpointSiteCandidate] = []
    for name, module in model.named_modules():
        if module.__class__.__name__ not in block_names:
            continue
        spec = STACRCheckpointSpec(
            site_id=name,
            module_type=module.__class__.__name__,
            model_family=family,
            hidden_size=hidden_size,
            rank=resolved_rank,
            residual_mode=ckpt_cfg.residual_mode,
            residual_budget=ckpt_cfg.residual_budget,
            basis_dtype=basis_dtype,
            quantize_coefficients=ckpt_cfg.quantize_coefficients,
            quantize_bits=ckpt_cfg.quantize_bits,
            quantize_residuals=ckpt_cfg.quantize_residuals,
            residual_quantize_bits=ckpt_cfg.residual_quantize_bits,
            warmup_steps=ckpt_cfg.warmup_steps,
            refresh_trigger=ckpt_cfg.refresh_trigger,
            refresh_period=ckpt_cfg.refresh_period,
            refresh_energy_threshold=_resolve_ckpt_refresh_energy_threshold(
                family, ckpt_cfg.refresh_energy_threshold, ckpt_cfg.refresh_energy_threshold_mode
            ),
            target_arg_index=0,
            min_numel=ckpt_cfg.min_numel,
        )
        candidates.append(CheckpointSiteCandidate(name=name, module=module, spec=spec))
    return candidates


def _infer_model_family(model: nn.Module, model_name: Optional[str]) -> Optional[str]:
    model_type = getattr(getattr(model, "config", None), "model_type", None)
    if model_type in {"gpt2", "llama", "bert"}:
        return model_type
    lowered_name = (model_name or "").lower()
    if "gpt2" in lowered_name:
        return "gpt2"
    if "llama" in lowered_name:
        return "llama"
    if "bert" in lowered_name:
        return "bert"
    return None


def _validate_llama_lossy_refresh_policy(
    *, family: str, hidden_size: int, rank: int, refresh_trigger: str
) -> None:
    if family == "llama" and refresh_trigger == "off" and (rank < hidden_size):
        raise ValueError(
            "C1 v1 does not allow lossy Llama checkpoint compression with --stacr-ckpt-refresh-trigger off. Use residual_energy/period/hybrid or set --stacr-ckpt-rank to the full hidden size."
        )


def _hidden_size_for_model(model: nn.Module) -> Optional[int]:
    model_config = getattr(model, "config", None)
    for attr in ("hidden_size", "n_embd"):
        value = getattr(model_config, attr, None)
        if isinstance(value, int) and value > 0:
            return value
    return None


def register_c1_sites(
    model: nn.Module, config: Any, *, runtime: Optional[STACRCheckpointRuntime] = None
) -> STACRCheckpointRuntime:
    runtime = runtime or STACRCheckpointRuntime()
    runtime.registry.clear()
    ckpt_cfg = build_stacr_checkpoint_config(config)
    candidates = collect_c1_site_candidates(model, config)
    runtime.site_count_total = len(candidates)
    runtime.planner_mode = ckpt_cfg.planner
    runtime.planner_summary = None
    runtime.planner_decisions = None
    if not candidates:
        raise ValueError(
            "C1 could not register any checkpoint-capable decoder blocks for this model."
        )
    kept_site_ids = {candidate.spec.site_id for candidate in candidates}
    if ckpt_cfg.planner != "off":
        from stacr.protocols.planner import (
            plan_stacr_checkpoint_sites,
            summarize_checkpoint_site_decisions,
        )

        planner_decisions = plan_stacr_checkpoint_sites(model, config, candidates=candidates)
        runtime.planner_decisions = [decision.to_dict() for decision in planner_decisions]
        runtime.planner_summary = summarize_checkpoint_site_decisions(
            planner_decisions,
            target=ckpt_cfg.target,
            planner_mode=ckpt_cfg.planner,
            planner_enforced=ckpt_cfg.planner == "enforce",
        )
        if ckpt_cfg.planner == "enforce":
            kept_site_ids = {decision.name for decision in planner_decisions if decision.keep}
            if not kept_site_ids:
                rejected: Dict[str, int] = {}
                for decision in planner_decisions:
                    rejected[decision.reason] = int(rejected.get(decision.reason, 0)) + 1
                raise ValueError(
                    "C1 planner enforce rejected every checkpoint site: "
                    + ", ".join(
                        (f"{reason}={count}" for (reason, count) in sorted(rejected.items()))
                    )
                )
    for candidate in candidates:
        if candidate.spec.site_id not in kept_site_ids:
            continue
        runtime.registry.register(candidate.module, candidate.spec)
    return runtime


def enable_c1_checkpointing_for_model(
    model: nn.Module, config: Any, checkpoint_mode: str
) -> STACRCheckpointRuntime:
    ckpt_cfg = build_stacr_checkpoint_config(config)
    if not ckpt_cfg.enabled:
        raise ValueError("C1 runtime requested while --stacr-compress-checkpoint is disabled.")
    if checkpoint_mode != "transformer_hf":
        raise ValueError("Boundary compression supports only --checkpointing transformer_hf.")
    if getattr(config, "use_tensor_parallel", False):
        raise ValueError("Boundary compression does not support tensor parallel.")
    if getattr(config, "use_data_parallel", False):
        raise ValueError("Boundary compression does not support --data-parallel.")
    if isinstance(model, nn.DataParallel):
        raise ValueError("Boundary compression does not support nn.DataParallel-wrapped models.")
    if getattr(config, "method", None) == "prac":
        raise ValueError("C1 is STAC-R-specific and incompatible with --method prac.")
    if getattr(config, "stacr_rank_mode", "fixed") != "fixed":
        raise ValueError("Boundary compression requires --stacr-rank-mode fixed.")
    if ckpt_cfg.residual_mode != "none":
        raise ValueError("Boundary compression requires --stacr-ckpt-residual-mode none.")
    if ckpt_cfg.quantize_coefficients and int(ckpt_cfg.quantize_bits) != 8:
        raise ValueError("Boundary coefficient quantization requires --stacr-ckpt-quantize-bits 8.")
    if ckpt_cfg.target not in {"decoder_block_input", "first_tensor"}:
        raise ValueError(
            "Boundary compression supports --stacr-ckpt-target decoder_block_input|first_tensor."
        )
    if ckpt_cfg.planner not in {"off", "heuristic", "enforce"}:
        raise ValueError("C1 planner mode must be one of {off, heuristic, enforce}.")
    runtime = register_c1_sites(model, config)
    runtime.activate()
    return runtime
