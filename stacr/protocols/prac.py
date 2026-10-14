"""PRAC-linear-only: random-projection activation compression on the linear surface.

Saved-tensor compressor that stores ``x_proj = x @ P`` for backward and
reconstructs ``x`` approximately as ``x_proj @ P.T`` during gradient
computation. ``P`` is a random Gaussian projection, resampled on a
global-step window.

This module implements the **linear-only** surface of PRAC for matched
comparison with STAC-R (see ``NeurIPS_Paper/t26_phase2_runbook.md``). It is
NOT a full PRAC paper replication because it deliberately does not cover
PRAC's non-linear surface (LayerNorm, GeLU output).

Layout convention:
- Internal storage is always ``nn.Linear`` (weight shape ``(out, in)``),
  matching STAC-R. HuggingFace Conv1D support (commit 2) works by
  transposing the weight during replacement so the wrapped module always
  carries weights in the Linear layout.

Backward math (Linear layout, unbiased estimator):
    grad_weight_est = grad_output.T @ x_proj @ P.T        shape (out, in)
    grad_input      = grad_output @ W                     (..., in), exact
    grad_bias       = grad_output.sum over sample dims    (out,), exact

Unbiasedness: entries of ``P`` are iid ``N(0, 1/rank)`` so
``E[P @ P.T] = I_in`` and ``E[x_proj @ P.T] = x``.

Refresh schedule: resample ``P`` is keyed to the optimizer/global training
step, NOT to forward calls. Use ``set_prac_global_step(model, step)``
once per training step before forward.

Stable seed helper: per-layer per-window seeds derive from
``hashlib.sha256`` so they are reproducible across processes and
platforms. DO NOT delegate to Python's builtin ``hash()`` — it is
PYTHONHASHSEED-randomized.
"""

from __future__ import annotations
import hashlib
import math
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class PRACConfig:
    """Configuration for PRAC-linear-only compression.

    Attributes:
        rank: Projection dimension ``r``. Entries of ``P`` drawn iid from
            ``N(0, 1/r)``.
        update_proj_gap: Resample ``P`` every N global training steps.
        proj_dtype: Dtype spec for the persistent projection buffer. ``None``
            defers to runtime-level AMP-aware resolution via
            ``experiments.convergence_training._resolve_runtime_prac_proj_dtype``.
            For the controlled T26 Phase 2 cohort this is pinned to
            ``"float16"``.
        seed_base: Base seed for derived per-layer projection seeds.
            Combined with layer name and window index via SHA-256.
        quiet: Suppress replacement/debug prints.
    """

    rank: int = 8
    update_proj_gap: int = 50
    proj_dtype: Optional[str] = None
    seed_base: int = 42
    quiet: bool = False


def _resolve_proj_dtype(dtype_spec: Optional[str]) -> torch.dtype:
    """Map a string dtype spec to a ``torch.dtype``.

    ``None`` and ``"auto"`` resolve to ``float32`` here; runtime-aware
    auto resolution happens upstream in
    ``_resolve_runtime_prac_proj_dtype`` in ``convergence_training.py``.
    """
    if dtype_spec is None:
        return torch.float32
    s = str(dtype_spec).lower()
    if s in ("auto", "float32", "fp32", "float"):
        return torch.float32
    if s in ("float16", "fp16", "half"):
        return torch.float16
    if s in ("bfloat16", "bf16"):
        return torch.bfloat16
    raise ValueError(f"Unknown PRAC proj_dtype: {dtype_spec!r}")


def _layer_seed(seed_base: int, layer_name: str, window_idx: int) -> int:
    """Derive a deterministic 32-bit seed from ``(seed_base, layer_name, window_idx)``.

    Uses ``hashlib.sha256`` so the mapping is stable across processes and
    platforms. DO NOT use Python's builtin ``hash()`` — it is
    ``PYTHONHASHSEED``-randomized per process by default.

    Returns an integer in ``[0, 2**32 - 1]`` safe for
    ``torch.Generator().manual_seed(seed)``.
    """
    payload = f"{int(seed_base)}:{layer_name}:{int(window_idx)}".encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:4], byteorder="big", signed=False)


def _is_hf_conv1d(module: nn.Module) -> bool:
    """Detect HuggingFace Conv1D (GPT-2 MLP) without importing transformers.

    Conv1D stores weight as ``(in_features, out_features)``, transposed vs
    ``nn.Linear``. ``replace_linear_with_prac`` transposes during weight copy
    so the internal ``nn.Linear`` always holds Linear-layout weights.
    """
    return module.__class__.__name__ == "Conv1D"


class PRACFunction(torch.autograd.Function):
    """Autograd function for PRAC-linear-only.

    Forward computes the standard Linear output but saves only
    ``x_proj = x @ P`` instead of ``x`` itself. Backward reconstructs ``x``
    approximately as ``x_proj @ P.T`` and produces an unbiased estimate of
    ``grad_weight`` in Linear layout ``(out, in)``.

    Call signature::

        output = PRACFunction.apply(input_tensor, weight, bias, projection)

    Arguments to forward():
        input_tensor: ``(..., in_features)``, arbitrary leading shape.
        weight:       ``(out_features, in_features)``.
        bias:         ``(out_features,)`` or ``None``.
        projection:   ``(in_features, rank)``, persistent P matrix.
    """

    @staticmethod
    def forward(ctx, input_tensor, weight, bias, projection):
        in_features = input_tensor.shape[-1]
        x_flat = input_tensor.reshape(-1, in_features)
        P_act = (
            projection if projection.dtype == x_flat.dtype else projection.to(dtype=x_flat.dtype)
        )
        x_proj = x_flat @ P_act
        output = F.linear(input_tensor, weight, bias)
        ctx.save_for_backward(x_proj, P_act, weight)
        ctx.input_shape = tuple(input_tensor.shape)
        ctx.has_bias = bias is not None
        return output

    @staticmethod
    def backward(ctx, grad_output):
        (x_proj, projection, weight) = ctx.saved_tensors
        input_shape = ctx.input_shape
        out_features = weight.shape[0]
        go_flat = grad_output.reshape(-1, out_features)
        go_fp32 = go_flat.to(torch.float32)
        xp_fp32 = x_proj.to(torch.float32)
        P_fp32 = projection.to(torch.float32)
        gW = go_fp32.t() @ xp_fp32
        gW = gW @ P_fp32.t()
        grad_weight = gW.to(dtype=weight.dtype)
        W_act = weight.to(dtype=go_flat.dtype)
        gi_flat = go_flat @ W_act
        grad_input = gi_flat.reshape(input_shape)
        grad_bias = go_flat.sum(dim=0) if ctx.has_bias else None
        return (grad_input, grad_weight, grad_bias, None)


class PRACLinear(nn.Module):
    """Linear-layout PRAC-linear-only module.

    Internal storage is always a plain ``nn.Linear`` with weight layout
    ``(out_features, in_features)``, regardless of whether the replaced
    module was ``nn.Linear`` or HuggingFace Conv1D. See
    ``replace_linear_with_prac`` for Conv1D handling via weight transpose.

    API used by the training loop:
        ``set_global_step(step)``: advance the step counter. When the
            window index changes, resample the projection buffer.
        ``forward(x)``: returns the (approximate) output of a plain
            Linear, saving ``x_proj = x @ P`` for backward instead of ``x``.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        config: PRACConfig,
        bias: bool = True,
        layer_name: str = "<unnamed>",
    ):
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.config = config
        self.layer_name = str(layer_name)
        self.linear = nn.Linear(self.in_features, self.out_features, bias=bias)
        self.effective_rank = max(1, min(int(config.rank), self.in_features))
        if self.effective_rank != int(config.rank) and (not config.quiet):
            warnings.warn(
                f"PRACLinear('{self.layer_name}'): clamped rank {config.rank} -> {self.effective_rank} (in_features={self.in_features})."
            )
        self.proj_dtype = _resolve_proj_dtype(config.proj_dtype)
        P0 = self._sample_projection(step=0, device=self.linear.weight.device)
        self.register_buffer("projection", P0)
        self.register_buffer("step_counter", torch.zeros(1, dtype=torch.long))
        self._last_window_idx = 0

    def _sample_projection(
        self, step: int, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None
    ) -> torch.Tensor:
        """Sample P on CPU (deterministic across hardware) then move to target.

        Entries iid ``N(0, 1/rank)`` so ``E[P @ P.T] = I`` and the backward
        estimator is unbiased with variance decreasing in rank.
        """
        gap = max(1, int(self.config.update_proj_gap))
        window_idx = int(step) // gap
        seed = _layer_seed(self.config.seed_base, self.layer_name, window_idx)
        g = torch.Generator(device="cpu")
        g.manual_seed(seed)
        std = 1.0 / math.sqrt(self.effective_rank)
        P = (
            torch.randn(self.in_features, self.effective_rank, generator=g, dtype=torch.float32)
            * std
        )
        target_device = device if device is not None else self.linear.weight.device
        target_dtype = dtype if dtype is not None else self.proj_dtype
        return P.to(device=target_device, dtype=target_dtype)

    def set_global_step(self, step: int) -> None:
        """Advance the step counter. Resample P on window boundary crossings.

        Call this once per training step BEFORE the forward pass. Do not
        call during eval-only forwards; eval should not resample.
        """
        step_int = int(step)
        self.step_counter.fill_(step_int)
        gap = max(1, int(self.config.update_proj_gap))
        new_window = step_int // gap
        if new_window != self._last_window_idx:
            self._last_window_idx = new_window
            new_P = self._sample_projection(
                step=step_int, device=self.projection.device, dtype=self.projection.dtype
            )
            self.projection.copy_(new_P)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if getattr(self, "_disable_prac_compression", False):
            return self.linear(x)
        self._last_saved_projection_dtype = (
            self.projection.dtype if x.dtype == self.projection.dtype else x.dtype
        )
        return PRACFunction.apply(x, self.linear.weight, self.linear.bias, self.projection)

    def projection_buffer_bytes(self) -> int:
        """Bytes occupied by the persistent projection buffer ``P``."""
        return int(self.projection.numel() * self.projection.element_size())

    def saved_activation_bytes(self, batch_tokens: int, act_dtype: torch.dtype) -> int:
        """Estimated bytes of saved ``x_proj`` per forward.

        ``batch_tokens`` is ``B_flat``. For a ``(B, L, N)`` input,
        ``B_flat = B * L``.
        """
        elem_size = torch.tensor([], dtype=act_dtype).element_size()
        return int(batch_tokens) * int(self.effective_rank) * int(elem_size)


def set_prac_global_step(model: nn.Module, step: int) -> None:
    """Set the current global training step on every PRAC module in the model.

    Call once per training-loop step BEFORE forward. Do not call on
    eval-only forwards.
    """
    for module in model.modules():
        if isinstance(module, PRACLinear):
            module.set_global_step(step)


def replace_linear_with_prac(
    model: nn.Module, config: PRACConfig, target_layers: Optional[List[str]] = None
) -> int:
    """Replace ``nn.Linear`` AND HF Conv1D modules with ``PRACLinear`` in-place.

    The wrapped module always carries weights in Linear layout
    ``(out_features, in_features)``. For HF Conv1D (weight stored as
    ``(in_features, out_features)``), the weight is transposed during copy
    so gradients produced by ``PRACLinear`` are directly compatible with
    Linear-layout optimizer updates — matching the STAC-R convention.

    If ``target_layers`` is ``None``, every replaceable module is wrapped.
    If a list is given, only matching named modules are wrapped.

    Returns the number of layers replaced.
    """
    module_dict = dict(model.named_modules())
    replaced = 0
    for name, module in list(model.named_modules()):
        is_linear = isinstance(module, nn.Linear)
        is_conv1d = _is_hf_conv1d(module)
        if not (is_linear or is_conv1d):
            continue
        if target_layers is not None and name not in target_layers:
            continue
        if is_linear:
            in_features = module.in_features
            out_features = module.out_features
            has_bias = module.bias is not None
        else:
            (in_features, out_features) = module.weight.shape
            has_bias = getattr(module, "bias", None) is not None
        new_module = PRACLinear(in_features, out_features, config, bias=has_bias, layer_name=name)
        target_device = module.weight.device
        target_dtype = module.weight.dtype
        if is_conv1d:
            new_module.linear.weight.data = (
                module.weight.data.t().clone().to(device=target_device, dtype=target_dtype)
            )
        else:
            new_module.linear.weight.data = module.weight.data.clone().to(
                device=target_device, dtype=target_dtype
            )
        if has_bias and new_module.linear.bias is not None:
            new_module.linear.bias.data = module.bias.data.clone().to(
                device=new_module.linear.bias.device, dtype=new_module.linear.bias.dtype
            )
        new_module.projection.data = new_module._sample_projection(step=0, device=target_device)
        parts = name.rsplit(".", 1)
        if len(parts) == 1:
            parent = model
            attr = parts[0]
        else:
            parent = module_dict[parts[0]]
            attr = parts[1]
        setattr(parent, attr, new_module)
        replaced += 1
        if not config.quiet:
            kind = "Conv1D" if is_conv1d else "Linear"
            print(f"  Replaced {name} ({kind}): {in_features}x{out_features} with PRACLinear")
    if not config.quiet:
        print(
            f"Replaced {replaced} layers with PRACLinear (rank={config.rank}, update_proj_gap={config.update_proj_gap})"
        )
    return replaced


def prac_stored_bytes_per_layer(
    batch_tokens: int,
    in_features: int,
    rank: int,
    proj_dtype: torch.dtype,
    act_dtype: Optional[torch.dtype] = None,
    saved_projection_dtype: Optional[torch.dtype] = None,
) -> Dict[str, int]:
    """Per-layer PRAC storage estimate, in bytes.

    Components:
        - ``saved_activation_bytes``: ``batch_tokens * rank * sizeof(act_dtype)``
        - ``projection_buffer_bytes``: ``in_features * rank * sizeof(proj_dtype)``
        - ``saved_projection_bytes``: extra saved casted projection tensor when
          ``act_dtype != proj_dtype`` and forward must materialize ``P_act``

    If ``act_dtype`` is ``None``, defaults to ``proj_dtype``. If
    ``saved_projection_dtype`` is ``None``, fall back to the previous heuristic:
    use ``act_dtype`` when ``act_dtype != proj_dtype`` and assume no extra saved
    cast tensor otherwise.
    """
    if act_dtype is None:
        act_dtype = proj_dtype
    if saved_projection_dtype is None:
        saved_projection_dtype = act_dtype if act_dtype != proj_dtype else proj_dtype
    act_elem = torch.tensor([], dtype=act_dtype).element_size()
    proj_elem = torch.tensor([], dtype=proj_dtype).element_size()
    saved_activation = int(batch_tokens) * int(rank) * int(act_elem)
    projection_buf = int(in_features) * int(rank) * int(proj_elem)
    saved_projection = 0
    if saved_projection_dtype != proj_dtype:
        saved_proj_elem = torch.tensor([], dtype=saved_projection_dtype).element_size()
        saved_projection = int(in_features) * int(rank) * int(saved_proj_elem)
    return {
        "saved_activation_bytes": saved_activation,
        "projection_buffer_bytes": projection_buf,
        "saved_projection_bytes": saved_projection,
        "stored_bytes": saved_activation + projection_buf + saved_projection,
    }
