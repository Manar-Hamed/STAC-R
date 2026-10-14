from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from .oja import orthonormalize
from .quant import QuantizedValues, quantize_values

RESIDUAL_MODES = {"chunk_topk", "topk", "global_topk", "uniform", "none", "block_sparse", "sampled"}


def _select_sparse_residual(x2d, coeff, basis, budget, chunk_rows):
    """Chunk-local magnitude selection, followed by a global budget trim.

    Each chunk contributes its budget fraction. This differs from exact global
    top-k; use residual_mode='global_topk' for that rule. Workspace is bounded
    by chunk_rows * features plus the selected entries.
    """
    rows, feats = x2d.shape
    total = rows * feats
    k_total = int(round(float(budget) * total))
    if k_total <= 0:
        return None, None
    idx_parts, val_parts = [], []
    chunk_rows = max(1, int(chunk_rows))
    for start in range(0, rows, chunk_rows):
        stop = min(start + chunk_rows, rows)
        flat = (x2d[start:stop] - coeff[start:stop].mm(basis.t())).reshape(-1)
        k = max(1, min(flat.numel(), int(round(float(budget) * flat.numel()))))
        local = torch.topk(flat.abs(), k=k, sorted=False).indices
        idx_parts.append(local + start * feats)
        val_parts.append(flat[local])
    idx, val = torch.cat(idx_parts), torch.cat(val_parts)
    if idx.numel() > k_total:
        keep = torch.topk(val.abs(), k=k_total, sorted=False).indices
        idx, val = idx[keep], val[keep]
    return idx.to(torch.int32 if total < 2**31 - 1 else torch.int64), val


def _select_residual(x, coeff, basis, budget, chunk_rows, mode, block_size, sample_ratio):
    if mode == "none" or budget <= 0 or x.numel() == 0:
        return None, None
    if mode in ("chunk_topk", "topk"):
        return _select_sparse_residual(x, coeff, basis, budget, chunk_rows)
    rows, features = x.shape
    total = x.numel()
    k_total = int(round(budget * total))
    if k_total == 0:
        return None, None
    index_dtype = torch.int32 if total < 2**31 - 1 else torch.int64
    if mode == "sampled":
        # Threshold ablation: seeded sampling, followed by first-passing entries.
        # Its dense selection workspace is part of this algorithm; backward is
        # still factor-native. The default chunk_topk path has bounded workspace.
        residual = (x - coeff.mm(basis.t())).reshape(-1)
        magnitude = residual.abs()
        generator = torch.Generator(device=x.device).manual_seed(0)
        sample_n = max(1024, int(total * sample_ratio))
        sample = magnitude[torch.randint(total, (sample_n,), device=x.device, generator=generator)]
        threshold = torch.quantile(sample.float(), 1.0 - budget)
        indices = (magnitude >= threshold).nonzero(as_tuple=True)[0][:k_total]
        return indices.to(index_dtype), residual[indices]
    if mode == "block_sparse" and features % block_size:
        return _select_sparse_residual(x, coeff, basis, budget, chunk_rows)
    indices, values = [], []
    retained_i = retained_v = None
    for start in range(0, rows, chunk_rows):
        stop = min(start + chunk_rows, rows)
        flat = (x[start:stop] - coeff[start:stop].mm(basis.t())).reshape(-1)
        if mode == "global_topk":
            local = torch.topk(flat.abs(), min(k_total, flat.numel()), sorted=False).indices
            ci, cv = local + start * features, flat[local]
            if retained_i is not None:
                ci, cv = torch.cat((retained_i, ci)), torch.cat((retained_v, cv))
            keep = torch.topk(cv.abs(), min(k_total, cv.numel()), sorted=False).indices
            retained_i, retained_v = ci[keep], cv[keep]
        elif mode == "block_sparse":
            grouped = flat.reshape(-1, block_size)
            count = max(1, int(round(budget * grouped.shape[0])))
            blocks = torch.topk(grouped.square().sum(1), count, sorted=False).indices
            offsets = torch.arange(block_size, device=x.device)
            local = (blocks[:, None] * block_size + offsets).reshape(-1)
            indices.append(local + start * features)
            values.append(flat[local])
        elif mode == "uniform":
            # Deterministic uniformly spaced entries are a sparse control.
            first = math.ceil(start * features * k_total / total)
            last = math.ceil(stop * features * k_total / total)
            absolute = torch.div(
                torch.arange(first, last, device=x.device) * total, k_total, rounding_mode="floor"
            )
            indices.append(absolute)
            values.append(flat[absolute - start * features])
        else:
            raise ValueError(f"Unknown residual_mode: {mode!r}")
    if mode == "global_topk":
        return retained_i.to(index_dtype), retained_v
    return torch.cat(indices).to(index_dtype), torch.cat(values)


class STACRFunction(torch.autograd.Function):
    """Exact linear forward/input gradient and factor-native weight gradient.

    grad_W = (G.T @ C) @ Q.T + G.T @ R_sparse. The saved basis is an immutable
    version: a later forward can replace it, but cannot mutate it.
    """

    @staticmethod
    def forward(
        ctx,
        x,
        weight,
        bias,
        basis,
        budget,
        residual_bits,
        chunk_rows,
        transposed,
        residual_mode="chunk_topk",
        residual_block_size=16,
        residual_sample_ratio=0.1,
    ):
        in_features = basis.shape[0]
        x2d = x.reshape(-1, in_features)
        if transposed:
            out2d = x2d.mm(weight) if bias is None else torch.addmm(bias, x2d, weight)
        else:
            out2d = F.linear(x2d, weight, bias)
        # Match the exact linear operation's dtype under autocast, then disable
        # ambient autocast for the explicitly typed compression arithmetic.
        with torch.autocast(device_type=x.device.type, enabled=False):
            activation, q = x2d.to(out2d.dtype), basis.to(out2d.dtype)
            coeff = activation.mm(q)
            idx, val = _select_residual(
                activation,
                coeff,
                q,
                float(budget),
                int(chunk_rows),
                residual_mode,
                int(residual_block_size),
                float(residual_sample_ratio),
            )
            packed = quantize_values(val, int(residual_bits)) if val is not None else None
        saved = [coeff, weight, q]
        if idx is not None:
            saved.extend([idx, packed.data, packed.scale])
        ctx.save_for_backward(*saved)
        ctx.has_residual = idx is not None
        ctx.residual_bits = packed.bits if packed is not None else 0
        ctx.residual_len = packed.length if packed is not None else 0
        ctx.leading, ctx.in_features = x.shape[:-1], in_features
        ctx.out_features, ctx.transposed = out2d.shape[1], bool(transposed)
        ctx.input_dtype, ctx.bias_dtype = x.dtype, None if bias is None else bias.dtype
        ctx.arg_count = len(ctx.needs_input_grad)
        return out2d.reshape(*ctx.leading, ctx.out_features)

    @staticmethod
    def backward(ctx, grad_out):
        coeff, weight, q, *residual = ctx.saved_tensors
        g2d = grad_out.reshape(-1, ctx.out_features)
        grad_input = grad_weight = grad_bias = None
        with torch.autocast(device_type=g2d.device.type, enabled=False):
            if ctx.needs_input_grad[0]:
                w = weight.to(g2d.dtype)
                grad_input = g2d.mm(w.t() if ctx.transposed else w)
                grad_input = grad_input.reshape(*ctx.leading, ctx.in_features).to(ctx.input_dtype)
            if ctx.needs_input_grad[1]:
                grad_weight = g2d.t().mm(coeff.to(g2d.dtype)).mm(q.to(g2d.dtype).t())
                if ctx.has_residual and residual[0].numel():
                    idx, qdata, qscale = residual
                    # Sparse mm avoids a dense rows-by-features reconstruction
                    # and a selected-entries-by-output-features gather workspace.
                    sparse_dtype = torch.float64 if g2d.dtype == torch.float64 else torch.float32
                    values = QuantizedValues(
                        qdata, qscale, ctx.residual_bits, ctx.residual_len
                    ).dequantize(sparse_dtype)
                    idx64 = idx.long()
                    coordinates = torch.stack(
                        (
                            torch.div(idx64, ctx.in_features, rounding_mode="floor"),
                            idx64 % ctx.in_features,
                        )
                    )
                    sparse = torch.sparse_coo_tensor(
                        coordinates, values, size=(g2d.shape[0], ctx.in_features), device=g2d.device
                    ).coalesce()
                    correction = torch.sparse.mm(sparse.t(), g2d.to(sparse_dtype)).t()
                    grad_weight = grad_weight + correction.to(grad_weight.dtype)
                if ctx.transposed:
                    grad_weight = grad_weight.t()
                grad_weight = grad_weight.to(weight.dtype)
            if ctx.needs_input_grad[2]:
                grad_bias = g2d.sum(0).to(ctx.bias_dtype)
        return (grad_input, grad_weight, grad_bias) + (None,) * (ctx.arg_count - 3)


class STACRLinear(nn.Module):
    """Linear projection with compressed saved inputs and unchanged parameters.

    Set the optimizer step with stacr.set_step before each training step.
    Repeated calls in that step share a basis, including checkpoint replay.
    Evaluation, no-grad calls and frozen weights use the exact native operation.
    Turbo freezes the basis and drops residuals; the weights still train.
    """

    def __init__(
        self,
        weight: nn.Parameter,
        bias: nn.Parameter | None,
        *,
        rank: int = 8,
        residual_budget: float = 0.005,
        residual_bits: int = 8,
        oja_lr: float = 0.01,
        update_interval: int = 1,
        warmup_steps: int = 0,
        freeze_after_warmup: bool = False,
        turbo_after_step: int | None = None,
        chunk_rows: int = 512,
        basis_dtype: str = "float32",
        transposed: bool = False,
        residual_mode: str = "chunk_topk",
        residual_block_size: int = 16,
        residual_sample_ratio: float = 0.1,
    ):
        super().__init__()
        if (
            not isinstance(weight, nn.Parameter)
            or weight.ndim != 2
            or not weight.is_floating_point()
        ):
            raise ValueError("weight must be a floating-point 2-D Parameter")
        for name, value in (
            ("rank", rank),
            ("update_interval", update_interval),
            ("chunk_rows", chunk_rows),
            ("residual_block_size", residual_block_size),
        ):
            if isinstance(value, bool) or int(value) != value or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not math.isfinite(residual_budget) or not 0 <= residual_budget <= 1:
            raise ValueError("residual_budget must be between 0 and 1")
        if residual_bits not in (0, 4, 8):
            raise ValueError("residual_bits must be 0, 4, or 8")
        if not math.isfinite(oja_lr) or oja_lr < 0:
            raise ValueError("oja_lr must be finite and nonnegative")
        for name, value in (("warmup_steps", warmup_steps), ("turbo_after_step", turbo_after_step)):
            if value is not None and (isinstance(value, bool) or int(value) != value or value < 0):
                raise ValueError(f"{name} must be a nonnegative integer")
        if residual_mode not in RESIDUAL_MODES:
            raise ValueError(f"residual_mode must be one of {sorted(RESIDUAL_MODES)}")
        if not math.isfinite(residual_sample_ratio) or not 0 < residual_sample_ratio <= 1:
            raise ValueError("residual_sample_ratio must be in (0, 1]")
        dtypes = {
            "float32": torch.float32,
            "fp32": torch.float32,
            "float16": torch.float16,
            "fp16": torch.float16,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "auto": weight.dtype,
            "activation": weight.dtype,
        }
        if basis_dtype not in dtypes:
            raise ValueError(f"Unknown basis_dtype: {basis_dtype!r}")
        self.weight, self.bias = weight, bias
        self.transposed = bool(transposed)
        self.in_features = int(weight.shape[0] if transposed else weight.shape[1])
        self.out_features = int(weight.shape[1] if transposed else weight.shape[0])
        if bias is not None and (bias.ndim != 1 or bias.numel() != self.out_features):
            raise ValueError("bias shape must match the output feature count")
        self.rank = min(int(rank), self.in_features)
        self.residual_budget, self.residual_bits = float(residual_budget), int(residual_bits)
        self.oja_lr, self.update_interval = float(oja_lr), int(update_interval)
        self.warmup_steps, self.freeze_after_warmup = int(warmup_steps), bool(freeze_after_warmup)
        self.turbo_after_step, self.chunk_rows = turbo_after_step, int(chunk_rows)
        self.residual_mode, self.residual_block_size = residual_mode, int(residual_block_size)
        self.residual_sample_ratio = float(residual_sample_ratio)
        self.register_buffer(
            "basis",
            torch.zeros(
                self.in_features, self.rank, dtype=dtypes[basis_dtype], device=weight.device
            ),
        )
        self.register_buffer("basis_ready", torch.zeros((), dtype=torch.bool, device=weight.device))
        self.register_buffer("_step", torch.zeros((), dtype=torch.int64, device=weight.device))
        self.register_buffer(
            "_last_update_step", torch.full((), -1, dtype=torch.int64, device=weight.device)
        )

    @classmethod
    def from_linear(cls, module: nn.Module, **options) -> "STACRLinear":
        """Wrap a standalone nn.Linear or Transformers Conv1D module."""
        transposed = type(module).__name__ == "Conv1D"
        if not isinstance(module, nn.Linear) and not transposed:
            raise TypeError("from_linear requires nn.Linear or Transformers Conv1D")
        result = cls(module.weight, getattr(module, "bias", None), transposed=transposed, **options)
        result.train(module.training)
        return result

    @property
    def step(self) -> int:
        return int(self._step.item())

    @step.setter
    def step(self, value: int) -> None:
        self.set_step(value)

    def set_step(self, step: int) -> None:
        if isinstance(step, bool) or int(step) != step or step < 0:
            raise ValueError("step must be a nonnegative integer")
        self._step.fill_(int(step))

    def _turbo(self) -> bool:
        return self.turbo_after_step is not None and self.step >= int(self.turbo_after_step)

    @torch.no_grad()
    def _refresh_basis(self, x2d: torch.Tensor) -> None:
        if not x2d.numel():
            return
        step = self.step
        if bool(self.basis_ready) and int(self._last_update_step.item()) == step:
            return
        target = self.basis.dtype
        with torch.autocast(device_type=x2d.device.type, enabled=False):
            if not bool(self.basis_ready):
                sample = x2d.detach().float()
                if sample.shape[0] >= self.rank:
                    _, _, vh = torch.linalg.svd(sample, full_matrices=False)
                    init = vh[: self.rank].t().contiguous()
                else:
                    generator = torch.Generator(device="cpu").manual_seed(0)
                    init = torch.randn(self.in_features, self.rank, generator=generator).to(
                        sample.device
                    )
                # Rebind, never mutate a tensor an outstanding graph saved.
                self.basis = orthonormalize(init).to(target)
                self.basis_ready.fill_(True)
                self._last_update_step.fill_(step)
                return
            if self.freeze_after_warmup and step >= self.warmup_steps:
                return
            if self._turbo() or step % self.update_interval:
                return
            x, qb = x2d.detach().float(), self.basis.float()
            y = x.mm(qb)
            update = x.t().mm(y) / x.shape[0] - qb.mm(y.t().mm(y) / x.shape[0])
            self.basis = orthonormalize(qb + self.oja_lr * update).to(target)
            self._last_update_step.fill_(step)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim < 1 or x.shape[-1] != self.in_features:
            raise ValueError(
                f"Expected last input dimension {self.in_features}, got {tuple(x.shape)}"
            )
        if (
            not self.training
            or not torch.is_grad_enabled()
            or not self.weight.requires_grad
            or x.numel() == 0
        ):
            if self.transposed:
                flat = x.reshape(-1, self.in_features)
                out = (
                    flat.mm(self.weight)
                    if self.bias is None
                    else torch.addmm(self.bias, flat, self.weight)
                )
                return out.reshape(*x.shape[:-1], self.out_features)
            return F.linear(x, self.weight, self.bias)
        from .adaptive_checkpointing import get_global_recomputation_guard

        guard = get_global_recomputation_guard()
        replay = guard.replay_layer(self) if guard.is_recomputing else None
        if replay is not None:
            basis, budget = replay
        else:
            if not guard.is_recomputing or not bool(self.basis_ready):
                self._refresh_basis(x.reshape(-1, self.in_features))
            basis = self.basis
            budget = 0.0 if self._turbo() else self.residual_budget
            guard.record_layer(self, (basis, budget))
        return STACRFunction.apply(
            x,
            self.weight,
            self.bias,
            basis,
            budget,
            self.residual_bits,
            self.chunk_rows,
            self.transposed,
            self.residual_mode,
            self.residual_block_size,
            self.residual_sample_ratio,
        )

    def stored_bytes(self, rows: int, value_bytes: int = 2) -> dict[str, Any]:
        """Factor capacity for one forward, excluding weights and workspaces.

        Chunk and block quotas include rounding; sampled selection reports an
        upper bound because its retained count depends on the activation.
        ``basis`` includes persistent storage and a saved cast when value and
        basis widths differ. Equal-width casts (fp16 versus bf16) and additional
        basis versions held by other outstanding forwards are not included.
        """
        if isinstance(rows, bool) or int(rows) != rows or rows < 0 or value_bytes not in (2, 4, 8):
            raise ValueError(
                "rows must be a nonnegative integer and value_bytes must be 2, 4, or 8"
            )
        rows = int(rows)
        coeff = rows * self.rank * value_bytes
        budget = 0.0 if self._turbo() or self.residual_mode == "none" else self.residual_budget
        count = int(round(budget * rows * self.in_features))
        if (
            self.residual_mode == "block_sparse"
            and self.in_features % self.residual_block_size == 0
            and count
        ):
            count = sum(
                max(
                    1,
                    int(
                        round(
                            budget
                            * min(self.chunk_rows, rows - start)
                            * self.in_features
                            / self.residual_block_size
                        )
                    ),
                )
                * self.residual_block_size
                for start in range(0, rows, self.chunk_rows)
            )
        elif (self.residual_mode in ("chunk_topk", "topk", "block_sparse")) and count:
            count = min(
                count,
                sum(
                    max(
                        1,
                        int(round(budget * min(self.chunk_rows, rows - start) * self.in_features)),
                    )
                    for start in range(0, rows, self.chunk_rows)
                ),
            )
        index_bytes = 4 if rows * self.in_features < 2**31 - 1 else 8
        values = (
            (count + 1) // 2
            if self.residual_bits == 4
            else count * (1 if self.residual_bits == 8 else value_bytes)
        )
        residual = count * index_bytes + values + (4 if count else 0)
        persistent_basis = self.basis.numel() * self.basis.element_size()
        basis_cast = (
            self.basis.numel() * value_bytes
            if self.basis.element_size() != value_bytes and rows
            else 0
        )
        basis = persistent_basis + basis_cast
        return {
            "coeff": coeff,
            "residual": residual,
            "basis": basis,
            "total": coeff + residual + basis,
            "persistent_basis": persistent_basis,
            "saved_basis_cast": basis_cast,
            "residual_count": count,
            "residual_count_is_upper_bound": self.residual_mode == "sampled",
        }
