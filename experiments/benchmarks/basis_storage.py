"""Allocation-only controls for duplicated bases and residual storage.

These altered contractions isolate allocation costs. Their losses and gradients
are not quality measurements of STAC-R.
"""

from __future__ import annotations
import torch
import sys
from pathlib import Path
from .common import RESULTS, emit, run_config
from stacr.layers import STACRLinear, STACRFunction
from stacr.quant import quantize_values
import stacr.layers as _layers

MODELS = [("qwen25", "Qwen/Qwen2.5-1.5B"), ("llama32", "meta-llama/Llama-3.2-1B")]
(BS, SEQ, STEPS) = (2, 1024, 24)


class FatFunction(torch.autograd.Function):
    basis_per_call = True
    fat_residual = True

    @staticmethod
    def forward(ctx, x, weight, bias, basis, budget, residual_bits, chunk_rows, transposed):
        in_features = basis.shape[0]
        leading = x.shape[:-1]
        x2d = x.reshape(-1, in_features)
        w = weight if weight.dtype == x2d.dtype else weight.to(x2d.dtype)
        w_eff = w if transposed else w.t()
        out2d = x2d.mm(w_eff)
        if bias is not None:
            out2d = out2d + bias.to(out2d.dtype).unsqueeze(0)
        q = basis if basis.dtype == x2d.dtype else basis.to(x2d.dtype)
        coeff = x2d.mm(q)
        (idx, val) = (None, None)
        if float(budget) > 0.0:
            (idx, val) = _layers._select_sparse_residual(x2d, coeff, q, float(budget), chunk_rows)
        q_store = q.clone() if FatFunction.basis_per_call else q
        saved = [coeff, weight, q_store]
        if idx is not None:
            if FatFunction.fat_residual:
                saved.extend([idx.long(), val.float(), torch.ones((), device=val.device)])
            else:
                packed = quantize_values(val, int(residual_bits))
                saved.extend([idx, packed.data, packed.scale])
        ctx.save_for_backward(*saved)
        ctx.has_residual = idx is not None
        ctx.fat = FatFunction.fat_residual
        ctx.residual_bits = int(residual_bits)
        ctx.residual_len = int(val.numel()) if val is not None else 0
        (ctx.leading, ctx.in_features) = (leading, in_features)
        (ctx.out_features, ctx.transposed) = (out2d.shape[1], bool(transposed))
        ctx.needs_bias_grad = bias is not None and bias.requires_grad
        return out2d.reshape(*leading, out2d.shape[1])

    @staticmethod
    def backward(ctx, grad_out):
        st = ctx.saved_tensors
        (coeff, weight, q) = (st[0], st[1], st[2])
        g2d = grad_out.reshape(-1, ctx.out_features)
        grad_input = grad_weight = grad_bias = None
        if ctx.needs_input_grad[0]:
            w = weight.to(g2d.dtype)
            grad_input = g2d.mm(w.t() if ctx.transposed else w).reshape(
                *ctx.leading, ctx.in_features
            )
        if ctx.needs_input_grad[1]:
            grad_weight = g2d.t().mm(coeff.to(g2d.dtype)).mm(q.to(g2d.dtype).t())
            if ctx.transposed:
                grad_weight = grad_weight.t()
            grad_weight = grad_weight.to(weight.dtype)
        if ctx.needs_bias_grad:
            grad_bias = g2d.sum(0)
        return (grad_input, grad_weight, grad_bias, None, None, None, None, None)


class FatLinear(STACRLinear):
    def forward(self, x):
        x2d = x.reshape(-1, self.in_features)
        self._refresh_basis(x2d)
        budget = 0.0 if self._turbo() else self.residual_budget
        return FatFunction.apply(
            x,
            self.weight,
            self.bias,
            self.basis,
            budget,
            self.residual_bits,
            self.chunk_rows,
            self.transposed,
        )


def apply_fat(model, cfg, all_linears: bool):
    from stacr.surgery import select_target_names, _is_conv1d

    if all_linears:
        names = [
            n
            for (n, m) in model.named_modules()
            if isinstance(m, torch.nn.Linear)
            and m.weight.dim() == 2
            and (m.weight.shape[1] >= int(cfg.get("min_features", 256)))
        ]
    else:
        names = select_target_names(
            model,
            surface=str(cfg.get("surface", "mlp")),
            min_features=int(cfg.get("min_features", 256)),
        )
    wrapped = []
    for name in names:
        parent = model
        parts = name.split(".")
        for p in parts[:-1]:
            parent = getattr(parent, p)
        mod = getattr(parent, parts[-1])
        lay = FatLinear(
            mod.weight,
            getattr(mod, "bias", None),
            rank=int(cfg.get("rank", 8)),
            residual_budget=float(cfg.get("residual_budget", 0.005)),
            residual_bits=int(cfg.get("residual_bits", 8)),
            chunk_rows=int(cfg.get("chunk_rows", 512)),
            transposed=_is_conv1d(mod),
        )
        lay.source_name = name
        setattr(parent, parts[-1], lay)
        wrapped.append(lay)
    return wrapped


VARIANTS = [
    ("A_basis_per_call", dict(basis=True, fat=False, allin=False)),
    ("B_fat_residual", dict(basis=False, fat=True, allin=False)),
    ("C_all_linears", dict(basis=False, fat=False, allin=True)),
    ("D_all_three", dict(basis=True, fat=True, allin=True)),
]


def main():
    out = RESULTS / "e2c_reconcile.jsonl"
    from . import common as harness

    for tagm, model_id in MODELS:
        base = run_config(
            model_id=model_id,
            bs=BS,
            seq=SEQ,
            steps=STEPS,
            seed=42,
            method="baseline",
            amp=True,
            eight_bit=True,
            eval_batches=2,
            measure_from=6,
            tag=f"rec/{tagm}",
        )
        emit(base, out)
        bp = base["peak_bytes"]
        for label, v in VARIANTS:
            FatFunction.basis_per_call = v["basis"]
            FatFunction.fat_residual = v["fat"]
            cfg = {
                "rank": 8,
                "residual_budget": 0.005,
                "residual_bits": 8,
                "surface": "mlp",
                "min_features": 256,
                "chunk_rows": 512,
            }
            orig = harness.stacr_on_blocks
            harness.stacr_on_blocks = lambda m, c, p=None, _a=v["allin"]: apply_fat(m, c, _a)
            try:
                r = run_config(
                    model_id=model_id,
                    bs=BS,
                    seq=SEQ,
                    steps=STEPS,
                    seed=42,
                    method=label,
                    stacr_cfg=cfg,
                    amp=True,
                    eight_bit=True,
                    eval_batches=2,
                    measure_from=6,
                    baseline_peak_bytes=bp,
                    tag=f"rec/{tagm}",
                )
                r["measured_saving_pct"] = 100.0 * (bp - r["peak_bytes"]) / bp
                r["variant"] = label
                r["quality_metric_valid"] = False
                r["metric_scope"] = "allocation_only"
                emit(r, out)
            except Exception as exc:
                emit(
                    {
                        "tag": f"rec/{tagm}",
                        "method": label,
                        "status": "OOM_or_error",
                        "error": f"{type(exc).__name__}: {exc}"[:200],
                    },
                    out,
                )
            finally:
                harness.stacr_on_blocks = orig


if __name__ == "__main__":
    main()
