"""Single model fidelity."""

from __future__ import annotations
import torch
import sys
from pathlib import Path
import torch.nn.functional as F
from .common import RESULTS, emit, load_model, get_batches, free_all, set_seed, amp_ctx, _cos
from stacr.surgery import apply_stacr, set_step
from stacr.layers import STACRLinear

BASE = {
    "turbo_after_step": None,
    "residual_bits": 8,
    "surface": "mlp",
    "min_features": 256,
    "chunk_rows": 512,
}
MODELS = [
    ("llama32_1b", "meta-llama/Llama-3.2-1B", 1, 512),
    ("qwen25_1.5b", "Qwen/Qwen2.5-1.5B", 1, 512),
    ("qwen3_1.7b", "Qwen/Qwen3-1.7B", 1, 512),
    ("gpt2m", "gpt2-medium", 2, 1024),
]
_BYPASS = {"on": False}
_orig_forward = STACRLinear.forward


def _patched(self, x):
    if _BYPASS["on"]:
        w = self.weight
        return (
            x.matmul(w) + (self.bias if self.bias is not None else 0)
            if self.transposed
            else F.linear(x, w, self.bias)
        )
    return _orig_forward(self, x)


def grads(model, batch, seed, amp):
    model.zero_grad(set_to_none=True)
    set_seed(seed)
    with amp_ctx(amp):
        loss = model(**batch).loss
    loss.backward()
    return (
        {
            n: p.grad.detach().float().clone()
            for (n, p) in model.named_parameters()
            if p.grad is not None
        },
        float(loss),
    )


def run(model_id, bs, seq, rank, budget, tag, steps=4):
    free_all()
    set_seed(42)
    m = load_model(model_id, "cuda")
    m.train()
    layers = apply_stacr(m, dict(BASE, rank=rank, residual_budget=budget))
    import types

    for layer in layers:
        layer.forward = types.MethodType(_patched, layer)
    wrapped = {l.source_name for l in layers}
    if not layers:
        raise ValueError("Gradient fidelity requires at least one wrapped layer")
    data = get_batches(model_id, "train", bs, seq, steps, "wikitext", "cuda")
    (full, loc, un, gap) = ([], [], [], [])
    for i, b in enumerate(data):
        set_step(layers, i)
        _BYPASS["on"] = True
        (g_ref, l_ref) = grads(m, b, 1000 + i, False)
        _BYPASS["on"] = False
        (g_test, l_test) = grads(m, b, 1000 + i, False)
        gap.append(abs(l_ref - l_test))
        num = da = db = 0.0
        (L, U) = ([], [])
        for n, gr in g_ref.items():
            gt = g_test.get(n)
            if gt is None:
                continue
            (a, c) = (gr.reshape(-1).double(), gt.reshape(-1).double())
            num += float(a @ c)
            da += float(a @ a)
            db += float(c @ c)
            (L if n.rsplit(".", 1)[0] in wrapped else U).append(_cos(a, c))
        full.append(num / max(da**0.5 * db**0.5, 1e-30))
        if L:
            loc.append(sum(L) / len(L))
        if U:
            un.append(min(U))
    rec = {
        "tag": tag,
        "model": model_id,
        "bs": bs,
        "seq": seq,
        "rank": rank,
        "residual_budget": budget,
        "n_wrapped": len(layers),
        "mean_cosine_full_model": sum(full) / len(full),
        "mean_cosine_local_wrapped": sum(loc) / len(loc) if loc else float("nan"),
        "worst_unwrapped_cosine": min(un) if un else float("nan"),
        "mean_loss_gap": sum(gap) / len(gap),
    }
    rec["forward_mode"] = "train_rng_pinned"
    rec["seed"] = 42
    rec["steps"] = len(data)
    rec["stacr_cfg"] = dict(BASE, rank=rank, residual_budget=budget)
    rec["compression_training_layers"] = sum(layer.training for layer in layers)
    rec["initialized_bases"] = sum(bool(layer.basis_ready) for layer in layers)
    del m, layers, data
    free_all()
    return rec


def main():
    out = RESULTS / "q4b_fidelity_single.jsonl"
    for tag, mid, bs, seq in MODELS:
        for rank in [8, 32, 128]:
            for budget in [0.005, 0.0]:
                t = f"{tag}/r{rank}/b{budget}"
                try:
                    r = run(mid, bs, seq, rank, budget, t)
                    emit(r, out)
                    print(
                        f"  {t:<30} full={r['mean_cosine_full_model']:.4f} local={r['mean_cosine_local_wrapped']:.4f} unwrapped={r['worst_unwrapped_cosine']:.5f}",
                        flush=True,
                    )
                except Exception as exc:
                    emit(
                        {
                            "tag": t,
                            "status": "error",
                            "error": f"{type(exc).__name__}: {exc}"[:250],
                        },
                        out,
                    )


if __name__ == "__main__":
    main()
