import torch

from stacr.layers import STACRLinear
from stacr.surgery import apply_stacr, select_target_names


def _mlp(hidden=256, expand=4):
    return torch.nn.Sequential(
        torch.nn.Linear(hidden, expand * hidden),
        torch.nn.GELU(),
        torch.nn.Linear(expand * hidden, hidden),
    )


def _lowrank(rows, feats, rank, seed=0):
    g = torch.Generator().manual_seed(seed)
    u = torch.randn(rows, rank, generator=g)
    v = torch.randn(rank, feats, generator=g)
    return u @ v / rank**0.5


def test_input_gradient_is_exact():
    """grad_A = G W must use the true weight, never a compressed one."""
    torch.manual_seed(0)
    w = torch.nn.Parameter(torch.randn(64, 256) * 0.05)
    layer = STACRLinear(w, None, rank=8, residual_budget=0.005)
    x = _lowrank(128, 256, 12).requires_grad_(True)
    layer(x).pow(2).sum().backward()

    x2 = x.detach().clone().requires_grad_(True)
    torch.nn.functional.linear(x2, w.detach()).pow(2).sum().backward()
    assert torch.allclose(x.grad, x2.grad, atol=1e-5)


def test_backward_never_stores_the_dense_activation():
    """Saved payload must be the factors, not a tensor the size of the activation."""
    torch.manual_seed(0)
    rows, feats = 256, 512
    w = torch.nn.Parameter(torch.randn(128, feats) * 0.05)
    layer = STACRLinear(w, None, rank=8, residual_budget=0.005)
    x = _lowrank(rows, feats, 16)
    out = layer(x)
    saved = out.grad_fn.saved_tensors
    dense = rows * feats
    for t in saved:
        assert t.numel() < dense, f"a saved tensor has {t.numel()} elements, activation has {dense}"


def test_weight_gradient_matches_the_factor_identity():
    """grad_W must equal (G^T C) Q^T + G^T R_s on the stored factors."""
    torch.manual_seed(0)
    rows, feats, out_f = 128, 256, 64
    w = torch.nn.Parameter(torch.randn(out_f, feats) * 0.05)
    layer = STACRLinear(w, None, rank=8, residual_budget=0.01, residual_bits=0)
    x = _lowrank(rows, feats, 12)
    y = layer(x)
    g = torch.randn_like(y)
    y.backward(g)

    q = layer.basis.float()
    coeff = x.float() @ q
    recon = coeff @ q.t()
    resid = x.float() - recon
    k = max(1, int(round(0.01 * resid.numel())))
    _, idx = torch.topk(resid.reshape(-1).abs(), k)
    sparse = torch.zeros_like(resid).reshape(-1)
    sparse[idx] = resid.reshape(-1)[idx]
    expected = g.t().float() @ (recon + sparse.reshape_as(resid))
    assert torch.allclose(w.grad, expected, atol=1e-3)


def test_surgery_wraps_only_activations_and_keeps_parameters_intact():
    model = _mlp()
    before = [p.detach().clone() for p in model.parameters()]
    layers = apply_stacr(model, {"surface": "all", "min_features": 128, "rank": 8})
    assert len(layers) == 2
    after = [p.detach().clone() for p in model.parameters()]
    for a, b in zip(before, after):
        assert torch.equal(a, b), "surgery must not modify parameter values"


def test_select_target_names_respects_min_features():
    model = _mlp(hidden=64)
    assert select_target_names(model, surface="all", min_features=1024) == []


def test_turbo_drops_the_residual_channel():
    w = torch.nn.Parameter(torch.randn(64, 256) * 0.05)
    layer = STACRLinear(w, None, rank=8, residual_budget=0.01, turbo_after_step=2)
    x = _lowrank(64, 256, 8)
    layer.set_step(0)
    assert len(layer(x).grad_fn.saved_tensors) == 6
    layer.set_step(5)
    assert len(layer(x).grad_fn.saved_tensors) == 3
