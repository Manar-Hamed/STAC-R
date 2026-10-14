import torch
from torch import nn

from stacr.surgery import apply_stacr, set_step


class _Block(nn.Module):
    """Minimal decoder-style block: a stochastic path plus an MLP."""

    def __init__(self, hidden=256, p=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(hidden)
        self.drop = nn.Dropout(p)
        self.mlp = nn.Module()
        self.mlp.c_fc = nn.Linear(hidden, 4 * hidden)
        self.mlp.act = nn.GELU()
        self.mlp.c_proj = nn.Linear(4 * hidden, hidden)

    def forward(self, x):
        h = self.drop(self.norm(x))
        return x + self.mlp.c_proj(self.mlp.act(self.mlp.c_fc(h)))


def _grads(model, x, seed=None):
    model.zero_grad(set_to_none=True)
    if seed is not None:
        torch.manual_seed(seed)
    model(x).pow(2).sum().backward()
    return {n: p.grad.detach().clone() for n, p in model.named_parameters() if p.grad is not None}


def _cos(a, b):
    a, b = a.reshape(-1).double(), b.reshape(-1).double()
    return float(a @ b / (a.norm() * b.norm()))


def test_unwrapped_parameters_keep_exact_gradients():
    """grad_A = G W is exact, so nothing outside the wrapped surface may drift."""
    torch.manual_seed(0)
    ref = _Block()
    test = _Block()
    test.load_state_dict(ref.state_dict())
    layers = apply_stacr(
        test, {"surface": "mlp", "min_features": 128, "rank": 8, "residual_budget": 0.005}
    )
    assert len(layers) == 2
    wrapped = {layer.source_name + ".weight" for layer in layers}
    x = torch.randn(64, 32, 256)

    set_step(layers, 0)
    g_ref = _grads(ref, x, seed=7)
    g_test = _grads(test, x, seed=7)
    for name, gr in g_ref.items():
        if name in wrapped:
            continue
        assert _cos(gr, g_test[name]) > 1 - 1e-6, f"{name} drifted"


def test_unpinned_stochastic_path_dominates_the_measured_gap():
    """Two unseeded passes draw different dropout masks; the replay must pin the seed."""
    torch.manual_seed(0)
    ref = _Block(p=0.1)
    test = _Block(p=0.1)
    test.load_state_dict(ref.state_dict())
    layers = apply_stacr(
        test, {"surface": "mlp", "min_features": 128, "rank": 8, "residual_budget": 0.005}
    )
    x = torch.randn(64, 32, 256)
    set_step(layers, 0)

    pinned = _cos(
        *(lambda a, b: (a["norm.weight"], b["norm.weight"]))(
            _grads(ref, x, seed=11), _grads(test, x, seed=11)
        )
    )
    torch.manual_seed(123)
    a = _grads(ref, x)
    b = _grads(test, x)
    unpinned = _cos(a["norm.weight"], b["norm.weight"])
    assert pinned > 1 - 1e-6
    assert unpinned < pinned, "unpinned masks should look worse than the real gap"


def test_forward_output_is_unchanged_by_compression():
    torch.manual_seed(0)
    ref = _Block(p=0.0)
    test = _Block(p=0.0)
    test.load_state_dict(ref.state_dict())
    apply_stacr(test, {"surface": "mlp", "min_features": 128, "rank": 8})
    x = torch.randn(16, 32, 256)
    with torch.no_grad():
        assert torch.allclose(ref(x), test(x), atol=1e-5)
