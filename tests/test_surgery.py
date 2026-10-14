import pytest
import torch
from torch import nn

from stacr.compressor import STACRConfig
from stacr.layers import STACRLinear
from stacr.surgery import apply_stacr, classify, select_target_names, set_step


class _Proj(nn.Module):
    """Stand-in for the GPT-2 Conv1D convention: weight stored as (in, out)."""

    def __init__(self, nin, nout):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(nin, nout) * 0.02)
        self.bias = nn.Parameter(torch.zeros(nout))

    def forward(self, x):
        return x @ self.weight + self.bias


_Proj.__name__ = "Conv1D"


def _gpt2_like(hidden=512):
    block = nn.Module()
    block.attn = nn.Module()
    block.attn.c_attn = nn.Linear(hidden, 3 * hidden)
    block.attn.c_proj = nn.Linear(hidden, hidden)
    block.mlp = nn.Module()
    block.mlp.c_fc = nn.Linear(hidden, 4 * hidden)
    block.mlp.c_proj = nn.Linear(4 * hidden, hidden)
    root = nn.Module()
    root.h = nn.ModuleList([block])
    return root


def test_mlp_surface_excludes_attention_projections():
    """GPT-2 names the attention output projection c_proj too; it must not be wrapped."""
    model = _gpt2_like()
    names = select_target_names(model, surface="mlp", min_features=256)
    assert any(n.endswith("mlp.c_fc") for n in names)
    assert any(n.endswith("mlp.c_proj") for n in names)
    assert not any("attn" in n for n in names)


def test_attention_surface_selects_attention_only():
    model = _gpt2_like()
    names = select_target_names(model, surface="attention", min_features=256)
    assert all("attn" in n for n in names)
    assert any(n.endswith("c_attn") for n in names)


def test_bert_output_dense_is_split_by_parent_block():
    assert classify("encoder.layer.0.attention.output.dense") == "attention"
    assert classify("encoder.layer.0.output.dense") == "mlp"
    assert classify("encoder.layer.0.intermediate.dense") == "mlp"


def test_transposed_weight_forward_matches_reference():
    torch.manual_seed(0)
    root = nn.Module()
    root.mlp = nn.Module()
    root.mlp.c_fc = _Proj(256, 512)
    x = torch.randn(64, 256)
    expected = root.mlp.c_fc(x)
    apply_stacr(root, {"surface": "mlp", "min_features": 128, "rank": 8, "residual_budget": 0.0})
    assert torch.allclose(root.mlp.c_fc(x), expected, atol=1e-5)


def test_basis_follows_the_weight_device():
    layer_holder = _gpt2_like(hidden=256)
    layers = apply_stacr(layer_holder, {"surface": "mlp", "min_features": 128, "rank": 4})
    for layer in layers:
        assert layer.basis.device == layer.weight.device


@pytest.mark.parametrize("full", [False, True], ids=["turbo_default", "full_explicit"])
def test_turbo_step_50_freezes_basis_and_drops_residual_while_weights_train(full):
    torch.manual_seed(19)
    model = nn.Sequential(nn.Linear(12, 7, bias=False))
    weight = model[0].weight
    options = {"turbo_after_step": None} if full else {}
    layers = apply_stacr(
        model,
        surface="all",
        min_features=1,
        rank=3,
        residual_budget=0.25,
        update_interval=1,
        oja_lr=0.2,
        **options,
    )
    layer = layers[0]
    assert layer.weight is weight and weight.requires_grad
    assert layer.turbo_after_step == (None if full else 50)

    set_step(layers, 48)
    model(torch.randn(32, 12)).sum().backward()
    initialized_basis = layer.basis.detach().clone()

    set_step(layers, 49)
    model.zero_grad(set_to_none=True)
    before_switch = model(torch.randn(32, 12))
    assert before_switch.grad_fn.has_residual
    assert not torch.equal(layer.basis, initialized_basis)
    before_switch.sum().backward()
    step_49_basis = layer.basis.detach().clone()

    set_step(layers, 50)
    model.zero_grad(set_to_none=True)
    x = torch.randn(32, 12, requires_grad=True)
    upstream = torch.randn(32, 7)
    output = model(x)
    assert output.grad_fn.has_residual is full
    if full:
        assert not torch.equal(layer.basis, step_49_basis)
    else:
        torch.testing.assert_close(layer.basis, step_49_basis, rtol=0, atol=0)
    torch.testing.assert_close(output, x @ weight.t())
    output.backward(upstream)
    torch.testing.assert_close(x.grad, upstream @ weight)
    assert torch.isfinite(weight.grad).all() and weight.grad.norm() > 0
    if not full:
        low_rank_input = (x.detach() @ step_49_basis) @ step_49_basis.t()
        torch.testing.assert_close(weight.grad, upstream.t() @ low_rank_input)
    before_update = weight.detach().clone()
    torch.optim.SGD(model.parameters(), lr=0.01).step()
    assert not torch.equal(weight, before_update)


@pytest.mark.parametrize(
    "config,overrides,expected",
    [
        (None, {}, 50),
        ({}, {}, 50),
        ({"turbo_after_step": None}, {}, None),
        ({"turbo_after_step": 17}, {}, 17),
        ({"turbo_after_step": 17}, {"turbo_after_step": None}, None),
        (STACRConfig(), {}, None),
        (STACRConfig(), {"turbo_after_step": 50}, 50),
    ],
)
def test_apply_stacr_honors_explicit_turbo_configuration(config, overrides, expected):
    model = nn.Sequential(nn.Linear(12, 7))
    layers = apply_stacr(model, config, surface="all", min_features=1, rank=3, **overrides)
    assert layers[0].turbo_after_step == expected


def test_low_level_and_saved_tensor_defaults_remain_full():
    layer = STACRLinear.from_linear(nn.Linear(12, 7), rank=3)
    assert layer.turbo_after_step is None
    assert STACRConfig().turbo_after_step is None
