"""Integration invariants that can be checked without downloads or CUDA."""

import sys
from types import SimpleNamespace

import pytest
import torch

from comparators.activation import apply_comparator
from comparators.instant.custom_op import LinearSVDOp
from comparators import subtrack_adapter
from stacr.surgery import apply_stacr, set_step


@pytest.mark.parametrize("kind", ["untracked", "oasis", "prac", "lowrank"])
def test_basis_control_preserves_forward_rng_and_parameter_identity(kind):
    model = torch.nn.Sequential(torch.nn.Linear(8, 5))
    original = model[0].weight
    x = torch.randn(2, 3, 8, requires_grad=True)
    expected = model(x).detach()
    layers = apply_comparator(model, kind, {"surface": "all", "min_features": 1, "rank": 3})
    set_step(layers, 0)
    state = torch.random.get_rng_state().clone()
    output = model(x)
    torch.testing.assert_close(output, expected)
    assert model[0].weight is original
    assert torch.equal(torch.random.get_rng_state(), state)
    output.square().sum().backward()
    assert torch.isfinite(original.grad).all()
    assert torch.isfinite(x.grad).all()


def test_instant_identity_projection_matches_dense_backward():
    x = torch.randn(2, 4, 3, requires_grad=True)
    weight = torch.randn(5, 3, requires_grad=True)
    bias = torch.randn(5, requires_grad=True)
    projection = torch.eye(4)
    compressed = LinearSVDOp.apply(x, weight, bias, projection, projection)
    dense = torch.nn.functional.linear(x, weight, bias)
    upstream = torch.randn_like(dense)
    expected = torch.autograd.grad(dense, (x, weight, bias), upstream)
    actual = torch.autograd.grad(compressed, (x, weight, bias), upstream)
    torch.testing.assert_close(compressed, dense)
    for left, right in zip(actual, expected):
        torch.testing.assert_close(left, right)


@pytest.mark.parametrize("engine", ["compact", "adaptive"])
@pytest.mark.parametrize("projection", ["linear", "conv1d"])
@pytest.mark.parametrize("explicit", [False, True])
def test_subtrack_groups_include_stacr_wrapped_weights(monkeypatch, engine, projection, explicit):
    monkeypatch.setattr(subtrack_adapter, "require_subtrack", lambda: None)
    monkeypatch.setattr(subtrack_adapter, "subtrack_revision", lambda: "test")
    monkeypatch.setitem(
        sys.modules, "low_rank_torch", SimpleNamespace(LowRankAdamW=torch.optim.AdamW)
    )
    model = torch.nn.Module()
    model.mlp = torch.nn.Module()

    class Conv1D(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.randn(8, 4))
            self.bias = torch.nn.Parameter(torch.randn(4))

        def forward(self, x):
            return torch.addmm(self.bias, x.reshape(-1, 8), self.weight).reshape(*x.shape[:-1], 4)

    model.mlp.fc1 = torch.nn.Linear(8, 4) if projection == "linear" else Conv1D()
    weight = model.mlp.fc1.weight
    from stacr.adaptive import apply_adaptive_stacr
    apply = apply_adaptive_stacr if engine == "adaptive" else apply_stacr
    layers = apply(model, {"surface": "mlp", "min_features": 1, "rank": 2})
    config = {"subtrack_rank": 2}
    if explicit:
        config["subtrack_target_modules"] = ["mlp.fc1"]
    optimizer = subtrack_adapter.create_subtrack_optimizer(model, config)
    projected = optimizer.param_groups[1]["params"]
    assert len(projected) == 1
    assert projected[0] is weight
    assert projected[0].is_leaf
    assert optimizer.param_groups[1]["subspace_update_method"] == "subtrack"
    assert optimizer.stacr_projected_parameter_names == ["mlp.fc1.weight"]
    grouped = [parameter for group in optimizer.param_groups for parameter in group["params"]]
    assert len(grouped) == len({id(parameter) for parameter in grouped})
    assert {id(parameter) for parameter in grouped} == {id(parameter) for parameter in model.parameters()}
    set_step(layers, 0)
    model.mlp.fc1(torch.randn(2, 3, 8)).square().mean().backward()
    assert weight.grad is not None and torch.isfinite(weight.grad).all()
    before = weight.detach().clone()
    optimizer.step()
    assert not torch.equal(weight, before)
