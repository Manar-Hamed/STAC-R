import copy

import pytest
import torch
from torch import nn

from stacr.adaptive import STACRConfig, STACRLinear
from stacr.diagnostics import compression_diagnostics


def test_protocol_fidelity_keeps_compression_active_and_restores_state():
    from experiments.protocols.training import compute_grad_cosine

    torch.manual_seed(18)
    layer = STACRLinear(
        16,
        5,
        STACRConfig(
            rank=1,
            residual_mode="none",
            freeze_basis_after_warmup=False,
            update_interval=1,
            lazy_update_enabled=False,
        ),
    )
    model = nn.Sequential(nn.Dropout(0.9), layer)
    inputs = torch.randn(12, 16)
    labels = torch.randn(12, 5)
    model(inputs).square().sum().backward()
    layer.eval()  # A deliberately mixed mode must also survive the diagnostic.
    modes = [module.training for module in model.modules()]
    basis = layer.basis.clone()
    steps = layer.step_counter.clone()
    state = copy.deepcopy(layer.layer_state)
    config = layer.config
    output = compute_grad_cosine(model, inputs, labels, nn.MSELoss(), "synthetic")
    assert output["grad_cosine_by_layer"]["1.linear.weight"] < 0.99
    assert [module.training for module in model.modules()] == modes
    assert torch.equal(layer.basis, basis)
    assert torch.equal(layer.step_counter, steps)
    assert layer.layer_state == state
    assert layer.config is config
    assert all(parameter.grad is None for parameter in model.parameters())


def test_diagnostics_restore_state_after_exception():
    layer = STACRLinear(8, 3, STACRConfig(rank=2))
    layer(torch.randn(4, 8))
    state = copy.deepcopy(layer.layer_state)
    config = layer.config
    with pytest.raises(RuntimeError, match="diagnostic failed"):
        with compression_diagnostics(layer):
            layer.layer_state["basis_updates"] = 999
            layer.config.warmup_steps = 999
            raise RuntimeError("diagnostic failed")
    assert layer.layer_state == state
    assert layer.config is config
    assert layer.config.warmup_steps == 0
    assert layer.training


def test_diagnostics_require_initialized_adaptive_basis():
    layer = STACRLinear(8, 3, STACRConfig(rank=2))
    with pytest.raises(ValueError, match="Initialize adaptive bases"):
        with compression_diagnostics(layer):
            pass
