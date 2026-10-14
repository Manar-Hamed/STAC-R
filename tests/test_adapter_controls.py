"""Frozen-MLP composition controls retain the exact PEFT trainable surface."""

import copy

import pytest
import torch

from experiments.benchmarks.lora import _apply_mlp_composition_control
from experiments.benchmarks.qlora import _apply_mlp_composition_control as qlora_control


CONFIG = {"surface": "mlp", "min_features": 1, "rank": 2}


class AttentionAdapter(torch.nn.Module):
    """The relevant PEFT layout: frozen base, trainable low-rank attention update."""

    def __init__(self):
        super().__init__()
        self.base_layer = torch.nn.Linear(8, 8)
        self.base_layer.requires_grad_(False)
        self.lora_A = torch.nn.ModuleDict({"default": torch.nn.Linear(8, 2, bias=False)})
        self.lora_B = torch.nn.ModuleDict({"default": torch.nn.Linear(2, 8, bias=False)})

    def forward(self, inputs):
        return self.base_layer(inputs) + self.lora_B["default"](self.lora_A["default"](inputs))


class AdapterModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.attention = torch.nn.Module()
        self.attention.query = AttentionAdapter()
        self.mlp = torch.nn.Module()
        self.mlp.fc1 = torch.nn.Linear(8, 8)
        self.mlp.fc1.requires_grad_(False)

    def forward(self, inputs):
        return self.mlp.fc1(self.attention.query(inputs))


@pytest.mark.parametrize("control", [_apply_mlp_composition_control, qlora_control])
def test_adapter_control_has_no_trainable_mlp_but_keeps_adapter_gradients(control):
    model = AdapterModel()
    reference = copy.deepcopy(model)
    identities = {name: id(parameter) for name, parameter in model.named_parameters()}
    frozen = model.mlp.fc1
    layers, metadata = control(model, CONFIG, enabled=True, adapter=True)
    assert layers == []
    assert metadata["n_eligible_mlp_layers"] == metadata["n_wrapped_layers"] == 0
    assert metadata["n_frozen_mlp_layers"] == 1
    assert metadata["empty_compression_control"]
    assert metadata["stacr_cfg"]["allow_empty"]
    assert model.mlp.fc1 is frozen
    assert identities == {name: id(parameter) for name, parameter in model.named_parameters()}
    inputs = torch.randn(2, 3, 8)
    actual, expected = model(inputs), reference(inputs)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    actual.square().mean().backward()
    expected.square().mean().backward()
    for parameter, reference_parameter in zip(model.parameters(), reference.parameters()):
        if parameter.requires_grad:
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
            torch.testing.assert_close(parameter.grad, reference_parameter.grad, rtol=0, atol=0)
        else:
            assert parameter.grad is None


def test_empty_selection_is_not_allowed_for_full_finetuning_control():
    with pytest.raises(ValueError, match="No trainable"):
        _apply_mlp_composition_control(AdapterModel(), CONFIG, enabled=True, adapter=False)


def test_full_finetuning_still_compresses_eligible_mlp():
    model = AdapterModel()
    model.mlp.fc1.requires_grad_(True)
    original = model.mlp.fc1.weight
    layers, metadata = _apply_mlp_composition_control(model, CONFIG, enabled=True, adapter=False)
    assert len(layers) == metadata["n_wrapped_layers"] == metadata["n_eligible_mlp_layers"] == 1
    assert metadata["n_frozen_mlp_layers"] == 0
    assert not metadata["empty_compression_control"]
    assert "allow_empty" not in metadata["stacr_cfg"]
    assert model.mlp.fc1.weight is original


def test_frozen_packed_nf4_weight_is_never_replaced_or_reinterpreted():
    model = AdapterModel()
    # Preserve Linear's logical dimensions while modeling the packed uint8
    # parameter layout of bitsandbytes; selection must never read it as dense.
    packed = torch.nn.Parameter(torch.zeros(32, 1, dtype=torch.uint8), requires_grad=False)
    model.mlp.fc1.weight = packed
    original_module = model.mlp.fc1
    layers, metadata = qlora_control(model, CONFIG, enabled=True, adapter=True)
    assert layers == [] and metadata["empty_compression_control"]
    assert model.mlp.fc1 is original_module
    assert model.mlp.fc1.weight is packed
    assert packed.dtype == torch.uint8 and packed.shape == (32, 1)
