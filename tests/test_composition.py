import copy

import pytest
import torch
from torch import nn

from stacr.adaptive_checkpointing import checkpoint
from stacr.layers import STACRLinear
from stacr.training import configure_model


class Blocks(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList(
            [nn.Sequential(nn.Linear(8, 8), nn.Linear(8, 8)) for _ in range(3)]
        )

    def forward(self, x):
        for block in self.model.layers:
            x = block(x)
        return x


@pytest.mark.parametrize("engine", ["projection", "adaptive"])
def test_checkpoint_complement_intersects_explicit_targets(engine):
    model = Blocks()
    config = {
        "task": "synthetic",
        "method": "stacr_checkpointing",
        "engine": engine,
        "rank": 2,
        "checkpoint_every": 2,
        "compression": {
            "surface": "all",
            "target_names": ["model.layers.0.0", "model.layers.1.1"],
            "basis_dtype": "float32",
        },
    }
    layers, info = configure_model(model, config)
    assert info["wrapped_names"] == ["model.layers.1.1"]
    assert len(layers) == 1
    assert info["checkpointed_blocks"] == [0, 2]
    model(torch.randn(4, 8, requires_grad=True)).square().sum().backward()
    assert all(parameter.grad is not None for parameter in model.parameters())


def test_invalid_compression_blocks_rejected_before_mutation():
    model = Blocks()
    with pytest.raises(ValueError, match="compression_blocks"):
        configure_model(model, {"compression_blocks": "typo"})
    assert isinstance(model.model.layers[0][0], nn.Linear)
    assert not hasattr(model.model.layers[0], "_stacr_checkpoint_forward")


def test_native_checkpoint_replays_saved_basis_and_turbo_step():
    direct = STACRLinear.from_linear(
        nn.Linear(8, 4), rank=2, turbo_after_step=1, residual_budget=0.25
    )
    replay = copy.deepcopy(direct)
    x = torch.randn(4, 8, requires_grad=True)
    y = x.detach().clone().requires_grad_(True)
    direct_losses, replay_losses = [], []
    for step in (0, 1):
        direct.set_step(step)
        replay.set_step(step)
        direct_losses.append(direct(x + step).square().sum())
        replay_losses.append(checkpoint(replay, y + step).square().sum())
    sum(direct_losses).backward()
    sum(replay_losses).backward()
    torch.testing.assert_close(replay.weight.grad, direct.weight.grad)
    torch.testing.assert_close(y.grad, x.grad)
    assert replay.step == 1
