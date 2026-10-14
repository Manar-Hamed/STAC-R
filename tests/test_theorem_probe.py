import math
from types import SimpleNamespace

import pytest
import torch

from scripts.theorem_probe import (
    _collect_linear_stats,
    _cosine_bound,
    _load_model_state,
    _pair_metrics,
    _projection_metrics,
)


def test_projection_bridge_and_residual_correction_match_direct_algebra():
    generator = torch.Generator().manual_seed(7)
    activation = torch.randn(17, 8, generator=generator)
    upstream = torch.randn(17, 5, generator=generator)
    true_gradient = (upstream.T @ activation).flatten()
    internal, projected = _projection_metrics(
        {"layer": activation}, {"layer": upstream}, {"layer": true_gradient}, rank=3
    )
    row = projected["layer"]
    assert row["grad_vector_matches_weight_grad"]
    assert row["cos_proj_observed"] == pytest.approx(row["cos_q_bridge"], abs=2e-6)
    assert row["cos_proj_observed"] == pytest.approx(math.sqrt(1 - row["p"]), abs=2e-6)
    full_gradient = true_gradient - 0.25 * internal["layer"]["delta_proj"]
    corrected = _pair_metrics(internal, projected, {"layer": full_gradient})["layer"]
    assert corrected["s"] == pytest.approx(row["p"] / 16, abs=1e-7)
    assert corrected["xi"] == pytest.approx(-0.75 * row["p"], abs=1e-6)
    assert corrected["cos_full_observed"] + 1e-6 >= corrected["cos_full_bound"]


def test_cosine_bound_marks_invalid_domain_and_checkpoint_weights_are_required():
    assert _cosine_bound(0.36) == pytest.approx(0.8)
    assert _cosine_bound(1) is None
    assert _cosine_bound(4) is None
    layer = torch.nn.Linear(4, 3)
    with pytest.raises(RuntimeError, match="weight"):
        _load_model_state(layer, {"bias": layer.bias.detach().clone()})


def test_checkpoint_migrates_legacy_runtime_but_requires_basis():
    from stacr.adaptive import AdaptiveSTACRConfig, AdaptiveSTACRLinear

    config = AdaptiveSTACRConfig(rank=2, residual_mode="none", basis_dtype="float32")
    layer = AdaptiveSTACRLinear(8, 3, config)
    layer(torch.randn(6, 8))
    state = layer.state_dict()
    legacy = {
        key: value
        for key, value in state.items()
        if key not in {"_runtime_state", "_shared_basis_state", "_basis_initialized"}
    }
    restored = AdaptiveSTACRLinear(8, 3, config)
    _load_model_state(restored, legacy)
    assert restored.basis_initialized
    torch.testing.assert_close(restored.basis, layer.basis)
    broken = {key: value for key, value in state.items() if key != "basis"}
    with pytest.raises(RuntimeError, match="basis"):
        _load_model_state(AdaptiveSTACRLinear(8, 3, config), broken)


def test_probe_activates_compression_and_preserves_basis_and_model_mode():
    from stacr.adaptive import AdaptiveSTACRConfig, AdaptiveSTACRLinear

    class TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.output = torch.nn.Module()
            self.output.dense = AdaptiveSTACRLinear(
                8, 3, AdaptiveSTACRConfig(rank=2, residual_mode="none", basis_dtype="float32")
            )
            self.dropout = torch.nn.Dropout(0.5)

        def forward(self, inputs):
            return SimpleNamespace(loss=self.output.dense(self.dropout(inputs)).sum())

    model = TinyModel().train()
    inputs = torch.randn(13, 8, generator=torch.Generator().manual_seed(81))
    model(inputs)
    basis = model.output.dense.basis.clone()
    steps = model.output.dense.step_counter.clone()
    _, _, gradients = _collect_linear_stats(model, {"inputs": inputs}, torch.device("cpu"))
    exact = inputs.sum(0).repeat(3).flatten()
    assert not torch.allclose(gradients["output.dense"], exact, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(model.output.dense.basis, basis)
    torch.testing.assert_close(model.output.dense.step_counter, steps)
    assert model.training and model.dropout.training
