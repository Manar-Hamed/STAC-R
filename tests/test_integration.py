import copy

import pytest
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from stacr.layers import STACRLinear, _select_residual
from stacr.quant import quantize_values
from stacr.surgery import apply_stacr, set_step


def make_layer(**options):
    return STACRLinear.from_linear(nn.Linear(16, 7), rank=4, **options)


def test_multiple_outstanding_forwards_keep_their_basis_versions():
    layer = make_layer(residual_budget=0)
    x = torch.randn(5, 16, requires_grad=True)
    a = layer(x)
    saved = a.grad_fn.saved_tensors[2].clone()
    first = layer.basis
    layer.set_step(1)
    b = layer(x * 3)
    assert layer.basis is not first
    assert torch.equal(saved, first)
    (a.square().sum() + b.square().sum()).backward()
    assert torch.isfinite(layer.weight.grad).all()
    assert torch.isfinite(x.grad).all()


@pytest.mark.parametrize("use_reentrant", [False, True])
def test_checkpoint_replay_matches_direct_compressed_gradient(use_reentrant):
    direct = make_layer(residual_budget=0.2)
    replay = copy.deepcopy(direct)
    x = torch.randn(5, 16, requires_grad=True)
    y = x.detach().clone().requires_grad_(True)
    direct(x).square().sum().backward()
    checkpoint(replay, y, use_reentrant=use_reentrant).square().sum().backward()
    assert torch.allclose(replay.weight.grad, direct.weight.grad)
    assert torch.allclose(y.grad, x.grad)
    assert torch.equal(replay.basis, direct.basis)


def test_state_dict_resumes_basis_and_step():
    original = make_layer(residual_budget=0)
    original.set_step(9)
    original(torch.randn(5, 16)).sum().backward()
    restored = make_layer(residual_budget=0)
    restored.load_state_dict(original.state_dict())
    assert restored.step == 9
    assert bool(restored.basis_ready)
    assert torch.equal(restored.basis, original.basis)
    x = torch.randn(5, 16)
    for layer in (original, restored):
        layer.zero_grad()
        layer.set_step(10)
        layer(x).sum().backward()
    assert torch.equal(restored.basis, original.basis)
    assert torch.equal(restored.weight.grad, original.weight.grad)


@pytest.mark.parametrize("mode", ["eval", "no_grad", "frozen"])
def test_exact_paths_do_not_initialize_or_refresh_basis(mode):
    layer = make_layer()
    if mode == "eval":
        layer.eval()
    if mode == "frozen":
        layer.weight.requires_grad_(False)
    x = torch.randn(5, 16, requires_grad=True)
    with torch.set_grad_enabled(mode != "no_grad"):
        actual = layer(x)
        expected = F.linear(x, layer.weight, layer.bias)
    assert torch.equal(actual, expected)
    assert not bool(layer.basis_ready)
    if mode != "no_grad":
        actual.sum().backward()
        assert x.grad is not None


@pytest.mark.parametrize("transposed", [False, True])
def test_autocast_forward_input_and_bias_gradient_match_native(transposed):
    torch.manual_seed(14)
    w = nn.Parameter(torch.randn((16, 7) if transposed else (7, 16)))
    b = nn.Parameter(torch.randn(7))
    layer = STACRLinear(w, b, rank=4, transposed=transposed)
    x = torch.randn(3, 2, 16, requires_grad=True)
    xr = x.detach().clone().requires_grad_(True)
    wr, br = w.detach().clone().requires_grad_(True), b.detach().clone().requires_grad_(True)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        actual = layer(x)
        expected = (
            torch.addmm(br, xr.reshape(-1, 16), wr).reshape(3, 2, 7)
            if transposed
            else F.linear(xr, wr, br)
        )
    assert torch.equal(actual, expected)
    g = torch.randn_like(actual)
    actual.backward(g)
    expected.backward(g)
    assert torch.equal(x.grad, xr.grad)
    assert torch.equal(b.grad, br.grad)


def test_global_topk_is_independent_of_chunk_boundaries():
    x = torch.tensor([[100.0, 90.0, 80.0, 70.0], [4.0, 3.0, 2.0, 1.0]])
    q, c = torch.zeros(4, 1), torch.zeros(2, 1)
    i, v = _select_residual(x, c, q, 0.5, 1, "global_topk", 2, 0.1)
    assert set(i.tolist()) == {0, 1, 2, 3}
    assert set(v.tolist()) == {100.0, 90.0, 80.0, 70.0}
    local, _ = _select_residual(x, c, q, 0.5, 1, "chunk_topk", 2, 0.1)
    assert set(local.tolist()) == {0, 1, 4, 5}


@pytest.mark.parametrize(
    "mode", ["chunk_topk", "global_topk", "uniform", "block_sparse", "sampled", "none"]
)
@pytest.mark.parametrize("bits", [0, 4, 8])
def test_all_selectors_contract_from_the_saved_factors(mode, bits):
    torch.manual_seed(4)
    layer = make_layer(
        residual_budget=0.25,
        residual_bits=bits,
        residual_mode=mode,
        chunk_rows=2,
        residual_block_size=4,
    )
    out = layer(torch.randn(5, 16))
    coeff, _, q, *sparse = out.grad_fn.saved_tensors
    reconstruction = coeff @ q.T
    if sparse:
        indices, data, scale = sparse
        from stacr.quant import QuantizedValues

        values = QuantizedValues(data, scale, bits, indices.numel()).dequantize(
            reconstruction.dtype
        )
        reconstruction.reshape(-1)[indices.long()] += values
    g = torch.randn_like(out)
    expected = g.T @ reconstruction
    out.backward(g)
    assert torch.allclose(layer.weight.grad, expected, atol=2e-5, rtol=1e-5)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_quantization_uses_nonzero_fp32_scale_for_tiny_residuals(dtype):
    values = torch.tensor([0.0, 1e-6, -1e-6], dtype=dtype)
    packed = quantize_values(values, 8)
    assert packed.scale.dtype == torch.float32
    assert packed.scale > 0
    assert torch.allclose(packed.dequantize(torch.float32), values.float(), atol=1e-8)


@pytest.mark.parametrize(
    "options",
    [
        dict(rank=0),
        dict(rank=1.5),
        dict(residual_bits=7),
        dict(residual_budget=-1),
        dict(residual_budget=1.1),
        dict(update_interval=0),
        dict(chunk_rows=0),
        dict(residual_mode="typo"),
        dict(basis_dtype="typo"),
        dict(turbo_after_step=-1),
    ],
)
def test_invalid_layer_options_fail_before_wrapping(options):
    model = nn.Sequential(nn.Linear(16, 7), nn.Linear(7, 2))
    with pytest.raises((ValueError, TypeError)):
        apply_stacr(model, surface="all", min_features=1, **options)
    assert isinstance(model[0], nn.Linear)
    assert isinstance(model[1], nn.Linear)


def test_custom_target_selection_and_step_helper():
    model = nn.Sequential(nn.Linear(16, 7), nn.Linear(7, 2))
    layers = apply_stacr(model, target_names=["1"], rank=2)
    assert len(layers) == 1
    assert isinstance(model[0], nn.Linear)
    set_step(model, 4)
    assert model[1].step == 4
    with pytest.raises(ValueError, match="Unknown STAC-R"):
        apply_stacr(model, rnak=2)


def test_empty_selection_is_explicit_and_frozen_weights_are_skipped():
    model = nn.Sequential(nn.Linear(16, 7))
    with pytest.raises(ValueError, match="No trainable"):
        apply_stacr(model)
    assert apply_stacr(model, allow_empty=True) == []
    model[0].weight.requires_grad_(False)
    assert apply_stacr(model, surface="all", min_features=1, allow_empty=True) == []


def test_turbo_freezes_basis_but_weight_still_receives_gradient():
    layer = make_layer(turbo_after_step=1)
    layer(torch.randn(5, 16)).sum().backward()
    basis = layer.basis.clone()
    layer.set_step(2)
    layer.zero_grad()
    out = layer(torch.randn(5, 16))
    assert len(out.grad_fn.saved_tensors) == 3
    out.square().sum().backward()
    assert torch.equal(basis, layer.basis)
    assert layer.weight.grad.abs().sum() > 0


def test_surgery_preserves_shared_module_aliases():
    shared = nn.Linear(16, 16)
    model = nn.Sequential(shared, shared)
    layers = apply_stacr(model, surface="all", min_features=1, rank=4)
    assert len(layers) == 1
    assert model[0] is model[1]
    model(torch.randn(8, 16)).sum().backward()
    assert shared.weight.grad is not None
