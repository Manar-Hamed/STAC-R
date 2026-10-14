import copy

import pytest
import torch
from torch import nn

from stacr.adaptive import (
    STACRConfig,
    STACRLinear,
    apply_adaptive_stacr,
    get_basis_pool,
    quantize_int4,
    dequantize_int4,
    quantize_int8,
    dequantize_int8,
    set_stacr_runtime_state,
)
from stacr.adaptive_checkpointing import checkpoint


@pytest.mark.parametrize(
    "mode", ["none", "topk", "threshold", "block_sparse", "adaptive_threshold", "sparse_only"]
)
def test_adaptive_modes_preserve_forward_and_input_gradient(mode):
    torch.manual_seed(21)
    layer = STACRLinear(16, 7, STACRConfig(rank=4, residual_mode=mode, residual_budget=0.2))
    x = torch.randn(8, 16, requires_grad=True)
    reference = x.detach().clone().requires_grad_(True)
    expected = reference.mm(layer.linear.weight.t()) + layer.linear.bias
    actual = layer(x)
    assert torch.equal(actual, expected)
    grad = torch.randn_like(actual)
    actual.backward(grad)
    expected.backward(grad)
    assert torch.equal(x.grad, reference.grad)
    assert torch.isfinite(layer.linear.weight.grad).all()


def test_adaptive_state_restores_runtime_controller_and_basis():
    cfg = STACRConfig(
        rank=4,
        residual_mode="topk",
        rank_mode="sensitivity",
        rank_min=1,
        rank_max=4,
        rank_adjust_interval=1,
        freeze_basis_after_warmup=False,
        update_interval=1,
        lazy_update_enabled=False,
        oja_update_per_optimizer_step=True,
    )
    first = STACRLinear(16, 7, cfg)
    set_stacr_runtime_state(first, 3)
    first(torch.randn(8, 16)).sum().backward()
    state = copy.deepcopy(first.state_dict())
    second = STACRLinear(16, 7, copy.deepcopy(cfg))
    second.load_state_dict(state)
    assert second.layer_state == first.layer_state
    assert torch.equal(second.basis, first.basis)
    for layer in (first, second):
        layer.zero_grad()
        set_stacr_runtime_state(layer, 4)
    x = torch.randn(8, 16)
    first(x).sum().backward()
    second(x).sum().backward()
    assert torch.equal(second.basis, first.basis)
    assert torch.equal(second.linear.weight.grad, first.linear.weight.grad)


def test_adaptive_variance_rank_checkpoint_restores_changed_shape():
    cfg = STACRConfig(
        rank=8, rank_mode="variance", rank_min=1, rank_max=8, rank_variance_threshold=0.9
    )
    first = STACRLinear(16, 7, cfg)
    x = torch.randn(12, 1) @ torch.randn(1, 16)
    first(x).sum().backward()
    assert first.basis.shape[1] == 1
    second = STACRLinear(16, 7, copy.deepcopy(cfg))
    second.load_state_dict(first.state_dict())
    assert torch.equal(first.basis, second.basis)
    assert second.basis_initialized


def test_adaptive_repeated_forward_preserves_saved_basis():
    layer = STACRLinear(
        16,
        7,
        STACRConfig(
            rank=4,
            residual_mode="topk",
            freeze_basis_after_warmup=False,
            update_trigger="fixed",
            update_interval=2,
            lazy_update_enabled=False,
        ),
    )
    x = torch.randn(8, 16)
    out = layer(x)
    saved = out.grad_fn.basis.clone()
    layer(x * 2).sum().backward()
    assert torch.equal(out.grad_fn.basis, saved)
    out.sum().backward()


@pytest.mark.parametrize("frozen", [False, True])
def test_adaptive_checkpoint_guard_preserves_gradient(frozen):
    cfg = STACRConfig(
        rank=4,
        residual_mode="topk",
        residual_budget=0.2,
        freeze_basis_after_warmup=frozen,
        update_interval=1,
        lazy_update_enabled=False,
    )
    first = STACRLinear(16, 7, cfg)
    second = copy.deepcopy(first)
    x = torch.randn(8, 16, requires_grad=True)
    y = x.detach().clone().requires_grad_(True)
    first(x).square().sum().backward()
    checkpoint(second, y).square().sum().backward()
    assert torch.allclose(first.linear.weight.grad, second.linear.weight.grad)
    assert torch.equal(x.grad, y.grad)
    assert int(first.step_counter) == int(second.step_counter) == 1


@pytest.mark.parametrize("mode", ["eval", "no_grad", "frozen"])
def test_adaptive_exact_paths_do_not_initialize(mode):
    layer = STACRLinear(16, 7, STACRConfig(rank=4))
    if mode == "eval":
        layer.eval()
    if mode == "frozen":
        layer.linear.weight.requires_grad_(False)
    with torch.set_grad_enabled(mode != "no_grad"):
        x = torch.randn(3, 16)
        assert torch.equal(layer(x), layer.linear(x))
    assert not layer.basis_initialized
    assert int(layer.step_counter) == 0


def test_adaptive_surgery_preserves_parameters_and_rng():
    model = nn.Sequential(nn.Linear(16, 7))
    parameters = list(model.parameters())
    rng = torch.get_rng_state()
    layers = apply_adaptive_stacr(model, {"rank": 4}, target_names=["0"])
    assert torch.equal(rng, torch.get_rng_state())
    assert all(a is b for a, b in zip(parameters, model.parameters()))
    assert layers[0].source_name == "0"
    model(torch.randn(8, 16)).sum().backward()


def test_adaptive_conv1d_surgery_keeps_original_parameter_identity():
    class Conv1D(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.randn(16, 7))
            self.bias = nn.Parameter(torch.randn(7))

        def forward(self, x):
            return torch.addmm(self.bias, x, self.weight)

    model = nn.Sequential(Conv1D())
    weight = model[0].weight
    x = torch.randn(8, 16)
    expected = model(x)
    apply_adaptive_stacr(model, rank=4, target_names=["0"])
    assert weight is model[0].linear.source_weight
    assert torch.allclose(model(x), expected, atol=1e-5)
    model(x).sum().backward()
    assert weight.grad is not None


@pytest.mark.parametrize("compression", ["csr", "delta"])
def test_adaptive_index_compression_runs_backward(compression):
    layer = STACRLinear(
        16,
        7,
        STACRConfig(
            rank=4, residual_mode="topk", residual_budget=0.3, index_compression=compression
        ),
    )
    layer(torch.randn(8, 16)).square().sum().backward()
    assert torch.isfinite(layer.linear.weight.grad).all()


@pytest.mark.parametrize("bits", [4, 8])
def test_adaptive_factor_quantization_roundtrip(bits):
    x = torch.tensor([-2.0, -0.5, 0.0, 0.5, 2.0])
    if bits == 4:
        data, scale, length = quantize_int4(x)
        recovered = dequantize_int4(data, scale, length)
    else:
        data, scale = quantize_int8(x)
        recovered = dequantize_int8(data, scale)
    assert torch.all((recovered - x).abs() <= scale / 2 + 1e-6)


def test_adaptive_config_rejects_unimplemented_or_invalid_settings():
    for options in [
        dict(streaming_algorithm="incremental_svd"),
        dict(update_interval=0),
        dict(residual_budget=2),
        dict(rank=0),
        dict(residual_quantize_bits=2),
        dict(basis_sharing="all", rank_mode="sensitivity"),
    ]:
        with pytest.raises(ValueError):
            STACRConfig(**options)


@pytest.mark.parametrize("compression", ["csr", "delta"])
def test_sparse_indices_above_int32_range_roundtrip(compression):
    from stacr.index_compression import compress_indices, decompress_indices
    from stacr.adaptive import _choose_index_dtype

    indices = torch.tensor([2**31 + 15, 2**32 - 1, 2**31 + 2], dtype=torch.int64)
    packed = compress_indices(indices, B_flat=4, N=2**30, mode=compression)
    assert torch.equal(decompress_indices(packed), indices)
    assert _choose_index_dtype(2**32) == torch.int64


def test_saved_tensor_hooks_never_compress_parameter_views():
    from stacr.compressor import STACRCompressor, STACRConfig

    weight = nn.Parameter(torch.randn(7, 16))
    compressor = STACRCompressor(STACRConfig(min_features=1, min_numel=1))
    assert not compressor.should_compress(weight.t())


@pytest.mark.parametrize("bits", [4, 8])
@pytest.mark.parametrize("value", [0.0, 6e-8])
def test_adaptive_quantization_preserves_zero_and_tiny_fp16_values(bits, value):
    values = torch.full((5,), value, dtype=torch.float16)
    if bits == 4:
        data, scale, length = quantize_int4(values)
        restored = dequantize_int4(data, scale, length)
    else:
        data, scale = quantize_int8(values)
        restored = dequantize_int8(data, scale)
    torch.testing.assert_close(restored.float(), values.float(), rtol=0.01, atol=0)


def test_adaptive_checkpoint_accepts_torch_basis_dtype():
    layer = STACRLinear(8, 3, STACRConfig(rank=2, basis_dtype=torch.float32))
    layer(torch.randn(4, 8))
    restored = STACRLinear(8, 3, STACRConfig(rank=2))
    restored.load_state_dict(layer.state_dict())
    assert torch.equal(restored.basis, layer.basis)
