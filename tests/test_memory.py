import torch
import pytest

from stacr.memory import dense_bytes, saved_surface_predictor, stacr_stored_bytes


def test_stacr_storage_is_smaller_than_dense_for_rank8():
    shape = (8, 512)
    dense = dense_bytes(shape, torch.float16)
    stored = stacr_stored_bytes(shape, rank=8, dtype=torch.float16, residual_budget=0.005)
    assert stored < dense


def test_saved_surface_predictor_fraction():
    assert saved_surface_predictor(10, 100) == 0.1


@pytest.mark.parametrize("bits", [0, 4, 8])
def test_quantized_storage_matches_linear_payload_estimate(bits):
    from stacr.layers import STACRLinear

    layer = STACRLinear.from_linear(
        torch.nn.Linear(16, 7), rank=4, residual_budget=0.25, residual_bits=bits
    )
    estimate = stacr_stored_bytes(
        (5, 16),
        rank=4,
        dtype=torch.float32,
        residual_budget=0.25,
        residual_bits=bits,
        basis_dtype=torch.float32,
        include_basis=True,
    )
    assert estimate == layer.stored_bytes(5, value_bytes=4)["total"]


@pytest.mark.parametrize("peak", [0, -1, float("nan"), float("inf")])
def test_predictor_rejects_invalid_baseline_peak(peak):
    with pytest.raises(ValueError):
        saved_surface_predictor(1, peak)


@pytest.mark.parametrize("mode", ["chunk_topk", "block_sparse", "global_topk", "uniform"])
def test_layer_capacity_accounts_for_actual_rounding_and_saved_basis_cast(mode):
    from stacr.layers import STACRLinear

    layer = STACRLinear.from_linear(
        torch.nn.Linear(16, 7),
        rank=4,
        residual_mode=mode,
        chunk_rows=1,
        residual_block_size=4,
        residual_budget=0.17,
    )
    with torch.autocast("cpu", dtype=torch.bfloat16):
        output = layer(torch.randn(5, 16))
    coeff, weight, basis, indices, values, scale = output.grad_fn.saved_tensors
    inventory = sum(
        tensor.numel() * tensor.element_size() for tensor in (coeff, basis, indices, values, scale)
    )
    persistent_basis = layer.basis.numel() * layer.basis.element_size()
    estimate = layer.stored_bytes(5, value_bytes=2)
    assert estimate["residual_count"] == indices.numel()
    assert estimate["total"] == inventory + persistent_basis
    assert estimate["saved_basis_cast"] == basis.numel() * basis.element_size()
