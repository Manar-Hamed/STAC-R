import math

import pytest
import torch

from stacr.analysis import gradient_bridge, residual_proxy_gain, surface_ledger


def test_bridge_holds_for_nontrivial_compression():
    torch.manual_seed(12)
    a, g = torch.randn(12, 9), torch.randn(12, 5)
    result = gradient_bridge(a, a + 0.03 * torch.randn_like(a), g)
    assert result["bridge_holds"]
    assert result["cosine_bound_holds"]
    assert result["gradient_relative_error"] <= result["gradient_error_upper"]
    assert result["cosine"] >= result["sharp_cosine_bound"] - 1e-10


def test_bridge_does_not_invent_relative_bound_for_zero_gradient():
    result = gradient_bridge(torch.zeros(4, 3), torch.ones(4, 3), torch.ones(4, 2))
    assert result["gradient_relative_error"] is None
    assert result["sharp_cosine_bound"] is None
    assert result["cosine"] == 0


def test_residual_table_proxy_uses_stated_formula():
    # Published cosines are rounded, so the resulting proxy is approximate.
    assert residual_proxy_gain(0.9813, 0.9792) == pytest.approx(0.193, abs=0.001)


def test_surface_ledger_keeps_basis_separate():
    ledger = surface_ledger(100, 20, 1000, basis_bytes=10, workspace_bytes=5)
    assert ledger["ssp"] == 0.08
    assert ledger["net_after_workspace_bytes"] == 65
