"""Keep the recorded optimizer settings explicit across bitsandbytes APIs."""

import sys
from types import SimpleNamespace

import pytest

from stacr.protocols.baselines.bnb_optimizer import BNBStatus, try_create_8bit_optimizer


class CurrentAdam:
    def __init__(self, params, lr, betas, eps, weight_decay, min_8bit_size):
        self.params = params
        self.lr = lr
        self.min_8bit_size = min_8bit_size


class LegacyAdam(CurrentAdam):
    def __init__(
        self, params, lr, betas, eps, weight_decay, min_8bit_size, percentile_clipping, block_wise
    ):
        super().__init__(params, lr, betas, eps, weight_decay, min_8bit_size)
        self.percentile_clipping = percentile_clipping
        self.block_wise = block_wise


def install_fake(monkeypatch, optimizer_class):
    monkeypatch.setitem(
        sys.modules,
        "bitsandbytes",
        SimpleNamespace(optim=SimpleNamespace(AdamW8bit=optimizer_class, Adam8bit=optimizer_class)),
    )
    return [SimpleNamespace(is_cuda=True, numel=lambda: 8192)]


@pytest.mark.parametrize("optim_type", ["adam", "adamw"])
def test_current_bnb_keeps_equivalent_default_settings(monkeypatch, optim_type):
    params = install_fake(monkeypatch, CurrentAdam)
    result = try_create_8bit_optimizer(params, optim_type=optim_type, lr=3e-5)
    assert result.ok
    assert result.optimizer.params == params
    assert result.optimizer.lr == 3e-5
    assert result.meta["compatibility_omitted_defaults"] == ["percentile_clipping", "block_wise"]


@pytest.mark.parametrize("options", [{"percentile_clipping": 50}, {"block_wise": False}])
def test_current_bnb_rejects_unsupported_nondefault_settings(monkeypatch, options):
    result = try_create_8bit_optimizer(install_fake(monkeypatch, CurrentAdam), **options)
    assert result.status == BNBStatus.ERROR
    assert "no longer supports" in result.reason
    assert next(iter(options)) in result.reason


def test_legacy_bnb_receives_requested_settings(monkeypatch):
    result = try_create_8bit_optimizer(
        install_fake(monkeypatch, LegacyAdam), percentile_clipping=50, block_wise=False
    )
    assert result.ok
    assert result.optimizer.percentile_clipping == 50
    assert result.optimizer.block_wise is False
    assert result.meta["compatibility_omitted_defaults"] == []
