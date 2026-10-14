import pytest
import torch

from stacr.protocols.baselines.amp import AMPManager, resolve_amp_dtype


@pytest.mark.parametrize("device", ["cuda", "cuda:1", torch.device("cuda:1")])
def test_indexed_cuda_fp16_uses_scaler(device):
    assert resolve_amp_dtype("fp16", device) == (torch.float16, True)


@pytest.mark.parametrize(
    "major, expected", [(7, (torch.float16, True)), (8, (torch.bfloat16, False))]
)
def test_auto_amp_queries_the_requested_cuda_device(monkeypatch, major, expected):
    requested = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    def capability(device):
        requested.append(device)
        return major, 0

    monkeypatch.setattr(torch.cuda, "get_device_capability", capability)
    assert resolve_amp_dtype("auto", "cuda:1") == expected
    assert requested == [torch.device("cuda:1")]


def test_cpu_amp_uses_no_cuda_scaler():
    manager = AMPManager("auto", device="cpu")
    assert manager.dtype == torch.bfloat16
    assert manager.scaler is None
    with manager.autocast():
        result = torch.randn(2, 3) @ torch.randn(3, 2)
    assert result.dtype == torch.bfloat16
