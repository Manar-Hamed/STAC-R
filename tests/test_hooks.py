import torch

from stacr import STACRCompressor, STACRConfig, stacr_saved_tensors


def test_saved_tensor_context_runs_backward():
    model = torch.nn.Linear(64, 32)
    x = torch.randn(8, 64, requires_grad=True)
    compressor = STACRCompressor(
        STACRConfig(rank=4, residual_budget=0.01, min_features=1, min_numel=1)
    )
    with stacr_saved_tensors(compressor):
        loss = model(x).pow(2).mean()
    loss.backward()
    assert model.weight.grad is not None
    assert x.grad is not None
