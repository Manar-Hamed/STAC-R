import torch

from stacr.residual import select_topk_residual


def test_sparse_residual_reconstructs_selected_entries():
    residual = torch.tensor([[1.0, -3.0, 2.0, 0.5]])
    packed = select_topk_residual(residual, budget=0.5)
    dense = packed.dense(device=residual.device, dtype=residual.dtype)
    assert int((dense != 0).sum()) == 2
    assert torch.allclose(dense.abs().max(), torch.tensor(3.0))
