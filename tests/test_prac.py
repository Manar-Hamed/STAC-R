import torch

from comparators.prac import PRACCompressor, PRACConfig


def test_prac_reconstruction_shape():
    x = torch.randn(3, 16)
    recon = PRACCompressor(PRACConfig(rank=4)).reconstruct(x)
    assert recon.shape == x.shape
