from __future__ import annotations

import torch


def cosine_similarity(
    reference: torch.Tensor,
    estimate: torch.Tensor,
    eps: float = 1e-12,
    *,
    reduction_dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """Flattened cosine, with both-zero=1 and exactly-one-zero=0.

    Scaling each vector before reduction avoids overflow and underflow. Tiny
    nonzero gradients retain their direction instead of being called identical.
    """
    if reference.shape != estimate.shape or reference.device != estimate.device:
        raise ValueError("Cosine inputs must have the same shape and device")
    if eps < 0 or reduction_dtype not in (torch.float32, torch.float64):
        raise ValueError("eps must be nonnegative; reduction_dtype must be float32 or float64")
    ref = reference.detach().reshape(-1).to(reduction_dtype)
    est = estimate.detach().reshape(-1).to(reduction_dtype)
    if not torch.isfinite(ref).all() or not torch.isfinite(est).all():
        raise ValueError("Cosine inputs must be finite")
    if ref.numel() == 0:
        return ref.new_ones(())
    ref_scale, est_scale = ref.abs().max(), est.abs().max()
    if ref_scale == 0 or est_scale == 0:
        return ref.new_tensor(1.0 if ref_scale == 0 and est_scale == 0 else 0.0)
    ref, est = ref / ref_scale, est / est_scale
    return ((ref @ est) / (ref.norm() * est.norm())).clamp(-1.0, 1.0)


def relative_mse(
    reference: torch.Tensor, estimate: torch.Tensor, eps: float = 1e-12
) -> torch.Tensor:
    """Return mean-squared error normalized by reference energy."""
    ref = reference.detach().to(torch.float32)
    est = estimate.detach().to(torch.float32)
    return (ref - est).pow(2).mean() / ref.pow(2).mean().clamp_min(eps)


def gradient_cosine_from_activations(
    activation: torch.Tensor,
    reconstructed: torch.Tensor,
    grad_output: torch.Tensor,
) -> torch.Tensor:
    """Compute the induced weight-gradient cosine for a linear layer."""
    a = activation.reshape(-1, activation.shape[-1]).to(torch.float32)
    ahat = reconstructed.reshape(-1, reconstructed.shape[-1]).to(torch.float32)
    g = grad_output.reshape(-1, grad_output.shape[-1]).to(torch.float32)
    grad_ref = g.t().mm(a)
    grad_est = g.t().mm(ahat)
    return cosine_similarity(grad_ref, grad_est)
