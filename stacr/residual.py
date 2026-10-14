from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class SparseResidual:
    shape: tuple[int, ...]
    indices: torch.Tensor
    values: torch.Tensor

    def dense(self, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        out = torch.zeros(self.shape, device=device, dtype=dtype)
        if self.indices.numel() > 0:
            flat = out.reshape(-1)
            flat[self.indices.to(device=device, dtype=torch.long)] = self.values.to(
                device=device, dtype=dtype
            )
        return out


def select_topk_residual(residual: torch.Tensor, budget: float) -> SparseResidual:
    """Keep the largest-magnitude residual entries under a fractional budget."""
    residual = residual.detach()
    flat = residual.reshape(-1)
    if budget <= 0 or flat.numel() == 0:
        empty_i = torch.empty(0, device=residual.device, dtype=torch.long)
        empty_v = torch.empty(0, device=residual.device, dtype=residual.dtype)
        return SparseResidual(tuple(residual.shape), empty_i, empty_v)
    k = max(1, min(flat.numel(), int(round(float(budget) * flat.numel()))))
    _, indices = torch.topk(flat.abs(), k=k, largest=True, sorted=False)
    return SparseResidual(tuple(residual.shape), indices.to(torch.long), flat[indices].clone())
