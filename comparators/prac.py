from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class PRACConfig:
    rank: int = 8
    seed: int = 0


class PRACCompressor:
    """Untracked random-subspace activation compressor."""

    def __init__(self, config: PRACConfig | None = None):
        self.config = config or PRACConfig()
        self._projections: dict[tuple[int, torch.device], torch.Tensor] = {}

    def projection(self, features: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        key = (int(features), device)
        if key not in self._projections:
            generator = torch.Generator(device="cpu")
            generator.manual_seed(int(self.config.seed) + int(features))
            rank = max(1, min(int(self.config.rank), int(features)))
            p = torch.randn(features, rank, generator=generator, dtype=torch.float32)
            p = p / (rank**0.5)
            self._projections[key] = p.to(device=device)
        return self._projections[key].to(dtype=dtype)

    def reconstruct(self, tensor: torch.Tensor) -> torch.Tensor:
        x = tensor.reshape(-1, tensor.shape[-1])
        p = self.projection(tensor.shape[-1], tensor.device, tensor.dtype)
        x_proj = x.mm(p)
        return x_proj.mm(p.t()).reshape_as(tensor)
