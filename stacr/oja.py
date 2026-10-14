from __future__ import annotations

from dataclasses import dataclass

import torch


def orthonormalize(matrix: torch.Tensor) -> torch.Tensor:
    """Return an orthonormal basis spanning the columns of matrix."""
    q, _ = torch.linalg.qr(matrix.to(torch.float32), mode="reduced")
    return q.to(matrix.dtype)


@dataclass
class OjaBasis:
    """Persistent per-feature basis updated from streamed activations."""

    features: int
    rank: int
    dtype: torch.dtype = torch.float32
    device: torch.device | str = "cpu"
    initialized: bool = False

    def __post_init__(self) -> None:
        self.rank = max(1, min(int(self.rank), int(self.features)))
        self.basis = torch.empty(self.features, self.rank, device=self.device, dtype=self.dtype)
        self._views: dict[torch.dtype, torch.Tensor] = {}
        self.reset_random(seed=0)
        self.initialized = False

    def _invalidate(self) -> None:
        self._views = {}

    def reset_random(self, seed: int = 0) -> None:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed))
        sample = torch.randn(self.features, self.rank, generator=generator, dtype=torch.float32)
        self.basis = orthonormalize(sample.to(device=self.basis.device, dtype=self.dtype))
        self._invalidate()
        self.initialized = True

    def initialize_from_batch(self, activations: torch.Tensor, centre: bool = False) -> None:
        """Initialize the basis from the top right singular vectors of a batch.

        The basis is fitted to the *uncentred* activation by default, because the
        reconstruction ``C Q^T`` never adds a mean back. Fitting on centred data while
        projecting uncentred data discards the mean direction, which carries real energy
        in post-LayerNorm and post-GELU streams: on GPT-2 Medium at rank 8 it costs about
        0.20 of local per-layer gradient cosine.
        """
        x = activations.reshape(-1, activations.shape[-1]).detach().to(torch.float32)
        if centre:
            x = x - x.mean(dim=0, keepdim=True)
        if x.shape[0] < self.rank:
            self.reset_random(seed=0)
            return
        _, _, vh = torch.linalg.svd(x, full_matrices=False)
        basis = vh[: self.rank].t().contiguous()
        self.basis = orthonormalize(basis).to(device=self.basis.device, dtype=self.dtype)
        self._invalidate()
        self.initialized = True

    @torch.no_grad()
    def update(self, activations: torch.Tensor, lr: float = 0.01) -> None:
        """One Oja-style subspace update followed by re-orthogonalization."""
        if not self.initialized:
            self.initialize_from_batch(activations)
            return
        x = (
            activations.reshape(-1, activations.shape[-1])
            .detach()
            .to(self.basis.device, torch.float32)
        )
        if x.numel() == 0:
            return
        q = self.basis.to(torch.float32)
        y = x.mm(q)
        update = x.t().mm(y) / max(1, x.shape[0]) - q.mm(y.t().mm(y) / max(1, x.shape[0]))
        q = q + float(lr) * update
        self.basis = orthonormalize(q).to(dtype=self.dtype)
        self._invalidate()

    def snapshot(self, dtype: torch.dtype | None = None) -> torch.Tensor:
        out = self.basis.detach()
        if dtype is not None:
            out = out.to(dtype=dtype)
        return out.clone()

    def shared(self, dtype: torch.dtype | None = None) -> torch.Tensor:
        """Basis tensor for the current version, shared across all packed activations.

        Updates rebind ``self.basis`` to a fresh tensor rather than mutating it, so a
        reference handed out here stays valid for the activations that used it. Reusing
        one tensor keeps the persistent basis cost amortized instead of paying an
        ``N x r`` copy per saved activation.
        """
        target = self.basis.dtype if dtype is None else dtype
        cached = self._views.get(target)
        if cached is None:
            cached = (
                self.basis.detach()
                if target == self.basis.dtype
                else self.basis.detach().to(dtype=target)
            )
            self._views[target] = cached
        return cached
