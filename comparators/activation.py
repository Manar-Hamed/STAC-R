"""Matched-surface basis controls using STAC-R's factor-native backward.

These are independent implementations of the stated basis updates. They compare
basis choice at the same saved-activation surface, not each method's full system.
"""

from __future__ import annotations

import inspect

import torch

from stacr.layers import STACRLinear
from stacr.oja import orthonormalize


def _fixed_randn(rows, cols, device, salt=0):
    # A separate generator preserves the caller's dropout stream.
    generator = torch.Generator(device="cpu").manual_seed(1234 + rows + cols + salt)
    return torch.randn(rows, cols, generator=generator).to(device)


class UntrackedLinear(STACRLinear):
    """A fixed random orthonormal basis without a residual."""

    @torch.no_grad()
    def _refresh_basis(self, x2d):
        if not bool(self.basis_ready):
            initial = _fixed_randn(self.in_features, self.rank, x2d.device)
            self.basis = orthonormalize(initial).to(self.basis.dtype)
            self.basis_ready.fill_(True)


class OasisLinear(STACRLinear):
    """Oja update with normalized step and explicit orthonormalization.

    Initialization uses the leading covariance eigenvectors. The subsequent
    normalization estimates the covariance norm by ||CU|| / ||U||, matching
    the matched-surface protocol without forming C at every step.
    """

    gamma = 1.0

    @torch.no_grad()
    def _refresh_basis(self, x2d):
        if bool(self.basis_ready) and getattr(self, "_last_comparator_step", None) == self.step:
            return
        self._last_comparator_step = self.step
        x = x2d.detach().float()
        n = max(1, x.shape[0])
        if not bool(self.basis_ready):
            covariance = x.t().mm(x) / n
            _, vectors = torch.linalg.eigh(covariance)
            self.basis = orthonormalize(vectors[:, -self.rank :]).to(self.basis.dtype)
            self.basis_ready.fill_(True)
            return
        basis = self.basis.float()
        product = x.t().mm(x.mm(basis)) / n
        norm = float(product.norm()) / max(float(basis.norm()), 1e-12)
        update = product - basis.mm(basis.t().mm(product))
        self.basis = orthonormalize(basis + self.gamma / max(norm, 1e-12) * update).to(
            self.basis.dtype
        )


class PracLinear(STACRLinear):
    """Principal/random hybrid with a configurable interval refresh.

    Half the directions are principal components and half are an orthogonal
    random complement. A rank of floor(0.3 * input width) and refresh interval
    500 are available in the principal_random_rank experiment.
    """

    principal_frac = 0.5
    refresh_every = 500

    @torch.no_grad()
    def _refresh_basis(self, x2d):
        if bool(self.basis_ready) and getattr(self, "_last_comparator_step", None) == self.step:
            return
        self._last_comparator_step = self.step
        if bool(self.basis_ready) and self.step % int(self.refresh_every) != 0:
            return
        x = x2d.detach().float()
        principal_rank = min(self.rank, max(1, round(self.principal_frac * self.rank)), x.shape[0])
        _, _, vh = torch.linalg.svd(x, full_matrices=False)
        principal = vh[:principal_rank].t()
        random_rank = self.rank - principal_rank
        if random_rank:
            random = _fixed_randn(self.in_features, random_rank, x.device, salt=2)
            principal = torch.cat([principal, random], dim=1)
        self.basis = orthonormalize(principal).to(self.basis.dtype)
        self.basis_ready.fill_(True)


COMPARATORS = {
    "stacr": (STACRLinear, 0.005),
    "lowrank": (STACRLinear, 0.0),
    "untracked": (UntrackedLinear, 0.0),
    "oasis": (OasisLinear, 0.0),
    "prac": (PracLinear, 0.0),
}


def apply_comparator(model, kind: str, config: dict):
    """Replace selected projections; preserve Parameter identities and dtype."""
    from stacr.surgery import _is_conv1d, select_target_names

    if kind not in COMPARATORS:
        raise ValueError(f"Unknown basis comparator {kind!r}; choose {sorted(COMPARATORS)}")
    cls, budget = COMPARATORS[kind]
    cfg = dict(config)
    if kind == "stacr":
        cfg.setdefault("residual_budget", budget)
    else:
        cfg["residual_budget"] = 0.0
    targets = select_target_names(
        model,
        surface=cfg.get("surface", "mlp"),
        min_features=cfg.get("min_features", 256),
        extra_names=cfg.get("extra_names", ()),
    )
    accepted = inspect.signature(STACRLinear.__init__).parameters
    options = {key: value for key, value in cfg.items() if key in accepted}
    layers = []
    for name in targets:
        parent_name, _, leaf = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        source = getattr(parent, leaf)
        layer = cls(
            source.weight,
            getattr(source, "bias", None),
            **dict(options, transposed=_is_conv1d(source)),
        )
        layer.source_name = name
        layer.train(source.training)
        if isinstance(layer, PracLinear):
            layer.refresh_every = int(cfg.get("refresh_every", 500))
            layer.principal_frac = float(cfg.get("principal_frac", 0.5))
        if isinstance(layer, OasisLinear):
            layer.gamma = float(cfg.get("gamma", 1.0))
        setattr(parent, leaf, layer)
        layers.append(layer)
    return layers
