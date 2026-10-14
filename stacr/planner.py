from __future__ import annotations

from dataclasses import dataclass


@dataclass
class LayerPlan:
    name: str
    features: int
    rank: int
    residual_budget: float
    compress: bool


def choose_rank(features: int, requested_rank: int, min_rank: int = 1) -> int:
    return max(int(min_rank), min(int(requested_rank), int(features)))


def plan_linear_layers(
    named_modules, rank: int = 8, residual_budget: float = 0.005, min_features: int = 128
):
    """Return compression plans for modules with an input feature dimension."""
    plans = []
    for name, module in named_modules:
        features = getattr(module, "in_features", None)
        if features is None:
            continue
        features = int(features)
        plans.append(
            LayerPlan(
                name=name,
                features=features,
                rank=choose_rank(features, rank),
                residual_budget=float(residual_budget),
                compress=features >= int(min_features),
            )
        )
    return plans
