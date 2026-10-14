from __future__ import annotations


def galore_available() -> bool:
    try:
        import galore_torch  # noqa: F401

        return True
    except Exception:
        return False


def create_galore_optimizer(
    model,
    lr: float = 1e-4,
    rank: int = 128,
    *,
    update_proj_gap: int = 200,
    scale: float = 0.25,
    proj_type: str = "std",
    weight_decay: float = 0.0,
    target_modules=None,
):
    """Build public GaLore parameter groups; its rank is independent of STAC-R rank."""
    try:
        from galore_torch import GaLoreAdamW
    except ImportError as exc:
        raise RuntimeError("GaLore requires: pip install 'stacr[galore]'") from exc
    if rank < 1 or update_proj_gap < 1 or scale <= 0:
        raise ValueError("GaLore rank, update_proj_gap, and scale must be positive")
    named = (
        model.named_parameters()
        if hasattr(model, "named_parameters")
        else ((str(i), p) for i, p in enumerate(model))
    )
    projected, regular = [], []
    for name, param in named:
        if not param.requires_grad:
            continue
        eligible = param.ndim == 2 and (
            target_modules is None or any(t in name for t in target_modules)
        )
        (projected if eligible else regular).append(param)
    if not projected:
        raise ValueError("No matrix parameters selected for GaLore")
    groups = [
        {
            "params": projected,
            "rank": rank,
            "update_proj_gap": update_proj_gap,
            "scale": scale,
            "proj_type": proj_type,
        }
    ]
    if regular:
        groups.append({"params": regular})
    return GaLoreAdamW(groups, lr=lr, weight_decay=weight_decay)
