from __future__ import annotations

import inspect
import re
from dataclasses import asdict, is_dataclass
from typing import Any, Iterable

from torch import nn

from .layers import STACRLinear

# A module is placed on a surface by its parent block, not by its leaf name alone:
# GPT-2 names both the MLP output and the attention output projection "c_proj", and
# BERT names both feed-forward and attention output projections "dense".
MLP_PARENTS = ("mlp", "ffn", "feed_forward", "intermediate")
ATTENTION_PARENTS = ("attn", "attention", "self_attn", "self_attention", "crossattention")
MLP_LEAVES = (
    "c_fc",
    "c_proj",
    "dense",
    "dense_h_to_4h",
    "dense_4h_to_h",
    "gate_proj",
    "up_proj",
    "down_proj",
    "fc1",
    "fc2",
    "w1",
    "w2",
    "w3",
)
ATTENTION_LEAVES = (
    "c_attn",
    "c_proj",
    "dense",
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "out_proj",
    "query",
    "key",
    "value",
    "query_key_value",
)

# OPT names its feed-forward projections directly on the decoder layer, as
# `model.decoder.layers.N.fc1`, with no `mlp` parent at all, so the parent-based rules below miss
# them and the MLP surface comes back empty for the whole family.  Keying on the leaf alone is
# safe for exactly these two names and would not be for `dense` or `c_proj`, which name attention
# projections in BERT and GPT-2 respectively: fc1 and fc2 never do.  The attention branch is
# still tested first, so anything under an attention parent cannot reach this rule.
UNAMBIGUOUS_MLP_LEAVES = ("fc1", "fc2")


def _is_conv1d(module: nn.Module) -> bool:
    return (
        type(module).__name__ == "Conv1D" and hasattr(module, "weight") and module.weight.dim() == 2
    )


def _feature_count(module: nn.Module) -> int | None:
    if isinstance(module, nn.Linear):
        return int(module.in_features)
    if _is_conv1d(module):
        return int(module.weight.shape[0])
    return None


def classify(name: str) -> str | None:
    """Return 'attention', 'mlp', or None for a module's saved-activation surface."""
    parts = name.split(".")
    leaf, parents = parts[-1], set(parts[:-1])
    if any(p in parents for p in ATTENTION_PARENTS):
        return "attention" if leaf in ATTENTION_LEAVES else None
    if any(p in parents for p in MLP_PARENTS) or "output" in parents:
        return "mlp" if leaf in MLP_LEAVES else None
    if leaf in UNAMBIGUOUS_MLP_LEAVES:
        return "mlp"
    return None


def select_target_names(
    model: nn.Module,
    *,
    surface: str = "mlp",
    min_features: int = 256,
    extra_names: Iterable[str] = (),
    target_names: Iterable[str] | None = None,
    target_pattern: str | None = None,
    exclude_names: Iterable[str] = (),
    trainable_only: bool = True,
) -> list[str]:
    """Select trainable linear inputs by surface, exact names, or regex.

    ``target_names`` and ``target_pattern`` restrict the selection. Exact
    ``extra_names`` add modules to a surface (use ``surface='none'`` for a custom
    list). Explicit names bypass the automatic minimum-feature threshold.
    """
    if surface not in {"mlp", "attention", "mlp+attention", "all", "none"}:
        raise ValueError(f"Unknown surface: {surface!r}")
    if isinstance(min_features, bool) or int(min_features) != min_features or min_features < 1:
        raise ValueError("min_features must be a positive integer")

    def name_set(values):
        return {values} if isinstance(values, str) else set(values)

    extras = name_set(extra_names)
    targets = None if target_names is None else name_set(target_names)
    excluded = name_set(exclude_names)
    pattern = None if target_pattern is None else re.compile(target_pattern)
    modules = dict(model.named_modules(remove_duplicate=False))
    explicit = extras | (targets or set())
    missing = explicit - modules.keys()
    if missing:
        raise ValueError(f"Unknown target module names: {sorted(missing)}")
    invalid = [name for name in explicit if _feature_count(modules[name]) is None]
    if invalid:
        raise ValueError(f"Targets are not supported linear projections: {sorted(invalid)}")
    wanted = {
        "mlp": {"mlp"},
        "attention": {"attention"},
        "mlp+attention": {"mlp", "attention"},
    }.get(surface)
    names = []
    for name, module in modules.items():
        feats = _feature_count(module)
        if feats is None or name in excluded:
            continue
        if trainable_only and not module.weight.requires_grad:
            continue
        if targets is not None and name not in targets:
            continue
        if pattern is not None and pattern.search(name) is None:
            continue
        if feats < int(min_features) and name not in explicit:
            continue
        if surface == "all" or name in explicit or targets is not None or pattern is not None:
            names.append(name)
            continue
        kind = classify(name)
        if wanted is not None and kind in wanted:
            names.append(name)
    return names


def apply_stacr(
    model: nn.Module, config: dict[str, Any] | None = None, **overrides: Any
) -> list[STACRLinear]:
    """Replace selected linear projections with Turbo-fast STAC-R layers, in place.

    By default, the basis freezes and residual storage stops at optimizer step 50;
    model weights continue training. Call ``set_step(layers, step)`` before each
    training forward to advance this schedule. Pass ``turbo_after_step=None`` for
    full STAC-R throughout training, or an integer to choose the transition step.

    Explicit config values are preserved: ``STACRConfig()`` supplies ``None`` and
    therefore selects full STAC-R unless overridden. Only saved input activations
    are compressed; the forward pass and input gradient stay exact.
    """
    cfg: dict[str, Any] = asdict(config) if is_dataclass(config) else dict(config or {})
    cfg.update(overrides)
    cfg.setdefault("turbo_after_step", 50)
    layer_keys = set(inspect.signature(STACRLinear.__init__).parameters) - {
        "self",
        "weight",
        "bias",
        "transposed",
    }
    selection_keys = {
        "surface",
        "min_features",
        "extra_names",
        "target_names",
        "target_pattern",
        "exclude_names",
        "trainable_only",
        "allow_empty",
    }
    # These keys belong to the generic saved-tensor storage path; accepting the
    # defaults keeps STACRConfig usable here without conflating the two paths.
    compatibility_keys = {"min_numel", "post_warmup_mode"}
    unknown = cfg.keys() - layer_keys - selection_keys - compatibility_keys
    if unknown:
        raise ValueError(f"Unknown STAC-R options: {sorted(unknown)}")
    if cfg.get("post_warmup_mode", "full") != "full":
        raise ValueError("Use turbo_after_step to switch the factor-native layer to Turbo")
    targets = select_target_names(
        model,
        surface=str(cfg.get("surface", "mlp")),
        min_features=cfg.get("min_features", 256),
        extra_names=cfg.get("extra_names", ()),
        target_names=cfg.get("target_names"),
        target_pattern=cfg.get("target_pattern"),
        exclude_names=cfg.get("exclude_names", ()),
        trainable_only=cfg.get("trainable_only", True),
    )
    if "" in targets:
        raise ValueError("For a standalone linear module use STACRLinear.from_linear(model, ...)")
    if not targets and not cfg.get("allow_empty", False) and cfg.get("surface") != "none":
        raise ValueError(
            "No trainable linear projections matched. Select surface='all', supply "
            "target_names, lower min_features, or explicitly set allow_empty=True."
        )
    # Construct the complete plan before mutating the model: an invalid option
    # must never leave a partially wrapped model behind.
    plan = []
    replacements = {}
    for name in targets:
        parent = model
        parts = name.split(".")
        for part in parts[:-1]:
            parent = getattr(parent, part)
        module = getattr(parent, parts[-1])
        layer = replacements.get(id(module))
        if layer is None:
            layer = STACRLinear.from_linear(
                module, **{key: cfg[key] for key in layer_keys if key in cfg}
            )
            layer.source_name = name
            replacements[id(module)] = layer
        plan.append((parent, parts[-1], layer))
    wrapped: list[STACRLinear] = []
    for parent, leaf, layer in plan:
        setattr(parent, leaf, layer)
        if layer not in wrapped:
            wrapped.append(layer)
    return wrapped


def set_step(
    layers: Iterable[STACRLinear] | nn.Module,
    step: int,
    *,
    microbatch_index: int = 0,
    accum_steps: int = 1,
) -> None:
    """Set optimizer scheduling state on either STAC-R engine or a model."""
    from .adaptive import STACRLinear as AdaptiveLinear

    if isinstance(step, bool) or int(step) != step or step < 0:
        raise ValueError("step must be a nonnegative integer")
    if int(accum_steps) != accum_steps or accum_steps < 1:
        raise ValueError("accum_steps must be a positive integer")
    if int(microbatch_index) != microbatch_index or not 0 <= microbatch_index < accum_steps:
        raise ValueError("microbatch_index must be in [0, accum_steps)")
    if isinstance(layers, nn.Module):
        layers = (
            module
            for module in layers.modules()
            if isinstance(module, (STACRLinear, AdaptiveLinear))
        )
    for layer in layers:
        if isinstance(layer, AdaptiveLinear):
            layer.set_runtime_state(
                optimizer_step=step, microbatch_index=microbatch_index, accum_steps=accum_steps
            )
        else:
            layer.set_step(step)
