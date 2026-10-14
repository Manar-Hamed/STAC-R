"""
Gradient checkpointing for transformer blocks and attention modules.

Provides:
- HuggingFace transformer checkpointing
- Vision model stage/block checkpointing
- RecomputationGuard for safe hook handling during checkpointing
"""

import inspect
import threading
import warnings
from contextlib import contextmanager, nullcontext, ExitStack
from typing import Optional, Any
import torch
import torch.nn as nn


class RecomputationGuard:
    """Detect when we're in checkpointing recomputation vs original forward.

    Gradient checkpointing recomputes the forward pass during backward.
    This can double-trigger hooks (e.g., compression hooks). Use this guard
    to skip hook execution during recomputation.

    Usage:
        guard = RecomputationGuard()

        def forward_hook(module, input, output):
            if guard.is_recomputing:
                return  # Skip during recomputation
            # ... normal hook logic

        with guard.recomputation_context():
            # This marks the recomputation phase
            recomputed_output = checkpoint_fn(...)
    """

    def __init__(self):
        self._local = threading.local()

    @property
    def is_recomputing(self) -> bool:
        """Check if currently in recomputation phase."""
        return getattr(self._local, "recomputing", False)

    @contextmanager
    def recomputation_context(self):
        """Context manager to mark recomputation phase."""
        old_value = getattr(self._local, "recomputing", False)
        self._local.recomputing = True
        try:
            yield
        finally:
            self._local.recomputing = old_value


from stacr.adaptive_checkpointing import get_global_recomputation_guard as _shared_guard

_global_guard = _shared_guard()


def get_global_recomputation_guard() -> RecomputationGuard:
    """Get the global RecomputationGuard instance."""
    return _global_guard


_checkpoint_guard_installed = False
_orig_checkpoint = None
_checkpoint_guard_c1_warning_emitted = False


def _guard_context_fn(forward_context=None):
    from stacr.adaptive_checkpointing import checkpoint_contexts

    recording, replay = checkpoint_contexts()
    if forward_context is None:
        return recording, replay

    @contextmanager
    def combined():
        with ExitStack() as stack:
            stack.enter_context(recording)
            stack.enter_context(forward_context)
            yield

    return combined(), replay


def _supports_context_fn() -> bool:
    from torch.utils.checkpoint import checkpoint

    return "context_fn" in inspect.signature(checkpoint).parameters


def _checkpoint_with_guard(function, *args, **kwargs):
    from torch.utils.checkpoint import checkpoint

    if _supports_context_fn():
        kwargs.setdefault("context_fn", _guard_context_fn)
    return checkpoint(function, *args, **kwargs)


def _hf_checkpoint_with_guard(function, *args, **kwargs):
    kwargs.setdefault("use_reentrant", False)
    return _checkpoint_with_guard(function, *args, **kwargs)


def _warn_checkpoint_guard_c1_fallback(exc: Exception) -> None:
    global _checkpoint_guard_c1_warning_emitted
    if _checkpoint_guard_c1_warning_emitted:
        return
    warnings.warn(
        f"C1 checkpoint interception failed during checkpoint resolution and the call is falling back to native torch checkpointing. This can make a run behave like baseline checkpointing instead of C1. Root cause: {type(exc).__name__}: {exc}",
        RuntimeWarning,
        stacklevel=3,
    )
    _checkpoint_guard_c1_warning_emitted = True


def _override_hf_gradient_checkpointing_func(model: nn.Module) -> None:
    updated = False
    if hasattr(model, "gradient_checkpointing"):
        setattr(model, "_gradient_checkpointing_func", _hf_checkpoint_with_guard)
        updated = True
    for module in model.modules():
        if hasattr(module, "gradient_checkpointing"):
            setattr(module, "_gradient_checkpointing_func", _hf_checkpoint_with_guard)
            updated = True
    if not updated:
        raise ValueError(
            f"Model {type(model).__name__} did not expose any gradient_checkpointing modules to override."
        )


def _assert_hf_gradient_checkpointing_override(model: nn.Module) -> None:
    overridden = []
    missing = []
    if hasattr(model, "gradient_checkpointing"):
        if getattr(model, "_gradient_checkpointing_func", None) is _hf_checkpoint_with_guard:
            overridden.append(type(model).__name__)
        else:
            missing.append(type(model).__name__)
    for module_name, module in model.named_modules():
        if module is model:
            continue
        if not hasattr(module, "gradient_checkpointing"):
            continue
        if getattr(module, "_gradient_checkpointing_func", None) is _hf_checkpoint_with_guard:
            overridden.append(module_name)
        else:
            missing.append(module_name)
    if not overridden:
        raise RuntimeError(
            "HuggingFace checkpoint override did not attach to any gradient_checkpointing modules."
        )
    if missing:
        missing_str = ", ".join(missing[:5])
        raise RuntimeError(
            f"HuggingFace checkpoint override attached incompletely; missing on: {missing_str}"
        )


def _install_checkpoint_guard() -> None:
    """Monkeypatch torch.utils.checkpoint.checkpoint to set recomputation guard."""
    global _checkpoint_guard_installed, _orig_checkpoint
    if _checkpoint_guard_installed:
        return
    from torch.utils import checkpoint as ckpt_mod

    if "context_fn" not in inspect.signature(ckpt_mod.checkpoint).parameters:
        _checkpoint_guard_installed = True
        return
    _orig_checkpoint = ckpt_mod.checkpoint

    def _wrapped_checkpoint(function, *args, **kwargs):
        use_reentrant = kwargs.get("use_reentrant", None)
        c1_site = None
        runtime = None
        preserve_rng_state = kwargs.get("preserve_rng_state", True)
        if use_reentrant is False or use_reentrant is None:
            if "context_fn" not in kwargs or kwargs["context_fn"] is None:
                try:
                    from stacr.protocols.checkpointing import (
                        get_active_c1_runtime,
                        stacr_checkpoint,
                    )

                    runtime = get_active_c1_runtime()
                    if runtime is not None:
                        c1_site = runtime.resolve_site(function, args)
                except Exception as exc:
                    _warn_checkpoint_guard_c1_fallback(exc)
                    runtime = None
                    c1_site = None
        if c1_site is not None and runtime is not None:
            if kwargs.get("context_fn") is not None:
                raise ValueError(
                    "C1 checkpoint compression does not support a custom context_fn. Disable C1 or remove the custom checkpoint context."
                )
            if kwargs.get("use_reentrant", False):
                raise ValueError("C1 checkpoint compression requires use_reentrant=False.")
            if kwargs.get("debug", False):
                raise ValueError(
                    "C1 checkpoint compression does not support checkpoint(debug=True)."
                )
            if kwargs.get("determinism_check", "default") not in ("default", None):
                raise ValueError(
                    "C1 checkpoint compression supports only determinism_check='default'."
                )
            return stacr_checkpoint(
                function, *args, resolved=c1_site, preserve_rng_state=preserve_rng_state
            )
        if use_reentrant is False or use_reentrant is None:
            if "context_fn" not in kwargs or kwargs["context_fn"] is None:
                kwargs["context_fn"] = _guard_context_fn
                kwargs.setdefault("use_reentrant", False)
        return _orig_checkpoint(function, *args, **kwargs)

    ckpt_mod.checkpoint = _wrapped_checkpoint
    _checkpoint_guard_installed = True


_install_checkpoint_guard()


def apply_checkpointing(model: nn.Module, mode: str) -> None:
    """Enable gradient checkpointing on a model.

    Args:
        model: PyTorch model instance
        mode: Checkpointing mode:
            - "none": No checkpointing
            - "transformer_hf": HuggingFace gradient_checkpointing_enable()
            - "vision_stage": Stage-level checkpointing (ResNet-like)
            - "vision_block": Block-level checkpointing

    Raises:
        ValueError: If mode is unknown or model doesn't support requested mode
    """
    if mode == "none":
        return
    _install_checkpoint_guard()
    if mode == "transformer_hf":
        if hasattr(model, "gradient_checkpointing_enable"):
            gc_enable = model.gradient_checkpointing_enable
            gc_signature = inspect.signature(gc_enable)
            if "gradient_checkpointing_kwargs" in gc_signature.parameters:
                gc_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            else:
                gc_enable()
            _override_hf_gradient_checkpointing_func(model)
            _assert_hf_gradient_checkpointing_override(model)
            if hasattr(model, "config") and hasattr(model.config, "use_cache"):
                model.config.use_cache = False
        else:
            raise ValueError(
                f"Model {type(model).__name__} does not support gradient_checkpointing_enable(). Use 'none' or implement manually."
            )
    elif mode == "vision_stage":
        _apply_vision_stage_checkpointing(model)
    elif mode == "vision_block":
        _apply_vision_block_checkpointing(model)
    elif mode == "attention_only":
        _apply_attention_only_checkpointing(model)
    else:
        raise ValueError(f"Unknown checkpointing mode: {mode}")


def disable_checkpointing(model: nn.Module, mode: str) -> None:
    """Disable gradient checkpointing on a model.

    Args:
        model: PyTorch model instance
        mode: Same mode that was used for apply_checkpointing

    Note:
        Vision modes (vision_stage, vision_block) modify forward methods
        and cannot be cleanly disabled. Only transformer_hf is fully
        reversible. For vision models, create a fresh model instance.
    """
    if mode == "none":
        return
    if mode == "transformer_hf":
        if hasattr(model, "gradient_checkpointing_disable"):
            model.gradient_checkpointing_disable()
            if hasattr(model, "config") and hasattr(model.config, "use_cache"):
                model.config.use_cache = True


def _apply_vision_stage_checkpointing(model: nn.Module) -> None:
    """Apply stage-level checkpointing to vision models.

    Works with ResNet-like architectures that have layer1, layer2, etc.
    """
    stage_names = ["layer1", "layer2", "layer3", "layer4"]
    found_stages = [name for name in stage_names if hasattr(model, name)]
    if not found_stages:
        raise ValueError(
            f"Model {type(model).__name__} does not have ResNet-like stages. Use 'none' or implement custom checkpointing."
        )
    for name in found_stages:
        stage = getattr(model, name)
        if isinstance(stage, nn.Sequential):
            wrapped = _CheckpointedSequential(stage)
            setattr(model, name, wrapped)


def _apply_vision_block_checkpointing(model: nn.Module) -> None:
    """Apply block-level checkpointing to vision models.

    Wraps individual residual blocks (BasicBlock, Bottleneck) with checkpointing.
    More fine-grained than stage-level, saves more memory but slower.
    """

    def make_checkpointed_forward(original_forward):

        def checkpointed_forward(*args, **kwargs):

            def run_fn(*inputs):
                return original_forward(*inputs, **kwargs)

            return _checkpoint_with_guard(run_fn, *args, use_reentrant=False)

        return checkpointed_forward

    for name, module in model.named_modules():
        if module.__class__.__name__ in ("BasicBlock", "Bottleneck"):
            original_forward = module.forward
            module.forward = make_checkpointed_forward(original_forward)


class _CheckpointedSequential(nn.Module):
    """Sequential module with gradient checkpointing."""

    def __init__(self, sequential: nn.Sequential):
        super().__init__()
        self.sequential = sequential

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        from torch.utils.checkpoint import checkpoint_sequential

        return checkpoint_sequential(
            self.sequential, segments=len(self.sequential), input=x, use_reentrant=False
        )


def _apply_attention_only_checkpointing(model: nn.Module) -> None:
    """Apply checkpointing to attention layers only.

    For Transformers, checkpoints only the attention layers, leaving FFN layers
    free for STAC-R compression. This combines:
    - Attention checkpointing recomputes attention activations during backward.
    - STAC-R compresses saved inputs to feed-forward projections.

    Works with:
    - HuggingFace BERT/RoBERTa: bert.encoder.layer[N].attention
    - GPT-2: transformer.h[N].attn
    - Other transformers with 'attention' or 'attn' in module names

    Note: This mode explicitly avoids checkpointing FFN/MLP layers to allow
    STAC-R to compress those activations instead.
    """

    def make_checkpointed_forward(original_forward):

        def checkpointed_forward(*args, **kwargs):

            def run_fn(*inputs):
                return original_forward(*inputs, **kwargs)

            return _checkpoint_with_guard(run_fn, *args, use_reentrant=False)

        return checkpointed_forward

    checkpointed_count = 0
    wrapped_containers = set()
    for name, module in model.named_modules():
        is_attention = (
            "attention" in name.lower() or "attn" in name.lower() or "self_attn" in name.lower()
        )
        is_ffn = (
            "intermediate" in name.lower()
            or "output.dense" in name.lower()
            or "mlp" in name.lower()
            or ("feed_forward" in name.lower())
            or ("ffn" in name.lower())
        )
        if is_attention and (not is_ffn):
            children = list(module.children())
            is_container = len(children) > 0
            if (
                is_container
                and hasattr(module, "forward")
                and ("Attention" in module.__class__.__name__)
            ):
                original_forward = module.forward
                module.forward = make_checkpointed_forward(original_forward)
                checkpointed_count += 1
                wrapped_containers.add(name)
            elif not is_container:
                if any((name.startswith(prefix + ".") for prefix in wrapped_containers)):
                    continue
                original_forward = module.forward
                module.forward = make_checkpointed_forward(original_forward)
                checkpointed_count += 1
    if checkpointed_count == 0:
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
            if hasattr(model, "config") and hasattr(model.config, "use_cache"):
                model.config.use_cache = False


@contextmanager
def checkpointing_context(model: nn.Module, mode: str):
    """Context manager for temporary checkpointing.

    Usage:
        with checkpointing_context(model, "transformer_hf"):
            output = model(input)  # Runs with checkpointing
        # Checkpointing disabled after context
    """
    apply_checkpointing(model, mode)
    try:
        yield
    finally:
        disable_checkpointing(model, mode)
