"""Compose activation compression with block recomputation and CPU offload."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from functools import wraps

import torch
from torch import nn


def saved_tensors_on_cpu(*, pin_memory: bool = True):
    """Offload saved tensors while preserving their dimension order in memory.

    Some fused attention backward kernels rely on the saved output's layout.
    Allocating pinned buffers from tensor.size() alone makes them contiguous
    and can corrupt those gradients. empty_like preserves the layout, matching
    the normal tensor.cpu() path while allowing pinned host storage.
    """

    def pack(tensor):
        device = tensor.device
        if device.type == "cpu":
            return device, tensor.detach()
        if pin_memory and device.type == "cuda" and tensor.layout == torch.strided:
            packed = torch.empty_like(
                tensor, device="cpu", pin_memory=True, memory_format=torch.preserve_format
            )
            packed.copy_(tensor)
        else:
            packed = tensor.detach().cpu()
        return device, packed

    def unpack(record):
        device, tensor = record
        pinned = tensor.layout == torch.strided and tensor.is_pinned()
        return tensor.to(device, non_blocking=pinned)

    return torch.autograd.graph.saved_tensors_hooks(pack, unpack)


def transformer_blocks(model: nn.Module) -> list[nn.Module]:
    """Find common Hugging Face block stacks; pass blocks explicitly for custom models."""
    for path in (
        "transformer.h",
        "model.layers",
        "gpt_neox.layers",
        "bert.encoder.layer",
        "roberta.encoder.layer",
        "model.decoder.layers",
        "transformer.blocks",
    ):
        try:
            blocks = model.get_submodule(path)
        except AttributeError:
            continue
        if isinstance(blocks, (nn.ModuleList, nn.Sequential)):
            return list(blocks)
    raise ValueError("Cannot locate transformer blocks; pass your block sequence explicitly.")


def block_target_names(
    model: nn.Module, blocks: Sequence[nn.Module], predicate: Callable[[int], bool], **selection
) -> list[str]:
    """Select linear inputs belonging to the requested block indices."""
    from .surgery import select_target_names

    members = {id(m) for i, block in enumerate(blocks) if predicate(i) for m in block.modules()}
    return [
        name
        for name in select_target_names(model, **selection)
        if id(model.get_submodule(name)) in members
    ]


def checkpoint_blocks(blocks: Sequence[nn.Module], indices: Sequence[int]) -> list[int]:
    """Apply non-reentrant checkpointing to specific blocks, preserving RNG state."""
    from .adaptive_checkpointing import checkpoint

    selected = _validate_indices(blocks, indices)
    for i in selected:
        block = blocks[i]
        if hasattr(block, "_stacr_checkpoint_forward"):
            raise ValueError(f"Block {i} already has checkpointing enabled")
        original = block.forward

        @wraps(original)
        def forward(*args, __original=original, **kwargs):
            if torch.is_grad_enabled():
                return checkpoint(
                    __original, *args, use_reentrant=False, preserve_rng_state=True, **kwargs
                )
            return __original(*args, **kwargs)

        block._stacr_checkpoint_forward = original
        block.forward = forward
    return selected


def offload_blocks(
    blocks: Sequence[nn.Module], indices: Sequence[int], *, pin_memory: bool = True
) -> list[int]:
    """Move saved tensors of selected blocks to CPU until backward needs them."""
    selected = _validate_indices(blocks, indices)
    for i in selected:
        block = blocks[i]
        if hasattr(block, "_stacr_offload_forward"):
            raise ValueError(f"Block {i} already has CPU offload enabled")
        original = block.forward

        @wraps(original)
        def forward(*args, __original=original, **kwargs):
            with saved_tensors_on_cpu(pin_memory=pin_memory):
                return __original(*args, **kwargs)

        block._stacr_offload_forward = original
        block.forward = forward
    return selected


def _validate_indices(blocks, indices) -> list[int]:
    selected = sorted(set(indices))
    if any(
        not isinstance(i, int) or isinstance(i, bool) or i < 0 or i >= len(blocks) for i in selected
    ):
        raise ValueError(f"Block indices must be integers in [0, {len(blocks)})")
    return selected
