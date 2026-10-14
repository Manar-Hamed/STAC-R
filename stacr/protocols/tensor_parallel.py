"""
Tensor Parallelism communication primitives and weight-sharding utilities.

Implements Megatron-style manual TP for GPT-2:
- Communication autograd functions (CopyToTPRegion, ReduceFromTPRegion)
- Conv1D -> nn.Linear conversion (TP-path only)
- Column-parallel and row-parallel weight sharding
- Replicated parameter gradient synchronization

Design decisions:
- clone() before all_reduce in ReduceFromTPRegion (avoids in-place aliasing)
- Conv1D conversion scoped to TP path only (no impact on single/DP/DDP)
- Row-parallel bias: only rank 0 has a bias parameter (avoids double-counting
  and avoids non-zero-rank bias drift during training)
- Replicated grads (embeddings, LayerNorm) are all-reduced + averaged

Reference: Megatron-LM (Shoeybi et al., 2019), Section 3.
"""

import torch
import torch.nn as nn
import torch.distributed as dist


class _CopyToTPRegion(torch.autograd.Function):
    """Forward: identity (input already replicated across ranks).
    Backward: all-reduce gradients so each rank gets the full gradient.

    Used before column-parallel layers (c_fc, c_attn) where the input is
    replicated and each rank computes a partial output.
    """

    @staticmethod
    def forward(ctx, input_, tp_group):
        ctx.tp_group = tp_group
        return input_

    @staticmethod
    def backward(ctx, grad_output):
        dist.all_reduce(grad_output, group=ctx.tp_group)
        return (grad_output, None)


class _ReduceFromTPRegion(torch.autograd.Function):
    """Forward: all-reduce partial outputs from row-parallel layers.
    Backward: identity (gradient is already correct per rank).

    Used after row-parallel layers (c_proj) where each rank has a partial
    result that must be summed to get the full output.

    IMPORTANT: Uses clone() before all_reduce to avoid in-place mutation
    of a tensor that may be aliased or needed for backward. (V2-54 fix)
    """

    @staticmethod
    def forward(ctx, input_, tp_group):
        ctx.tp_group = tp_group
        output = input_.clone()
        dist.all_reduce(output, group=ctx.tp_group)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        return (grad_output, None)


def copy_to_tp_region(input_, tp_group):
    """Convenience wrapper: identity forward, all-reduce backward."""
    return _CopyToTPRegion.apply(input_, tp_group)


def reduce_from_tp_region(input_, tp_group):
    """Convenience wrapper: all-reduce forward, identity backward."""
    return _ReduceFromTPRegion.apply(input_, tp_group)


def convert_conv1d_to_linear(model):
    """Replace all HuggingFace Conv1D modules with nn.Linear.

    HF GPT-2 uses Conv1D with weight shape (in_features, out_features),
    which is the TRANSPOSE of nn.Linear's (out_features, in_features).
    TP sharding code expects nn.Linear layout, so we convert first.

    This conversion is ONLY called on the TP code path. Single-GPU, DP,
    and DDP paths are unaffected.

    Args:
        model: GPT-2 model to convert

    Returns:
        Number of modules converted
    """
    converted = 0
    module_dict = dict(model.named_modules())
    for name, module in list(module_dict.items()):
        if module.__class__.__name__ != "Conv1D":
            continue
        (in_features, out_features) = module.weight.shape
        has_bias = module.bias is not None
        new_linear = nn.Linear(
            in_features,
            out_features,
            bias=has_bias,
            device=module.weight.device,
            dtype=module.weight.dtype,
        )
        new_linear.weight.data = module.weight.data.t().contiguous()
        if has_bias:
            new_linear.bias.data = module.bias.data.clone()
        if "." in name:
            (parent_name, attr) = name.rsplit(".", 1)
            setattr(module_dict[parent_name], attr, new_linear)
        else:
            setattr(model, name, new_linear)
        converted += 1
    return converted


def shard_linear_column(linear, tp_rank, tp_world_size):
    """Shard a Linear layer column-parallel: split the OUTPUT dimension.

    Each rank gets out_features/tp_world_size output neurons.
    Bias is also sharded along the output dimension.

    Used for: c_fc (up projection), c_attn (QKV projection)

    Args:
        linear: nn.Linear to shard
        tp_rank: This rank's index in the TP group
        tp_world_size: Number of ranks in the TP group

    Returns:
        New nn.Linear with sharded weights
    """
    assert linear.out_features % tp_world_size == 0, (
        f"Column shard: out_features={linear.out_features} not divisible by tp_world_size={tp_world_size}"
    )
    shard_size = linear.out_features // tp_world_size
    start = tp_rank * shard_size
    end = (tp_rank + 1) * shard_size
    new = nn.Linear(
        linear.in_features,
        shard_size,
        bias=linear.bias is not None,
        device=linear.weight.device,
        dtype=linear.weight.dtype,
    )
    new.weight.data = linear.weight.data[start:end, :].contiguous()
    if linear.bias is not None:
        new.bias.data = linear.bias.data[start:end].contiguous()
    return new


def shard_linear_row(linear, tp_rank, tp_world_size):
    """Shard a Linear layer row-parallel: split the INPUT dimension.

    Each rank gets in_features/tp_world_size input neurons.
    Bias handling: only rank 0 gets a bias parameter. Zeroing the bias value on
    non-zero ranks is NOT sufficient, because those bias parameters would still
    receive gradients and start drifting after the first optimizer step.

    Used for: c_proj (down projection in both MLP and attention)

    Args:
        linear: nn.Linear to shard
        tp_rank: This rank's index in the TP group
        tp_world_size: Number of ranks in the TP group

    Returns:
        New nn.Linear with sharded weights
    """
    assert linear.in_features % tp_world_size == 0, (
        f"Row shard: in_features={linear.in_features} not divisible by tp_world_size={tp_world_size}"
    )
    shard_size = linear.in_features // tp_world_size
    start = tp_rank * shard_size
    end = (tp_rank + 1) * shard_size
    keep_bias = linear.bias is not None and tp_rank == 0
    new = nn.Linear(
        shard_size,
        linear.out_features,
        bias=keep_bias,
        device=linear.weight.device,
        dtype=linear.weight.dtype,
    )
    new.weight.data = linear.weight.data[:, start:end].contiguous()
    if keep_bias:
        new.bias.data = linear.bias.data.clone()
    return new


def sync_replicated_grads(model, tp_group):
    """All-reduce + average gradients for non-sharded (replicated) parameters.

    In TP, embeddings and LayerNorm are replicated on all ranks. Each rank
    computes gradients from the same forward pass but different backward
    paths (through different column/row shards). We all-reduce these
    gradients so all ranks take the same optimizer step.

    Sharded parameters (c_attn, c_fc, c_proj) are NOT synced - each rank
    updates only its own shard.

    Must be called after loss.backward() and before optimizer.step().

    Args:
        model: The TP-sharded model
        tp_group: The TP process group (None = default group)
    """
    ws = dist.get_world_size(tp_group)
    sharded_prefixes = set()
    for name, module in model.named_modules():
        if module.__class__.__name__ in ("TPGPT2MLP", "TPGPT2Attention"):
            sharded_prefixes.add(name + ".")
    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        if any((name.startswith(prefix) for prefix in sharded_prefixes)):
            continue
        dist.all_reduce(param.grad, group=tp_group)
        param.grad.div_(ws)
