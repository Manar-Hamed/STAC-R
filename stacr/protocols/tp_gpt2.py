"""
GPT-2-specific Tensor Parallelism via module replacement.

Replaces GPT-2 MLP (and later Attention) modules with TP-aware versions
that place all-reduce communication at the correct points:

    Column-parallel (c_fc, c_attn):
        CopyToTPRegion (identity fwd, all-reduce bwd) -> sharded Linear

    Row-parallel (c_proj):
        sharded Linear -> ReduceFromTPRegion (all-reduce fwd, identity bwd) -> dropout

The key correctness property: dropout happens AFTER all-reduce on the
fully-reduced (replicated) tensor. HF GPT-2 applies dropout INSIDE the
original MLP/Attention forward, so monkey-patching would put all-reduce
after dropout -- wrong. Module replacement is the structurally correct fix.

scope="mlp_only": TPGPT2MLP only. Attention stays original (replicated).
scope="full":     TPGPT2MLP + TPGPT2Attention with interleaved QKV head sharding.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from stacr.protocols.tensor_parallel import (
    copy_to_tp_region,
    reduce_from_tp_region,
    convert_conv1d_to_linear,
    shard_linear_column,
    shard_linear_row,
)


def shard_gpt2_qkv_column(linear, tp_rank, tp_world_size, num_heads, head_dim):
    """Shard the fused c_attn [Q_all|K_all|V_all] weight with correct
    per-head interleaved extraction.

    GPT-2's c_attn projects input to Q, K, V concatenated:
        weight shape: (3 * embed_dim, in_features) after Conv1D->Linear
        layout: [Q_head0 Q_head1 ... | K_head0 K_head1 ... | V_head0 V_head1 ...]

    CRITICAL: Naive contiguous slicing (weight[:3*embed_dim//tp, :]) gives
    rank 0 all of Q + part of K -- WRONG. Must extract per-head slices from
    each of Q, K, V independently.

    Args:
        linear: nn.Linear with out_features = 3 * embed_dim
        tp_rank: This rank's index
        tp_world_size: Number of TP ranks
        num_heads: Total number of attention heads
        head_dim: Dimension per head

    Returns:
        New nn.Linear with out_features = 3 * (num_heads // tp_world_size) * head_dim
    """
    embed_dim = num_heads * head_dim
    assert num_heads % tp_world_size == 0, (
        f"QKV shard: num_heads={num_heads} not divisible by tp_world_size={tp_world_size}"
    )
    heads_per_rank = num_heads // tp_world_size
    head_start = tp_rank * heads_per_rank * head_dim
    head_end = (tp_rank + 1) * heads_per_rank * head_dim
    w = linear.weight.data
    q_shard = w[head_start:head_end, :]
    k_shard = w[embed_dim + head_start : embed_dim + head_end, :]
    v_shard = w[2 * embed_dim + head_start : 2 * embed_dim + head_end, :]
    shard_w = torch.cat([q_shard, k_shard, v_shard], dim=0).contiguous()
    out_local = 3 * heads_per_rank * head_dim
    new = nn.Linear(
        linear.in_features, out_local, bias=linear.bias is not None, device=w.device, dtype=w.dtype
    )
    new.weight.data = shard_w
    if linear.bias is not None:
        b = linear.bias.data
        new.bias.data = torch.cat(
            [
                b[head_start:head_end],
                b[embed_dim + head_start : embed_dim + head_end],
                b[2 * embed_dim + head_start : 2 * embed_dim + head_end],
            ]
        ).contiguous()
    return new


class TPGPT2MLP(nn.Module):
    """Drop-in replacement for GPT2MLP with Megatron-style TP communication.

    Forward path:
        CopyToTPRegion(hidden_states)   # identity fwd, all-reduce bwd
        -> c_fc (column-parallel)         # local: (B,S,D) -> (B,S,4D/tp)
        -> activation (GELU)
        -> c_proj (row-parallel)          # local: (B,S,4D/tp) -> (B,S,D) partial
        -> ReduceFromTPRegion             # all-reduce partial -> replicated
        -> dropout                        # on replicated tensor (CORRECT)

    The original GPT2MLP applies dropout INSIDE forward after c_proj.
    If we wrapped the whole module, all-reduce would happen AFTER dropout
    on partial outputs - each rank would mask differently -> corrupted sum.
    Module replacement puts all-reduce BETWEEN c_proj and dropout.
    """

    def __init__(self, original_mlp, tp_rank, tp_world_size, tp_group):
        """Initialize from an existing GPT2MLP module.

        Args:
            original_mlp: The original GPT2MLP module (after Conv1D->Linear)
            tp_rank: This rank's index in the TP group
            tp_world_size: Number of ranks in the TP group
            tp_group: The distributed process group for TP
        """
        super().__init__()
        self.c_fc = shard_linear_column(original_mlp.c_fc, tp_rank, tp_world_size)
        self.c_proj = shard_linear_row(original_mlp.c_proj, tp_rank, tp_world_size)
        self.act = original_mlp.act
        self.dropout = original_mlp.dropout
        self.tp_group = tp_group

    def forward(self, hidden_states):
        x = copy_to_tp_region(hidden_states, self.tp_group)
        x = self.c_fc(x)
        x = self.act(x)
        x = self.c_proj(x)
        x = reduce_from_tp_region(x, self.tp_group)
        x = self.dropout(x)
        return x


class TPGPT2Attention(nn.Module):
    """Drop-in replacement for GPT2Attention with Megatron-style TP communication.

    Forward path:
        CopyToTPRegion(hidden_states)      # identity fwd, all-reduce bwd
        -> c_attn (column-parallel QKV)     # sharded: 3*(heads/tp)*head_dim output
        -> split into local Q, K, V heads
        -> scaled dot-product attention     # on local heads only
        -> attn_dropout                     # on local weights (OK before reduce)
        -> merge heads
        -> c_proj (row-parallel)            # partial output
        -> ReduceFromTPRegion               # all-reduce partial -> replicated
        -> resid_dropout                    # on replicated tensor (CORRECT)

    The original GPT2Attention applies resid_dropout INSIDE forward after c_proj.
    Module replacement puts all-reduce BETWEEN c_proj and resid_dropout.

    Compatible with both HF transformers v4.x (self.bias causal buffer) and
    v5.x+ (causal mask passed via attention_mask argument).

    Return structure: (attn_output, attn_weights_or_none) -- 2-tuple that
    satisfies both v4.x GPT2Block (indexes [0] and [1:]) and v5.x GPT2Block
    (unpacks with attn_output, _ = self.attn(...)).
    """

    def __init__(self, original_attn, tp_rank, tp_world_size, tp_group):
        """Initialize from an existing GPT2Attention module.

        Args:
            original_attn: The original GPT2Attention (after Conv1D->Linear conversion)
            tp_rank: This rank's index in the TP group
            tp_world_size: Number of ranks in the TP group
            tp_group: The distributed process group for TP
        """
        super().__init__()
        assert not getattr(original_attn, "is_cross_attention", False), (
            "TPGPT2Attention does not support cross-attention"
        )
        assert not getattr(original_attn, "reorder_and_upcast_attn", False), (
            "TPGPT2Attention does not support reorder_and_upcast_attn"
        )
        assert len(getattr(original_attn, "pruned_heads", set())) == 0, (
            "TPGPT2Attention does not support pruned heads"
        )
        num_heads = original_attn.num_heads
        head_dim = original_attn.head_dim
        assert num_heads % tp_world_size == 0, (
            f"num_heads={num_heads} not divisible by tp_world_size={tp_world_size}"
        )
        self.c_attn = shard_gpt2_qkv_column(
            original_attn.c_attn, tp_rank, tp_world_size, num_heads, head_dim
        )
        self.c_proj = shard_linear_row(original_attn.c_proj, tp_rank, tp_world_size)
        self.num_heads = num_heads // tp_world_size
        self.head_dim = head_dim
        self.split_size = self.num_heads * head_dim
        self.embed_dim = num_heads * head_dim
        self.tp_rank = tp_rank
        self.tp_world_size = tp_world_size
        if hasattr(original_attn, "scaling"):
            self.scaling = original_attn.scaling
        else:
            self.scaling = 1.0
            if getattr(original_attn, "scale_attn_weights", True):
                self.scaling = head_dim ** (-0.5)
            if getattr(original_attn, "scale_attn_by_inverse_layer_idx", False):
                layer_idx = getattr(original_attn, "layer_idx", None)
                if layer_idx is not None:
                    self.scaling /= float(layer_idx + 1)
        self.attn_dropout = original_attn.attn_dropout
        self.resid_dropout = original_attn.resid_dropout
        if hasattr(original_attn, "bias") and original_attn.bias is not None:
            self.register_buffer("bias", original_attn.bias)
        else:
            max_positions = getattr(
                getattr(original_attn, "config", None), "max_position_embeddings", 1024
            )
            self.register_buffer(
                "bias",
                torch.tril(
                    torch.ones(
                        (max_positions, max_positions),
                        dtype=torch.bool,
                        device=original_attn.c_attn.weight.device,
                    )
                ).view(1, 1, max_positions, max_positions),
                persistent=False,
            )
        self._legacy_api = hasattr(original_attn, "bias") and original_attn.bias is not None
        self.is_causal = getattr(original_attn, "is_causal", True)
        self.layer_idx = getattr(original_attn, "layer_idx", None)
        self.tp_group = tp_group

    def _split_heads(self, tensor):
        """(B, S, local_heads * head_dim) -> (B, local_heads, S, head_dim)"""
        new_shape = tensor.size()[:-1] + (self.num_heads, self.head_dim)
        tensor = tensor.view(new_shape)
        return tensor.permute(0, 2, 1, 3)

    def _merge_heads(self, tensor):
        """(B, local_heads, S, head_dim) -> (B, S, local_heads * head_dim)"""
        tensor = tensor.permute(0, 2, 1, 3).contiguous()
        new_shape = tensor.size()[:-2] + (self.num_heads * self.head_dim,)
        return tensor.view(new_shape)

    def forward(
        self,
        hidden_states,
        layer_past=None,
        attention_mask=None,
        head_mask=None,
        use_cache=False,
        output_attentions=False,
        past_key_values=None,
        **kwargs,
    ):
        kv_cache = past_key_values if past_key_values is not None else layer_past
        hidden_states = copy_to_tp_region(hidden_states, self.tp_group)
        qkv = self.c_attn(hidden_states)
        (query, key, value) = qkv.split(self.split_size, dim=2)
        query = self._split_heads(query)
        key = self._split_heads(key)
        value = self._split_heads(value)
        if kv_cache is not None:
            if isinstance(kv_cache, tuple):
                (past_key, past_value) = kv_cache
                key = torch.cat((past_key, key), dim=-2)
                value = torch.cat((past_value, value), dim=-2)
        attn_weights = torch.matmul(query, key.transpose(-1, -2)) * self.scaling
        query_length = query.size(-2)
        key_length = key.size(-2)
        causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
        mask_value = torch.finfo(attn_weights.dtype).min
        attn_weights = torch.where(
            causal_mask, attn_weights, torch.full_like(attn_weights, mask_value)
        )
        if attention_mask is not None:
            if attention_mask.dtype == torch.bool:
                mask_value = torch.finfo(attn_weights.dtype).min
                attn_weights = attn_weights.masked_fill(~attention_mask, mask_value)
            else:
                attn_weights = attn_weights + attention_mask
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32)
        attn_weights = attn_weights.to(query.dtype)
        attn_weights = self.attn_dropout(attn_weights)
        if head_mask is not None:
            global_heads = self.num_heads * self.tp_world_size
            head_start = self.tp_rank * self.num_heads
            head_end = head_start + self.num_heads
            if head_mask.dim() >= 2 and head_mask.size(1) == global_heads:
                head_mask = head_mask[:, head_start:head_end, ...]
            elif head_mask.dim() == 1 and head_mask.numel() == global_heads:
                head_mask = head_mask[head_start:head_end]
            attn_weights = attn_weights * head_mask
        attn_output = torch.matmul(attn_weights, value)
        attn_output = self._merge_heads(attn_output)
        attn_output = self.c_proj(attn_output)
        attn_output = reduce_from_tp_region(attn_output, self.tp_group)
        attn_output = self.resid_dropout(attn_output)
        if self._legacy_api:
            present = (key, value) if use_cache else None
            outputs = (attn_output, present)
            if output_attentions:
                outputs += (attn_weights,)
            return outputs
        return (attn_output, attn_weights if output_attentions else None)


def apply_tensor_parallel_gpt2(model, tp_rank, tp_world_size, tp_group, scope="mlp_only"):
    """Apply Tensor Parallelism to a GPT-2 model via module replacement.

    Steps:
    1. Convert all Conv1D -> nn.Linear (TP-path only, does not affect
       single/DP/DDP code paths)
    2. Replace each block's MLP with TPGPT2MLP (shards + correct dropout)
    3. If scope="full", replace Attention with TPGPT2Attention
    4. Disable embedding dropout for TP correctness (replicated path,
       no synchronized dropout mechanism yet)
    5. Embeddings and lm_head stay replicated

    Args:
        model: GPT2LMHeadModel (from HuggingFace transformers)
        tp_rank: This rank's index in the TP group
        tp_world_size: Number of ranks in the TP group
        tp_group: The distributed process group (None = default group)
        scope: "mlp_only" (MLP sharding only) or "full" (MLP + Attention)

    Returns:
        The modified model (in-place modification + return)
    """
    if scope not in ("mlp_only", "full"):
        raise ValueError(f"Unknown TP scope: {scope!r}. Use 'mlp_only' or 'full'.")
    num_heads = model.config.n_head
    if num_heads % tp_world_size != 0:
        raise ValueError(
            f"num_heads={num_heads} is not divisible by tp_world_size={tp_world_size}. GPT-2 TP requires num_heads % tp_world_size == 0. For this model, valid TP degrees are: {[d for d in range(1, num_heads + 1) if num_heads % d == 0]}"
        )
    n_converted = convert_conv1d_to_linear(model)
    if tp_rank == 0:
        print(f"[TP] Converted {n_converted} Conv1D -> nn.Linear")
    n_blocks = len(model.transformer.h)
    for i, block in enumerate(model.transformer.h):
        block.mlp = TPGPT2MLP(block.mlp, tp_rank, tp_world_size, tp_group)
        if scope == "full":
            block.attn = TPGPT2Attention(block.attn, tp_rank, tp_world_size, tp_group)
    model.transformer.drop.p = 0.0
    if tp_rank == 0:
        print(f"[TP] Replaced MLP in {n_blocks} blocks (scope={scope})")
        if scope == "mlp_only":
            print(f"[TP] Attention modules are REPLICATED (Batch A mode)")
        elif scope == "full":
            print(f"[TP] Attention modules are SHARDED (Batch B mode)")
        param_count = sum((p.numel() for p in model.parameters()))
        print(
            f"[TP] Per-rank parameters: {param_count:,} ({param_count * 4 / 1000000000.0:.2f} GB at FP32)"
        )
    return model


def get_tp_stacr_targets(model, tp_stacr_scope="row_only"):
    """Filter compression targets for TP-compatible STAC-R scope.

    In TP, column-parallel layers (c_fc, c_attn) see replicated input — STAC-R
    works identically to single-GPU. Row-parallel layers (c_proj in MLP and
    attn) see sharded input — STAC-R adapts automatically (smaller basis).

    Scopes:
        row_only  — MLP c_proj only (most conservative)
        all_mlp   — MLP c_fc + c_proj
        all_linear — all 4 linear layers per block: mlp.c_fc, mlp.c_proj,
                     attn.c_attn, attn.c_proj

    Args:
        model: The TP-sharded model
        tp_stacr_scope: One of "row_only", "all_mlp", "all_linear"

    Returns:
        List of module name strings to pass to replace_linear_with_stacr
    """
    VALID_SCOPES = ("row_only", "all_mlp", "all_linear")
    if tp_stacr_scope not in VALID_SCOPES:
        raise ValueError(f"Unknown tp_stacr_scope: {tp_stacr_scope}. Must be one of {VALID_SCOPES}")
    if tp_stacr_scope == "row_only":
        patterns = ("mlp.c_proj",)
    elif tp_stacr_scope == "all_mlp":
        patterns = ("mlp.c_fc", "mlp.c_proj")
    else:
        patterns = ("mlp.c_fc", "mlp.c_proj", "attn.c_attn", "attn.c_proj")
    targets = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if any((p in name for p in patterns)):
            targets.append(name)
    return targets


def disable_tp_dropout(model):
    """Set p=0.0 on all Dropout modules in the model.

    Disabling dropout prevents mask divergence across ranks. Use
    synced_dropout to retain dropout with synchronized masks.

    Args:
        model: The TP-sharded model

    Returns:
        Number of Dropout modules disabled
    """
    count = 0
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            if module.p > 0.0:
                module.p = 0.0
                count += 1
    return count


class TPActivationProfiler:
    """Profile linear output bytes by layer family in a TP-sharded GPT-2 model.

    Registers forward hooks on Linear layers, records output tensor sizes for
    one forward pass, then removes hooks and produces a per-family summary.

    Families: mlp.c_fc, mlp.c_proj, attn.c_attn, attn.c_proj, other_linear

    Important: this is not an exact saved-tensor / autograd-memory profiler.
    It measures Linear output tensor sizes only. That makes it useful for
    comparing which projection families dominate the linear activation budget,
    but it does not include non-linear attention intermediates such as QK^T.

    Usage:
        profiler = TPActivationProfiler(model)
        with profiler:
            outputs = model(**batch)
        summary = profiler.summary()
    """

    FAMILIES = ("mlp.c_fc", "mlp.c_proj", "attn.c_attn", "attn.c_proj")

    def __init__(self, model):
        self.model = model
        self._hooks = []
        self._records = {}

    def _classify(self, name):
        """Classify a module name into a family."""
        for fam in self.FAMILIES:
            if fam in name:
                return fam
        return "other_linear"

    def _make_hook(self, name):
        family = self._classify(name)

        def hook(module, input, output):
            if isinstance(output, torch.Tensor):
                nbytes = output.nelement() * output.element_size()
                self._records[name] = {
                    "family": family,
                    "bytes": nbytes,
                    "shape": list(output.shape),
                    "dtype": str(output.dtype),
                }

        return hook

    def __enter__(self):
        self._records.clear()
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear) or module.__class__.__name__ == "STACRLinear":
                h = module.register_forward_hook(self._make_hook(name))
                self._hooks.append(h)
        return self

    def __exit__(self, *args):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def summary(self):
        """Return a dict with per-family profiled byte totals and details."""
        family_totals = {}
        family_details = {}
        for name, rec in sorted(self._records.items()):
            fam = rec["family"]
            family_totals[fam] = family_totals.get(fam, 0) + rec["bytes"]
            family_details.setdefault(fam, []).append(
                {"name": name, "shape": rec["shape"], "dtype": rec["dtype"], "bytes": rec["bytes"]}
            )
        total = sum(family_totals.values())
        return {
            "total_profiled_bytes": total,
            "by_family": family_totals,
            "details": family_details,
            "notes": "Measures Linear output tensor sizes only; does not include non-linear attention intermediates or exact autograd saved-tensor bytes.",
        }

    def print_summary(self, rank=0):
        """Pretty-print the profiled linear-output breakdown."""
        s = self.summary()
        total = s["total_profiled_bytes"]
        print(f"\n{'=' * 60}")
        print(f" TP Linear Output Breakdown (rank {rank})")
        print(f"{'=' * 60}")
        print(f" {'Family':<20} {'Bytes':>12} {'MB':>8} {'%':>6}")
        print(f" {'-' * 20} {'-' * 12} {'-' * 8} {'-' * 6}")
        for fam in list(self.FAMILIES) + ["other_linear"]:
            b = s["by_family"].get(fam, 0)
            pct = 100 * b / total if total > 0 else 0
            print(f" {fam:<20} {b:>12,} {b / 1000000.0:>8.1f} {pct:>5.1f}%")
        print(f" {'-' * 20} {'-' * 12} {'-' * 8} {'-' * 6}")
        print(f" {'TOTAL':<20} {total:>12,} {total / 1000000.0:>8.1f} 100.0%")
        print(f"{'=' * 60}\n")
