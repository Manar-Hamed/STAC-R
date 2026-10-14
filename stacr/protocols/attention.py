"""STAC-R attention compression utilities.

Wraps Q/K/V projection layers with STACRLinear while preserving standard
attention behavior. Designed for HuggingFace BERT and GPT-2 (optional c_attn split).
"""

from __future__ import annotations
from contextlib import contextmanager
import copy
from dataclasses import dataclass
import inspect
import threading
from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
from stacr.adaptive import STACRConfig, STACRLinear

_MASK_CONTEXT = threading.local()


def _get_current_attention_mask() -> Optional[torch.Tensor]:
    return getattr(_MASK_CONTEXT, "mask", None)


@contextmanager
def _attention_mask_context(mask: Optional[torch.Tensor]):
    prev = getattr(_MASK_CONTEXT, "mask", None)
    _MASK_CONTEXT.mask = mask
    try:
        yield
    finally:
        _MASK_CONTEXT.mask = prev


def _extract_token_mask(
    attention_mask: torch.Tensor, seq_len: int, device: torch.device
) -> Optional[torch.Tensor]:
    if attention_mask is None:
        return None
    mask = attention_mask
    if mask.dim() not in (2, 3, 4) or mask.size(-1) != seq_len:
        return None
    if mask.dtype.is_floating_point:
        # HF uses binary masks in 2-D and additive masks after causal expansion.
        token_mask = mask >= 0 if mask.dim() > 2 or (mask < 0).any() else mask != 0
    else:
        token_mask = mask != 0
    if mask.dim() > 2:
        # A key is valid if any query may attend to it. Reading only query zero
        # would incorrectly discard all later tokens in a causal mask.
        token_mask = token_mask.reshape(mask.shape[0], -1, seq_len).any(dim=1)
    return token_mask.to(device=device)


def _resolve_mask_arg_index(fn) -> Optional[int]:
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return None
    params = list(sig.parameters.keys())
    mask_idx = None
    for idx, name in enumerate(params):
        if name in ("attention_mask", "attn_mask"):
            mask_idx = idx
            break
    if mask_idx is None:
        return None
    if params and params[0] == "self":
        mask_idx -= 1
    return mask_idx if mask_idx >= 0 else None


def _wrap_attention_forward(module: nn.Module) -> None:
    if getattr(module, "_stacr_mask_wrapped", False):
        return
    original_forward = module.forward
    mask_arg_index = _resolve_mask_arg_index(original_forward)

    def wrapped_forward(*args, **kwargs):
        mask = kwargs.get("attention_mask")
        if mask is None:
            mask = kwargs.get("attn_mask")
        if mask is None and mask_arg_index is not None and (len(args) > mask_arg_index):
            mask = args[mask_arg_index]
        with _attention_mask_context(mask):
            return original_forward(*args, **kwargs)

    module.forward = wrapped_forward
    module._stacr_mask_wrapped = True


@dataclass
class AttentionCompressionReport:
    """Summary of attention compression replacements."""

    replaced_qkv: int
    skipped: int
    warnings: List[str]
    joint_qk_pairs: int = 0


def _is_conv1d_module(module: nn.Module) -> bool:
    """Detect HuggingFace Conv1D (GPT-2 style) without importing transformers."""
    return module.__class__.__name__ == "Conv1D"


def _copy_linear_weights(old_module: nn.Module, new_stacr: STACRLinear) -> None:
    """Copy weights/bias from nn.Linear or HF Conv1D into STACRLinear.

    Handles device and dtype mismatches by converting source to destination format.
    """
    target_device = new_stacr.linear.weight.device
    target_dtype = new_stacr.linear.weight.dtype
    if isinstance(old_module, nn.Linear):
        new_stacr.linear.weight.data.copy_(
            old_module.weight.data.to(device=target_device, dtype=target_dtype)
        )
        if old_module.bias is not None and new_stacr.linear.bias is not None:
            new_stacr.linear.bias.data.copy_(
                old_module.bias.data.to(device=target_device, dtype=target_dtype)
            )
        new_stacr.linear.weight.requires_grad = old_module.weight.requires_grad
        if old_module.bias is not None and new_stacr.linear.bias is not None:
            new_stacr.linear.bias.requires_grad = old_module.bias.requires_grad
        return
    if _is_conv1d_module(old_module):
        weight = old_module.weight.data
        new_stacr.linear.weight.data.copy_(weight.t().to(device=target_device, dtype=target_dtype))
        bias = getattr(old_module, "bias", None)
        if bias is not None and new_stacr.linear.bias is not None:
            new_stacr.linear.bias.data.copy_(bias.data.to(device=target_device, dtype=target_dtype))
        new_stacr.linear.weight.requires_grad = old_module.weight.requires_grad
        if bias is not None and new_stacr.linear.bias is not None:
            new_stacr.linear.bias.requires_grad = bias.requires_grad
        return
    raise TypeError(f"Unsupported module type for STACR attention: {type(old_module)}")


class _MaskAwareSTACRLinear(nn.Module):
    """Mask-aware wrapper that skips padded tokens during compression."""

    def __init__(self, inner: STACRLinear):
        super().__init__()
        self.inner = inner

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mask = _get_current_attention_mask()
        if mask is None or x.dim() != 3:
            return self.inner(x)
        token_mask = _extract_token_mask(mask, x.shape[1], x.device)
        if token_mask is None:
            return self.inner(x)
        mask_flat = token_mask.reshape(-1)
        if mask_flat.all():
            return self.inner(x)
        x_flat = x.reshape(-1, x.shape[-1])
        valid_idx = mask_flat.nonzero(as_tuple=False).reshape(-1)
        if valid_idx.numel() == 0:
            out_shape = (x.shape[0], x.shape[1], self.inner.linear.out_features)
            return x.new_zeros(out_shape)
        valid_x = x_flat.index_select(0, valid_idx)
        valid_out = self.inner(valid_x)
        out_flat = torch.zeros(
            (x_flat.shape[0], valid_out.shape[1]), dtype=valid_out.dtype, device=valid_out.device
        )
        out_flat.index_copy_(0, valid_idx, valid_out)
        return out_flat.view(x.shape[0], x.shape[1], -1)

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.inner, name)


class JointQKSTACRLinear(nn.Module):
    """Wrap Q and K projections with a shared activation basis.

    Storage savings:
        - Independent: 2 bases (N × R each) = 2NR params
        - Joint: 1 shared basis (N × R) = NR params
        This concerns basis storage only; coefficients and residuals remain separate.
    """

    def __init__(
        self,
        q_proj: nn.Module,
        k_proj: nn.Module,
        config: STACRConfig,
        *,
        mask_aware: bool = False,
        layer_index: Optional[int] = None,
    ):
        """Initialize joint Q/K compression.

        Args:
            q_proj: Query projection module (nn.Linear or Conv1D)
            k_proj: Key projection module (nn.Linear or Conv1D)
            config: STACR configuration for shared basis
            mask_aware: Whether to skip padded tokens during compression
        """
        super().__init__()
        if getattr(config, "rank_mode", "fixed") != "fixed":
            raise ValueError(
                "JointQKSTACRLinear requires rank_mode='fixed' because Q/K share one basis."
            )
        if isinstance(q_proj, nn.Linear):
            q_in_features = q_proj.in_features
            q_out_features = q_proj.out_features
        elif _is_conv1d_module(q_proj):
            (q_in_features, q_out_features) = q_proj.weight.shape
        else:
            raise TypeError(f"Unsupported q_proj type: {type(q_proj)}")
        if isinstance(k_proj, nn.Linear):
            k_in_features = k_proj.in_features
            k_out_features = k_proj.out_features
        elif _is_conv1d_module(k_proj):
            (k_in_features, k_out_features) = k_proj.weight.shape
        else:
            raise TypeError(f"Unsupported k_proj type: {type(k_proj)}")
        if q_in_features != k_in_features:
            raise ValueError(
                f"Q and K must have same in_features for joint compression. Got q_in={q_in_features}, k_in={k_in_features}"
            )
        device = q_proj.weight.device if hasattr(q_proj, "weight") else None
        dtype = q_proj.weight.dtype if hasattr(q_proj, "weight") else None
        shared_config = copy.deepcopy(config)
        shared_config.basis_sharing = "all"
        shared_config.basis_sharing_group = -id(self)
        self.q_stacr = STACRLinear(
            q_in_features,
            q_out_features,
            shared_config,
            bias=q_proj.bias is not None
            if isinstance(q_proj, nn.Linear)
            else getattr(q_proj, "bias", None) is not None,
            layer_index=layer_index,
        )
        self.k_stacr = STACRLinear(
            k_in_features,
            k_out_features,
            copy.deepcopy(shared_config),
            bias=k_proj.bias is not None
            if isinstance(k_proj, nn.Linear)
            else getattr(k_proj, "bias", None) is not None,
            layer_index=layer_index,
        )
        if device is not None or dtype is not None:
            self.q_stacr = self.q_stacr.to(device=device, dtype=dtype)
            self.k_stacr = self.k_stacr.to(device=device, dtype=dtype)
        _copy_linear_weights(q_proj, self.q_stacr)
        _copy_linear_weights(k_proj, self.k_stacr)
        self._mask_aware = mask_aware
        if mask_aware:
            self._q_wrapper = _MaskAwareSTACRLinear(self.q_stacr)
            self._k_wrapper = _MaskAwareSTACRLinear(self.k_stacr)
        else:
            self._q_wrapper = self.q_stacr
            self._k_wrapper = self.k_stacr
        self.train(q_proj.training and k_proj.training)

    @property
    def shared_basis(self):
        """Current immutable basis version shared through the pool."""
        return self.q_stacr.effective_basis

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with shared basis compression.

        Args:
            x: Input tensor (B, seq_len, hidden_dim)

        Returns:
            Tuple of (q_out, k_out)
        """
        q_out = self._q_wrapper(x)
        k_out = self._k_wrapper(x)
        return (q_out, k_out)

    def __getattr__(self, name: str):
        """Delegate attribute access to q_stacr for compatibility."""
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.q_stacr, name)


CompressedProjection = Union[STACRLinear, _MaskAwareSTACRLinear, JointQKSTACRLinear]


def _wrap_projection(
    module: nn.Module,
    config: STACRConfig,
    *,
    mask_aware: bool = False,
    layer_index: Optional[int] = None,
) -> CompressedProjection:
    """Create STACRLinear from an existing projection module.

    Args:
        module: Module to wrap (nn.Linear or Conv1D)
        config: STACR configuration
        mask_aware: Whether to skip padded tokens during compression
        layer_index: Layer index for basis sharing (required when basis_sharing != "none")
    """
    if isinstance(module, (STACRLinear, _MaskAwareSTACRLinear)):
        return module
    device = None
    dtype = None
    if hasattr(module, "weight"):
        device = module.weight.device
        dtype = module.weight.dtype
    if isinstance(module, nn.Linear):
        stacr = STACRLinear(
            module.in_features,
            module.out_features,
            config,
            bias=module.bias is not None,
            layer_index=layer_index,
        )
        if device is not None or dtype is not None:
            stacr = stacr.to(device=device, dtype=dtype)
        _copy_linear_weights(module, stacr)
        stacr.train(module.training)
        if mask_aware:
            return _MaskAwareSTACRLinear(stacr)
        return stacr
    if _is_conv1d_module(module):
        (in_features, out_features) = module.weight.shape
        stacr = STACRLinear(
            in_features,
            out_features,
            config,
            bias=getattr(module, "bias", None) is not None,
            layer_index=layer_index,
        )
        if device is not None or dtype is not None:
            stacr = stacr.to(device=device, dtype=dtype)
        _copy_linear_weights(module, stacr)
        stacr.train(module.training)
        if mask_aware:
            return _MaskAwareSTACRLinear(stacr)
        return stacr
    raise TypeError(f"Unsupported projection module type: {type(module)}")


class _SplitQKVWrapper(nn.Module):
    """Split GPT-2 fused c_attn into Q/K/V STACRLinears and re-concatenate."""

    def __init__(
        self,
        c_attn: nn.Module,
        config: STACRConfig,
        *,
        mask_aware: bool = False,
        layer_start_index: int = 0,
    ):
        super().__init__()
        target_device = c_attn.weight.device if hasattr(c_attn, "weight") else None
        target_dtype = c_attn.weight.dtype if hasattr(c_attn, "weight") else None
        if _is_conv1d_module(c_attn):
            (in_features, out_features_total) = c_attn.weight.shape
            if out_features_total % 3 != 0:
                raise ValueError(
                    f"c_attn out_features must be divisible by 3 for Q/K/V split, got {out_features_total}"
                )
            out_features = out_features_total // 3
            weight = c_attn.weight.data
            bias = getattr(c_attn, "bias", None)
            (q_w, k_w, v_w) = (
                weight[:, :out_features],
                weight[:, out_features : 2 * out_features],
                weight[:, 2 * out_features :],
            )
            q_b = bias[:out_features] if bias is not None else None
            k_b = bias[out_features : 2 * out_features] if bias is not None else None
            v_b = bias[2 * out_features :] if bias is not None else None
            q_proj = STACRLinear(
                in_features,
                out_features,
                config,
                bias=q_b is not None,
                layer_index=layer_start_index,
            )
            k_proj = STACRLinear(
                in_features,
                out_features,
                config,
                bias=k_b is not None,
                layer_index=layer_start_index + 1,
            )
            v_proj = STACRLinear(
                in_features,
                out_features,
                config,
                bias=v_b is not None,
                layer_index=layer_start_index + 2,
            )
            if target_device is not None or target_dtype is not None:
                q_proj = q_proj.to(device=target_device, dtype=target_dtype)
                k_proj = k_proj.to(device=target_device, dtype=target_dtype)
                v_proj = v_proj.to(device=target_device, dtype=target_dtype)
            q_proj.linear.weight.data.copy_(
                q_w.t().to(device=q_proj.linear.weight.device, dtype=q_proj.linear.weight.dtype)
            )
            k_proj.linear.weight.data.copy_(
                k_w.t().to(device=k_proj.linear.weight.device, dtype=k_proj.linear.weight.dtype)
            )
            v_proj.linear.weight.data.copy_(
                v_w.t().to(device=v_proj.linear.weight.device, dtype=v_proj.linear.weight.dtype)
            )
            if q_b is not None:
                q_proj.linear.bias.data.copy_(
                    q_b.to(device=q_proj.linear.bias.device, dtype=q_proj.linear.bias.dtype)
                )
            if k_b is not None:
                k_proj.linear.bias.data.copy_(
                    k_b.to(device=k_proj.linear.bias.device, dtype=k_proj.linear.bias.dtype)
                )
            if v_b is not None:
                v_proj.linear.bias.data.copy_(
                    v_b.to(device=v_proj.linear.bias.device, dtype=v_proj.linear.bias.dtype)
                )
            q_proj.linear.weight.requires_grad = c_attn.weight.requires_grad
            k_proj.linear.weight.requires_grad = c_attn.weight.requires_grad
            v_proj.linear.weight.requires_grad = c_attn.weight.requires_grad
            if bias is not None:
                if q_proj.linear.bias is not None:
                    q_proj.linear.bias.requires_grad = bias.requires_grad
                if k_proj.linear.bias is not None:
                    k_proj.linear.bias.requires_grad = bias.requires_grad
                if v_proj.linear.bias is not None:
                    v_proj.linear.bias.requires_grad = bias.requires_grad
            self.q_proj = _MaskAwareSTACRLinear(q_proj) if mask_aware else q_proj
            self.k_proj = _MaskAwareSTACRLinear(k_proj) if mask_aware else k_proj
            self.v_proj = _MaskAwareSTACRLinear(v_proj) if mask_aware else v_proj
        elif isinstance(c_attn, nn.Linear):
            in_features = c_attn.in_features
            if c_attn.out_features % 3 != 0:
                raise ValueError(
                    f"c_attn out_features must be divisible by 3 for Q/K/V split, got {c_attn.out_features}"
                )
            out_features = c_attn.out_features // 3
            weight = c_attn.weight.data
            bias = c_attn.bias.data if c_attn.bias is not None else None
            (q_w, k_w, v_w) = (
                weight[:out_features, :],
                weight[out_features : 2 * out_features, :],
                weight[2 * out_features :, :],
            )
            q_b = bias[:out_features] if bias is not None else None
            k_b = bias[out_features : 2 * out_features] if bias is not None else None
            v_b = bias[2 * out_features :] if bias is not None else None
            q_proj = STACRLinear(
                in_features,
                out_features,
                config,
                bias=q_b is not None,
                layer_index=layer_start_index,
            )
            k_proj = STACRLinear(
                in_features,
                out_features,
                config,
                bias=k_b is not None,
                layer_index=layer_start_index + 1,
            )
            v_proj = STACRLinear(
                in_features,
                out_features,
                config,
                bias=v_b is not None,
                layer_index=layer_start_index + 2,
            )
            if target_device is not None or target_dtype is not None:
                q_proj = q_proj.to(device=target_device, dtype=target_dtype)
                k_proj = k_proj.to(device=target_device, dtype=target_dtype)
                v_proj = v_proj.to(device=target_device, dtype=target_dtype)
            q_proj.linear.weight.data.copy_(
                q_w.to(device=q_proj.linear.weight.device, dtype=q_proj.linear.weight.dtype)
            )
            k_proj.linear.weight.data.copy_(
                k_w.to(device=k_proj.linear.weight.device, dtype=k_proj.linear.weight.dtype)
            )
            v_proj.linear.weight.data.copy_(
                v_w.to(device=v_proj.linear.weight.device, dtype=v_proj.linear.weight.dtype)
            )
            if q_b is not None:
                q_proj.linear.bias.data.copy_(
                    q_b.to(device=q_proj.linear.bias.device, dtype=q_proj.linear.bias.dtype)
                )
            if k_b is not None:
                k_proj.linear.bias.data.copy_(
                    k_b.to(device=k_proj.linear.bias.device, dtype=k_proj.linear.bias.dtype)
                )
            if v_b is not None:
                v_proj.linear.bias.data.copy_(
                    v_b.to(device=v_proj.linear.bias.device, dtype=v_proj.linear.bias.dtype)
                )
            q_proj.linear.weight.requires_grad = c_attn.weight.requires_grad
            k_proj.linear.weight.requires_grad = c_attn.weight.requires_grad
            v_proj.linear.weight.requires_grad = c_attn.weight.requires_grad
            if c_attn.bias is not None:
                if q_proj.linear.bias is not None:
                    q_proj.linear.bias.requires_grad = c_attn.bias.requires_grad
                if k_proj.linear.bias is not None:
                    k_proj.linear.bias.requires_grad = c_attn.bias.requires_grad
                if v_proj.linear.bias is not None:
                    v_proj.linear.bias.requires_grad = c_attn.bias.requires_grad
            self.q_proj = _MaskAwareSTACRLinear(q_proj) if mask_aware else q_proj
            self.k_proj = _MaskAwareSTACRLinear(k_proj) if mask_aware else k_proj
            self.v_proj = _MaskAwareSTACRLinear(v_proj) if mask_aware else v_proj
        else:
            raise TypeError(f"Unsupported c_attn type: {type(c_attn)}")
        self.train(c_attn.training)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        return torch.cat([q, k, v], dim=-1)


def _wrap_joint_qk_forward_bert(module: nn.Module, joint_wrapper: JointQKSTACRLinear) -> None:
    """Wrap BERT-style attention forward to use joint Q/K compression.

    For BERT attention, we need to intercept the forward pass and use the joint
    wrapper for Q and K projections while keeping V separate.
    """
    if getattr(module, "_stacr_joint_qk_wrapped", False):
        return
    original_forward = module.forward
    original_query = module.query
    original_key = module.key

    class _QProxy(nn.Module):
        def __init__(self, wrapper):
            super().__init__()
            self._wrapper = wrapper

        def forward(self, x):
            return self._wrapper._q_wrapper(x)

    class _KProxy(nn.Module):
        def __init__(self, wrapper):
            super().__init__()
            self._wrapper = wrapper

        def forward(self, x):
            return self._wrapper._k_wrapper(x)

    module.query = _QProxy(joint_wrapper)
    module.key = _KProxy(joint_wrapper)
    module._stacr_joint_qk_wrapped = True


def _wrap_joint_qk_forward_gpt(module: nn.Module, joint_wrapper: JointQKSTACRLinear) -> None:
    """Wrap GPT-style attention forward to use joint Q/K compression.

    For GPT attention (q_proj/k_proj/v_proj), we replace q_proj and k_proj with
    proxy modules that use the shared basis joint wrapper.
    """
    if getattr(module, "_stacr_joint_qk_wrapped", False):
        return

    class _QProxy(nn.Module):
        def __init__(self, wrapper):
            super().__init__()
            self._wrapper = wrapper

        def forward(self, x):
            return self._wrapper._q_wrapper(x)

    class _KProxy(nn.Module):
        def __init__(self, wrapper):
            super().__init__()
            self._wrapper = wrapper

        def forward(self, x):
            return self._wrapper._k_wrapper(x)

    module.q_proj = _QProxy(joint_wrapper)
    module.k_proj = _KProxy(joint_wrapper)
    module._stacr_joint_qk_wrapped = True


def apply_stacr_to_attention(
    model: nn.Module,
    config: STACRConfig,
    *,
    enable_gpt2_c_attn_split: bool = False,
    mask_aware: bool = False,
    joint_qk: bool = False,
) -> AttentionCompressionReport:
    """Wrap attention Q/K/V projections with STACRLinear.

    Supports modules with query/key/value or q_proj/k_proj/v_proj attributes.
    GPT-2 fused c_attn is supported only when enable_gpt2_c_attn_split is True.

    Args:
        model: The model to modify
        config: STACR configuration
        enable_gpt2_c_attn_split: Enable splitting of GPT-2 fused c_attn
        mask_aware: Skip padded tokens during compression
        joint_qk: Use one shared basis for Q and K projections.
    """
    replaced = 0
    skipped = 0
    joint_qk_pairs = 0
    warnings: List[str] = []
    layer_index = 0
    if joint_qk and getattr(config, "rank_mode", "fixed") != "fixed":
        warnings.append(
            f"joint_qk only supports rank_mode='fixed'; disabling joint_qk for rank_mode={config.rank_mode!r}."
        )
        joint_qk = False
    compressed_types = (STACRLinear, _MaskAwareSTACRLinear, JointQKSTACRLinear)
    for module in model.modules():
        if isinstance(module, _SplitQKVWrapper):
            continue
        if hasattr(module, "query") and hasattr(module, "key") and hasattr(module, "value"):
            try:
                if (
                    joint_qk
                    and (not isinstance(module.query, compressed_types))
                    and (not isinstance(module.key, compressed_types))
                ):
                    joint_wrapper = JointQKSTACRLinear(
                        module.query,
                        module.key,
                        config,
                        mask_aware=mask_aware,
                        layer_index=layer_index,
                    )
                    module._joint_qk_wrapper = joint_wrapper
                    _wrap_joint_qk_forward_bert(module, joint_wrapper)
                    layer_index += 1
                    replaced += 2
                    joint_qk_pairs += 1
                else:
                    if not isinstance(module.query, compressed_types):
                        module.query = _wrap_projection(
                            module.query, config, mask_aware=mask_aware, layer_index=layer_index
                        )
                        layer_index += 1
                        replaced += 1
                    if not isinstance(module.key, compressed_types):
                        module.key = _wrap_projection(
                            module.key, config, mask_aware=mask_aware, layer_index=layer_index
                        )
                        layer_index += 1
                        replaced += 1
                if not isinstance(module.value, compressed_types):
                    module.value = _wrap_projection(
                        module.value, config, mask_aware=mask_aware, layer_index=layer_index
                    )
                    layer_index += 1
                    replaced += 1
                if mask_aware:
                    _wrap_attention_forward(module)
            except TypeError as exc:
                warnings.append(str(exc))
                skipped += 1
            continue
        if hasattr(module, "q_proj") and hasattr(module, "k_proj") and hasattr(module, "v_proj"):
            try:
                if (
                    joint_qk
                    and (not isinstance(module.q_proj, compressed_types))
                    and (not isinstance(module.k_proj, compressed_types))
                ):
                    joint_wrapper = JointQKSTACRLinear(
                        module.q_proj,
                        module.k_proj,
                        config,
                        mask_aware=mask_aware,
                        layer_index=layer_index,
                    )
                    module._joint_qk_wrapper = joint_wrapper
                    _wrap_joint_qk_forward_gpt(module, joint_wrapper)
                    layer_index += 1
                    replaced += 2
                    joint_qk_pairs += 1
                else:
                    if not isinstance(module.q_proj, compressed_types):
                        module.q_proj = _wrap_projection(
                            module.q_proj, config, mask_aware=mask_aware, layer_index=layer_index
                        )
                        layer_index += 1
                        replaced += 1
                    if not isinstance(module.k_proj, compressed_types):
                        module.k_proj = _wrap_projection(
                            module.k_proj, config, mask_aware=mask_aware, layer_index=layer_index
                        )
                        layer_index += 1
                        replaced += 1
                if not isinstance(module.v_proj, compressed_types):
                    module.v_proj = _wrap_projection(
                        module.v_proj, config, mask_aware=mask_aware, layer_index=layer_index
                    )
                    layer_index += 1
                    replaced += 1
                if mask_aware:
                    _wrap_attention_forward(module)
            except TypeError as exc:
                warnings.append(str(exc))
                skipped += 1
            continue
        if hasattr(module, "c_attn"):
            if enable_gpt2_c_attn_split:
                try:
                    module.c_attn = _SplitQKVWrapper(
                        module.c_attn, config, mask_aware=mask_aware, layer_start_index=layer_index
                    )
                    layer_index += 3
                    replaced += 3
                    if mask_aware:
                        _wrap_attention_forward(module)
                except TypeError as exc:
                    warnings.append(str(exc))
                    skipped += 1
            else:
                warnings.append(
                    "Found GPT-2 c_attn but split is disabled. Set enable_gpt2_c_attn_split=True to replace it."
                )
                skipped += 1
    if replaced == 0 and skipped == 0:
        warnings.append(
            "No attention modules found. Model may not have query/key/value, q_proj/k_proj/v_proj, or c_attn attributes. Check model architecture."
        )
    return AttentionCompressionReport(
        replaced_qkv=replaced, skipped=skipped, warnings=warnings, joint_qk_pairs=joint_qk_pairs
    )
