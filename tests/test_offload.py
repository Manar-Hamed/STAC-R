"""Saved-tensor offload must preserve layouts used by fused attention backward."""

import contextlib
import copy

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from stacr.composition import offload_blocks


class _FusedQKVAttention(nn.Module):
    def __init__(self, dropout):
        super().__init__()
        self.qkv = nn.Linear(128, 384)
        self.dropout = dropout

    def forward(self, x):
        batch, length, _ = x.shape
        q, k, v = self.qkv(x).split(128, dim=-1)
        q, k, v = (tensor.view(batch, length, 2, 64).transpose(1, 2) for tensor in (q, k, v))
        assert not q.is_contiguous()
        attended = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout, is_causal=True)
        return attended.transpose(1, 2).reshape(batch, length, 128)


def _attention_gradients(block, inputs, upstream, dtype):
    from torch.nn.attention import SDPBackend, sdpa_kernel

    x = inputs.detach().clone().requires_grad_(True)
    block.zero_grad(set_to_none=True)
    torch.manual_seed(231)
    device_type = inputs.device.type
    if device_type == "cuda":
        torch.cuda.manual_seed_all(231)
    amp = (
        torch.autocast(device_type, dtype=dtype)
        if dtype != torch.float32
        else contextlib.nullcontext()
    )
    # Efficient attention assumes a particular saved-output stride in backward.
    # A contiguous pinned copy can violate it even when tensor values are exact.
    backend = SDPBackend.EFFICIENT_ATTENTION if device_type == "cuda" else SDPBackend.MATH
    with sdpa_kernel(backend), amp:
        output = block(x)
    gradients = torch.autograd.grad(output, (x, *block.parameters()), upstream.to(output.dtype))
    return output.detach(), gradients


@pytest.mark.parametrize(
    "device_type,dtype",
    [
        ("cpu", torch.float32),
        ("cuda", torch.float32),
        ("cuda", torch.float16),
        ("cuda", torch.bfloat16),
    ],
)
@pytest.mark.parametrize("dropout", [0.0, 0.1])
@pytest.mark.parametrize("pin_memory", [False, True])
@pytest.mark.parametrize("implementation", ["blocks", "protocol"])
def test_offload_preserves_fused_qkv_attention_gradients(
    device_type, dtype, dropout, pin_memory, implementation
):
    if device_type == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    if device_type == "cuda" and dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        pytest.skip("CUDA BF16 unavailable")
    torch.manual_seed(47)
    reference = _FusedQKVAttention(dropout).to(device_type)
    offloaded = copy.deepcopy(reference)
    if implementation == "blocks":
        assert offload_blocks([offloaded], [0], pin_memory=pin_memory) == [0]
    else:
        from stacr.protocols.baselines.offload import SavedTensorOffloader

        offloader = SavedTensorOffloader("cpu_pinned" if pin_memory else "cpu", min_bytes=0)
        original_forward = offloaded.forward

        def forward(*args, **kwargs):
            with offloader.context():
                return original_forward(*args, **kwargs)

        offloaded.forward = forward
    inputs = torch.randn(2, 64, 128, device=device_type)
    upstream = torch.randn_like(inputs)

    expected, expected_gradients = _attention_gradients(reference, inputs, upstream, dtype)
    actual, actual_gradients = _attention_gradients(offloaded, inputs, upstream, dtype)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    rtol, atol = (
        (0.02, 0.002)
        if dtype == torch.bfloat16
        else (0.002, 0.0003)
        if dtype == torch.float16
        else (2e-5, 2e-5)
    )
    for actual_gradient, expected_gradient in zip(actual_gradients, expected_gradients):
        assert torch.isfinite(actual_gradient).all()
        torch.testing.assert_close(actual_gradient, expected_gradient, rtol=rtol, atol=atol)


@pytest.mark.parametrize("device_type", ["cpu", "cuda"])
@pytest.mark.parametrize("pin_memory", [False, True])
def test_offload_context_preserves_saved_dense_transpose_strides(device_type, pin_memory):
    from stacr.composition import saved_tensors_on_cpu

    if device_type == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    source = torch.randn(2, 64, 2, 64, device=device_type).transpose(1, 2)
    source.requires_grad_(True)
    stride = source.stride()

    class InspectSaved(torch.autograd.Function):
        @staticmethod
        def forward(ctx, tensor):
            ctx.save_for_backward(tensor)
            return tensor.square()

        @staticmethod
        def backward(ctx, gradient):
            (restored,) = ctx.saved_tensors
            assert restored.stride() == stride
            torch.testing.assert_close(restored, source.detach(), rtol=0, atol=0)
            return gradient * 2 * restored

    with saved_tensors_on_cpu(pin_memory=pin_memory):
        output = InspectSaved.apply(source)
    output.sum().backward()
    torch.testing.assert_close(source.grad, 2 * source.detach(), rtol=0, atol=0)
