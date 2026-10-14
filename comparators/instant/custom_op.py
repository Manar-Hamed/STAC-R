# Adapted from INSTANT, copyright (c) 2024 Trung-Hieu Tran. MIT; see LICENSE.txt.
# Upstream revision: 8a04bd43c19cc52984a49fb1b7fd1dae14fb2ffc.
import torch as th
import torch.nn as nn
from torch.autograd import Function
import torch.nn.functional as F
from typing import Any
import math


class LinearSVDOp(Function):
    @staticmethod
    def jvp(ctx: Any, *grad_inputs: Any) -> Any:
        Function.jvp(ctx, *grad_inputs)

    @staticmethod
    def forward(ctx: Any, *args: Any, **kwargs: Any) -> Any:
        (x, weight, bias, CompressionTensor_x, CompressionTensor_gy) = args
        x_compressed = CompressionTensor_x @ x
        y = F.linear(x, weight, bias)
        ctx.save_for_backward(
            x_compressed,
            weight,
            th.tensor([bias is not None]),
            CompressionTensor_x,
            CompressionTensor_gy,
        )
        return y

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Any:
        (x_compressed, weight, has_bias, CompressionTensor_x, CompressionTensor_gy) = (
            ctx.saved_tensors
        )
        (grad_y,) = grad_outputs
        grad_x = CompressionTensor_gy.T @ (CompressionTensor_gy @ grad_y @ weight)
        grad_w = (grad_y.transpose(1, 2) @ CompressionTensor_x.T @ x_compressed).sum(dim=0)
        if has_bias:
            grad_b = grad_y.sum(dim=(0, 1))
        else:
            grad_b = None
        return (grad_x, grad_w, grad_b, None, None)


class LinearCompressClass(nn.Linear):
    def __init__(
        self, in_features: int, out_features: int, bias: bool = True, device=None, dtype=None
    ) -> None:
        super().__init__(in_features, out_features, bias, device, dtype)

    def forward(self, input: th.Tensor) -> th.Tensor:
        if self.activate and self.training:
            y = LinearSVDOp.apply(
                input,
                self.weight,
                self.bias,
                self.CompressionTensor_x.T.clone().detach().requires_grad_(False).to(input.device),
                self.CompressionTensor_gy.T.clone().detach().requires_grad_(False).to(input.device),
            )
        else:
            y = super().forward(input)
        return y


def wrap_linear_compression_layer(linear_layer: nn.Linear, **kwargs):
    new_linear = LinearCompressClass(
        in_features=linear_layer.in_features,
        out_features=linear_layer.out_features,
        bias=linear_layer.bias is not None,
    )
    new_linear.weight.data = linear_layer.weight.data
    new_linear.weight.requires_grad_(linear_layer.weight.requires_grad)
    if linear_layer.bias is not None:
        new_linear.bias.data = linear_layer.bias.data
        new_linear.bias.requires_grad_(linear_layer.bias.requires_grad)
    new_linear.train(linear_layer.training)
    return new_linear


def register_filter(model, modules_compressed):
    for name, module in model.named_modules():
        if name in modules_compressed:
            compressed_module = wrap_linear_compression_layer(module)
            parent_module = dict(model.named_modules())[name.rsplit(".", 1)[0]]
            setattr(parent_module, name.rsplit(".", 1)[1], compressed_module)
    return model
