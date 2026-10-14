from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class QuantizedValues:
    """Uniform symmetric quantization of a 1-D value vector."""

    data: torch.Tensor
    scale: torch.Tensor
    bits: int
    length: int

    def dequantize(self, dtype: torch.dtype) -> torch.Tensor:
        if self.bits == 4:
            unpacked = unpack_int4(self.data, self.length)
            return unpacked.to(dtype) * self.scale.to(dtype)
        return self.data.to(dtype) * self.scale.to(dtype)

    def stored_bytes(self) -> int:
        return (
            self.data.numel() * self.data.element_size()
            + self.scale.numel() * self.scale.element_size()
        )


def pack_int4(values: torch.Tensor) -> torch.Tensor:
    """Pack signed 4-bit values (-8..7) two per byte."""
    flat = values.reshape(-1).to(torch.int8)
    if flat.numel() % 2 == 1:
        flat = torch.cat([flat, flat.new_zeros(1)])
    low = (flat[0::2] & 0x0F).to(torch.uint8)
    high = (flat[1::2] & 0x0F).to(torch.uint8)
    return (low | (high << 4)).contiguous()


def unpack_int4(packed: torch.Tensor, length: int) -> torch.Tensor:
    low = (packed & 0x0F).to(torch.int8)
    high = ((packed >> 4) & 0x0F).to(torch.int8)
    low = torch.where(low > 7, low - 16, low)
    high = torch.where(high > 7, high - 16, high)
    out = torch.stack([low, high], dim=1).reshape(-1)
    return out[:length]


def quantize_values(values: torch.Tensor, bits: int) -> QuantizedValues:
    """Symmetric per-tensor quantization; ``bits=0`` keeps full precision."""
    if bits not in (0, 4, 8):
        raise ValueError("residual_bits must be 0, 4, or 8")
    flat = values.reshape(-1)
    length = int(flat.numel())
    if bits == 0 or length == 0:
        return QuantizedValues(
            flat, torch.ones((), device=flat.device, dtype=torch.float32), 0, length
        )
    qmax = 7.0 if bits == 4 else 127.0
    # Computing the scale in fp16 underflows for small (or zero) residuals.
    amax = flat.float().abs().max()
    scale = (amax / qmax).clamp_min(torch.finfo(torch.float32).tiny)
    q = torch.round(flat.float() / scale).clamp_(-qmax - 1.0, qmax).to(torch.int8)
    data = pack_int4(q) if bits == 4 else q
    return QuantizedValues(data, scale.reshape(()).float(), bits, length)
