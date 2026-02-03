"""
Quantization transformation functions for weights.

These functions apply actual quantization transformations to tensors,
converting floating-point weights to quantized representations (INT8, FP16).
"""

from typing import Union

import torch

from mono_quant.core.mappers import calculate_scale_zp_per_channel


def quantize_weight_int8(
    weight: torch.Tensor,
    symmetric: bool = False,
    axis: int = 0,
) -> torch.Tensor:
    """
    Quantize a weight tensor to INT8 using per-channel scaling.

    This function applies per-channel affine quantization to weight tensors,
    which is the standard approach for neural network weight quantization.
    Per-channel quantization uses separate scale and zero-point for each
    channel along the specified axis.

    Args:
        weight: Input weight tensor to quantize. For standard PyTorch layer
            weights, this is typically (out_features, in_features) for Linear
            or (out_channels, in_channels, kernel_h, kernel_w) for Conv2d.
        symmetric: If True, use symmetric quantization (zero_point = 0).
                   If False, use asymmetric (affine) quantization. Default is False.
        axis: The channel axis for per-channel quantization. Default is 0,
              which corresponds to the output channel dimension for standard
              PyTorch weight layouts (nn.Linear, nn.Conv2d).

    Returns:
        Quantized tensor with dtype torch.qint8. The tensor stores the
        quantized integer values along with per-channel scale and zero_point
        metadata.

    Examples:
        >>> import torch
        >>> # Linear layer weight: (out_features, in_features)
        >>> w = torch.randn(64, 128)
        >>> qw = quantize_weight_int8(w, symmetric=False, axis=0)
        >>> assert qw.dtype == torch.qint8
        >>> assert qw.qscheme() == torch.per_channel_affine
    """
    # Calculate per-channel scale and zero-point
    scale, zero_point = calculate_scale_zp_per_channel(
        weight, dtype=torch.qint8, symmetric=symmetric, axis=axis
    )

    # Apply per-channel quantization using PyTorch's native primitive
    # zero_point must be int type for torch.quantize_per_channel
    quantized = torch.quantize_per_channel(
        weight,
        scales=scale,
        zero_points=zero_point.int(),
        axis=axis,
        dtype=torch.qint8,
    )

    return quantized


def quantize_weight_fp16(weight: torch.Tensor) -> torch.Tensor:
    """
    Quantize a weight tensor to FP16 using dtype casting.

    FP16 quantization uses simple dtype casting rather than a full
    quantization pipeline with scale and zero-point. This approach is
    simpler and effective for memory reduction, as FP16 has native
    hardware support on modern accelerators.

    See 01-RESEARCH.md "FP16 Quantization (Simple Cast Approach)" and
    discussion of Float16 Dynamic Quantization model size benefits.

    Args:
        weight: Input weight tensor to convert to FP16.

    Returns:
        Weight tensor with dtype torch.float16.

    Examples:
        >>> import torch
        >>> w = torch.randn(64, 128)
        >>> w_fp16 = quantize_weight_fp16(w)
        >>> assert w_fp16.dtype == torch.float16
        >>> assert w_fp16.element_size() == 2  # 2 bytes for float16
    """
    return weight.to(torch.float16)


def dequantize_weight(q_weight: torch.Tensor) -> torch.Tensor:
    """
    Dequantize a quantized weight tensor back to float32.

    This function handles both INT8 quantized tensors (created by
    torch.quantize_per_channel) and FP16 tensors. For INT8, it uses
    torch.dequantize() to apply the affine transformation. For FP16,
    it simply casts back to float32.

    Args:
        q_weight: Quantized weight tensor. Either a torch.qint8 quantized
                  tensor or a torch.float16 tensor.

    Returns:
        Dequantized weight tensor with dtype torch.float32.

    Raises:
        TypeError: If the input tensor is not a recognized quantized type.

    Examples:
        >>> import torch
        >>> w = torch.randn(64, 128)
        >>> qw = quantize_weight_int8(w)
        >>> w_dq = dequantize_weight(qw)
        >>> assert w_dq.dtype == torch.float32
    """
    # Check if it's a quantized tensor (has dequantize method)
    if hasattr(q_weight, 'dequantize'):
        # INT8 quantized tensor - use torch.dequantize()
        return torch.dequantize(q_weight)
    elif q_weight.dtype == torch.float16:
        # FP16 tensor - simple dtype cast
        return q_weight.to(torch.float32)
    else:
        raise TypeError(
            f"Unsupported quantized tensor type: {type(q_weight)}. "
            f"Expected torch.qint8 quantized tensor or torch.float16 tensor."
        )
