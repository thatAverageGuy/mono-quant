"""
Scale and zero-point calculation functions for quantization.

These functions provide a functional API for calculating quantization parameters
without requiring scheme class instantiation. They support per-tensor and per-channel
quantization with both symmetric and asymmetric modes.
"""

from typing import Optional, Tuple

import torch


def get_dtype_range(dtype: torch.dtype) -> Tuple[int, int]:
    """
    Get the quantization range (qmin, qmax) for a given dtype.

    Args:
        dtype: PyTorch quantization dtype (e.g., torch.qint8, torch.quint8).

    Returns:
        Tuple of (qmin, qmax) representing the minimum and maximum
        quantized integer values for the dtype.

    Raises:
        ValueError: If the dtype is not supported for quantization.
    """
    # Note: qint4/quint4 will be added when PyTorch supports them
    # PyTorch 2.10+ has int1-int7 dtypes but not qint4 specifically
    dtype_ranges = {
        torch.qint8: (-128, 127),      # Signed 8-bit
        torch.quint8: (0, 255),         # Unsigned 8-bit
    }

    if dtype not in dtype_ranges:
        raise ValueError(
            f"Unsupported dtype {dtype}. "
            f"Supported dtypes: {list(dtype_ranges.keys())}"
        )

    return dtype_ranges[dtype]


def calculate_scale_zp_per_tensor(
    tensor: torch.Tensor,
    dtype: torch.dtype = torch.qint8,
    symmetric: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Calculate per-tensor scale and zero-point for quantization.

    Per-tensor quantization uses a single scale and zero-point for the entire tensor.
    This is simpler but may be less accurate than per-channel quantization.

    Args:
        tensor: Input tensor to calculate parameters for.
        dtype: Target quantization dtype (default: torch.qint8).
        symmetric: If True, use symmetric quantization (zero_point = 0).
                   If False, use asymmetric (affine) quantization.

    Returns:
        Tuple of (scale, zero_point). Both are scalar tensors.
        - For FP16 (torch.float16), returns (None, None) as FP16 doesn't use scale/zp.

    Examples:
        >>> import torch
        >>> w = torch.randn(128, 256)
        >>> scale, zp = calculate_scale_zp_per_tensor(w, symmetric=True)
        >>> assert zp == 0  # Symmetric has zero offset
    """
    # FP16 doesn't use scale/zero-point (simple dtype casting)
    if dtype == torch.float16:
        return None, None

    qmin, qmax = get_dtype_range(dtype)

    if symmetric:
        # Symmetric: zero_point = 0, scale from max absolute value
        max_abs = tensor.abs().max()
        scale = max_abs / qmax
        zero_point = 0  # Symmetric always has zero offset
    else:
        # Asymmetric: full affine quantization
        min_val = tensor.amin()
        max_val = tensor.amax()
        range_val = max_val - min_val
        q_range = qmax - qmin

        scale = range_val / q_range
        # Clamp scale BEFORE calculating zero_point to avoid inf when range is 0
        scale = torch.clamp(scale, min=1e-8)
        zero_point = qmin - (min_val / scale)

    # Clamp scale (already done for asymmetric, do for symmetric too)
    scale = torch.clamp(scale, min=1e-8)

    # Convert zero_point to int (for asymmetric; symmetric is already 0)
    if not symmetric:
        zero_point = int(zero_point.item())

    return scale, zero_point


def calculate_scale_zp_per_channel(
    tensor: torch.Tensor,
    dtype: torch.dtype = torch.qint8,
    symmetric: bool = False,
    axis: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Calculate per-channel scale and zero-point for quantization.

    Per-channel quantization calculates separate scale and zero-point for each
    channel along the specified axis. This is the standard approach for weight
    quantization in neural networks (e.g., per-output-channel for Linear layers).

    Args:
        tensor: Input tensor to calculate parameters for.
        dtype: Target quantization dtype (default: torch.qint8).
        symmetric: If True, use symmetric quantization (zero_point = 0).
                   If False, use asymmetric (affine) quantization.
        axis: The channel axis to calculate per-channel parameters for.
              Default is 0, which corresponds to the output channel dimension
              for standard nn.Linear and nn.Conv2d weight tensors.

    Returns:
        Tuple of (scale, zero_point) tensors.
        - scale: Per-channel scale tensor with shape matching tensor.shape[axis]
        - zero_point: Per-channel zero-point tensor (zeros for symmetric)
        - For FP16, returns (None, None)

    Examples:
        >>> import torch
        >>> # Linear layer weights: (out_features, in_features)
        >>> w = torch.randn(64, 128)  # 64 output channels
        >>> scale, zp = calculate_scale_zp_per_channel(w, axis=0)
        >>> assert scale.shape == (64,)  # One scale per output channel
        >>> assert zp.shape == (64,)
    """
    # FP16 doesn't use scale/zero-point
    if dtype == torch.float16:
        return None, None

    qmin, qmax = get_dtype_range(dtype)

    # Determine reduction dimensions: all EXCEPT the specified axis
    # For axis=0 on shape (64, 128), this reduces over dim 1, giving (64,)
    reduce_dims = tuple(i for i in range(tensor.dim()) if i != axis)

    if not reduce_dims:
        # Single dimension tensor - no reduction needed
        if symmetric:
            max_abs = tensor.abs()
            scale = max_abs / qmax
            zero_point = torch.zeros_like(scale)
        else:
            min_val = tensor
            max_val = tensor
            range_val = max_val - min_val
            q_range = qmax - qmin
            scale = range_val / q_range
            # Clamp scale BEFORE calculating zero_point to avoid inf when range is 0
            scale = torch.clamp(scale, min=1e-8)
            zero_point = qmin - (min_val / scale)
    else:
        if symmetric:
            # Symmetric: reduce max_abs over non-channel dimensions
            max_abs = tensor.abs().amax(dim=reduce_dims, keepdim=False)
            scale = max_abs / qmax
            zero_point = torch.zeros_like(scale)
        else:
            # Asymmetric: reduce min/max over non-channel dimensions
            min_val = tensor.amin(dim=reduce_dims, keepdim=False)
            max_val = tensor.amax(dim=reduce_dims, keepdim=False)
            range_val = max_val - min_val
            q_range = qmax - qmin
            scale = range_val / q_range
            # Clamp scale BEFORE calculating zero_point to avoid inf when range is 0
            scale = torch.clamp(scale, min=1e-8)
            zero_point = qmin - (min_val / scale)

    # Clamp scale (already done for asymmetric, do for symmetric too)
    scale = torch.clamp(scale, min=1e-8)

    # Convert zero_point to int for asymmetric
    if not symmetric:
        zero_point = zero_point.to(torch.int32)

    return scale, zero_point
