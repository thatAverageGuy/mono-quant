"""
Scale and zero-point calculation functions for quantization.

These functions provide a functional API for calculating quantization parameters
without requiring scheme class instantiation. They support per-tensor, per-channel,
and group-wise quantization with both symmetric and asymmetric modes.
"""

from typing import Tuple

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
        # Clamp zero_point to valid range [qmin, qmax] to avoid runtime errors
        zero_point = max(qmin, min(qmax, zero_point))

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

    # Convert zero_point to int for asymmetric and clamp to valid range
    if not symmetric:
        zero_point = zero_point.to(torch.int32)
        # Clamp zero_point to valid range [qmin, qmax] to avoid runtime errors
        zero_point = torch.clamp(zero_point, qmin, qmax)

    return scale, zero_point


def calculate_scale_zp_groupwise(
    tensor: torch.Tensor,
    group_size: int = 128,
    axis: int = 0,
    symmetric: bool = True,
    dtype: torch.dtype = torch.qint8,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Calculate group-wise scale and zero-point for INT4 quantization.

    Group-wise quantization divides output channels into groups of fixed size,
    with each group sharing a scale and zero-point. This is essential for INT4
    quantization where per-channel scaling would result in unacceptable accuracy
    loss. The industry standard group_size of 128 is used by AWQ, GPTQ, and
    HuggingFace.

    Layers smaller than group_size safely fall back to per-channel quantization
    to avoid edge cases with partial groups.

    Args:
        tensor: Input weight tensor to calculate parameters for. For standard
            PyTorch layer weights, this is typically (out_features, in_features)
            for Linear or (out_channels, in_channels, kernel_h, kernel_w) for Conv2d.
        group_size: Number of channels per group. Default is 128 (industry standard).
        axis: The channel axis to group along. Default is 0, which corresponds
              to the output channel dimension for standard PyTorch weight layouts.
        symmetric: If True, use symmetric quantization (zero_point = 0).
                   If False, use asymmetric (affine) quantization. Default is True.
        dtype: Target quantization dtype. Default is torch.qint8.
               Note: INT4 uses [-8, 7] range regardless of dtype parameter.

    Returns:
        Tuple of (scale, zero_point) tensors.
        - scale: Per-group scale tensor with shape (num_groups,)
        - zero_point: Per-group zero-point tensor with shape (num_groups,)
                      (zeros for symmetric quantization)

    Raises:
        ValueError: If group_size is not positive.

    Examples:
        >>> import torch
        >>> # Linear layer weight: (out_features=256, in_features=128)
        >>> w = torch.randn(256, 128)
        >>> scale, zp = calculate_scale_zp_groupwise(w, group_size=128, axis=0)
        >>> assert scale.shape == (2,)  # 256 / 128 = 2 groups
        >>> assert zp.shape == (2,)
        >>> assert torch.all(zp == 0)  # Symmetric has zero offset

        Layer smaller than group_size (falls back to per-channel):
        >>> w_small = torch.randn(64, 128)  # 64 < 128
        >>> scale_small, zp_small = calculate_scale_zp_groupwise(w_small, group_size=128)
        >>> assert scale_small.shape == (64,)  # One per channel (fallback)
    """
    if group_size <= 0:
        raise ValueError(f"group_size must be positive, got {group_size}")

    # Get dimension size along the grouping axis
    dim_size = tensor.shape[axis]

    # Fallback to per-channel for layers smaller than group_size
    # This avoids edge cases with partial groups and is the safe approach
    if dim_size < group_size:
        return calculate_scale_zp_per_channel(
            tensor, dtype=dtype, symmetric=symmetric, axis=axis
        )

    # Calculate number of groups (handle partial groups)
    num_groups = dim_size // group_size
    has_partial_group = (dim_size % group_size) != 0
    if has_partial_group:
        num_groups += 1

    # Move axis to dim 0 for easier grouping
    # Shape after permute: (dim_size, *other_dims)
    permuted_dims = list(range(tensor.dim()))
    permuted_dims[0], permuted_dims[axis] = permuted_dims[axis], permuted_dims[0]
    weight_permuted = tensor.permute(permuted_dims)

    # Flatten remaining dimensions for processing
    # Shape: (dim_size, -1)
    weight_flat = weight_permuted.reshape(dim_size, -1)

    # INT4 signed range: [-8, 7]
    # For symmetric: zero_point = 0, scale = max_abs / qmax
    # For asymmetric: use standard affine calculation
    qmin, qmax = -8, 7

    if symmetric:
        # Symmetric: zero_point = 0 for all groups
        zero_point = torch.zeros(num_groups, dtype=torch.int32, device=tensor.device)

        # Calculate scale per group
        scales = []

        for g in range(num_groups):
            start_idx = g * group_size
            end_idx = min(start_idx + group_size, dim_size)

            # Extract group weights
            group_weights = weight_flat[start_idx:end_idx]  # (group_size, -1)

            # Max absolute value in the group
            max_abs = group_weights.abs().max()

            # Scale = max_abs / qmax
            scale = max_abs / qmax
            scales.append(scale)

        scale = torch.stack(scales)

    else:
        # Asymmetric: full affine quantization per group
        scales = []
        zero_points = []

        for g in range(num_groups):
            start_idx = g * group_size
            end_idx = min(start_idx + group_size, dim_size)

            # Extract group weights
            group_weights = weight_flat[start_idx:end_idx]  # (group_size, -1)

            # Min/max in the group
            min_val = group_weights.min()
            max_val = group_weights.max()

            # Calculate scale and zero_point
            range_val = max_val - min_val
            q_range = qmax - qmin

            scale_val = range_val / q_range
            # Clamp scale to avoid inf when range is 0
            scale_val = max(scale_val.item(), 1e-8)

            zp_val = qmin - (min_val / scale_val)
            # Convert to int and clamp to valid range
            zp_val = int(round(zp_val.item()))
            zp_val = max(qmin, min(qmax, zp_val))

            scales.append(scale_val)
            zero_points.append(zp_val)

        scale = torch.tensor(scales, dtype=torch.float32, device=tensor.device)
        zero_point = torch.tensor(zero_points, dtype=torch.int32, device=tensor.device)

    # Clamp scale to avoid numerical issues
    scale = torch.clamp(scale, min=1e-8)

    return scale, zero_point


def _pack_int4_to_int8(int4_tensor: torch.Tensor) -> torch.Tensor:
    """
    Pack INT4 values into INT8 bytes.

    Each INT8 byte stores two INT4 values:
    - Low nibble (bits 0-3): First INT4 value
    - High nibble (bits 4-7): Second INT4 value

    This follows the standard bit packing pattern used by AWQ, GPTQ,
    and HuggingFace for efficient INT4 weight storage.

    Args:
        int4_tensor: Tensor with INT4 values in range [-8, 7].
                     Should have even number of elements.

    Returns:
        Packed int8 tensor with half the number of elements.
        Each byte contains two packed INT4 values.

    Examples:
        >>> import torch
        >>> # Create INT4 values (should be in range [-8, 7])
        >>> int4_vals = torch.tensor([1, -2, 7, -8, 0, 3], dtype=torch.int32)
        >>> packed = _pack_int4_to_int8(int4_vals)
        >>> assert packed.dtype == torch.int8
        >>> assert packed.numel() == 3  # Half of original

    Note:
        Input values are assumed to be in the valid INT4 range [-8, 7].
        Values outside this range will be truncated to 4 bits during packing.
    """
    # Ensure we're working with a flat tensor
    flat = int4_tensor.flatten()

    # Handle odd length by padding
    original_length = flat.numel()
    if original_length % 2 != 0:
        flat = torch.cat([flat, torch.zeros(1, dtype=flat.dtype, device=flat.device)])
        padded = True
    else:
        padded = False

    # Reshape to pairs
    pairs = flat.reshape(-1, 2)

    # Pack two INT4 values into one INT8 byte
    # Low nibble: first value & 0x0F
    # High nibble: (second value & 0x0F) << 4
    low = pairs[:, 0] & 0x0F
    high = (pairs[:, 1] & 0x0F) << 4

    packed = low | high

    # Convert to int8 dtype
    packed = packed.to(torch.int8)

    # Remove padding if we added it
    if padded:
        packed = packed[:-1]

    return packed


def _unpack_int8_to_int4(packed_int8: torch.Tensor, num_elements: int) -> torch.Tensor:
    """
    Unpack INT8 bytes to INT4 values.

    This is the inverse of _pack_int4_to_int8. Each INT8 byte contains
    two INT4 values that need to be unpacked.

    Args:
        packed_int8: Packed int8 tensor containing INT4 values.
        num_elements: Number of INT4 values to unpack (may be odd).

    Returns:
        Unpacked tensor with INT4 values as int32 (for signed computation).

    Examples:
        >>> import torch
        >>> packed = torch.tensor([0x12, 0x34], dtype=torch.int8)
        >>> unpacked = _unpack_int8_to_int4(packed, num_elements=4)
        >>> assert unpacked.numel() == 4
    """
    # Number of packed bytes needed
    num_packed = (num_elements + 1) // 2

    # Only use the bytes we need
    if packed_int8.numel() > num_packed:
        packed_int8 = packed_int8[:num_packed]

    # Extract low and high nibbles
    low = packed_int8.to(torch.int32) & 0x0F
    high = (packed_int8.to(torch.int32) >> 4) & 0x0F

    # Interleave low and high
    unpacked = torch.zeros(num_elements, dtype=torch.int32, device=packed_int8.device)

    # Fill in pairs
    num_pairs = num_elements // 2
    unpacked[0::2] = low[:num_pairs]
    unpacked[1::2] = high[:num_pairs]

    # Handle odd length (last element is from high nibble of last byte)
    if num_elements % 2 != 0:
        unpacked[-1] = high[num_pairs]

    # Convert to signed INT4 (two's complement interpretation)
    # If value >= 8, it's negative: value - 16
    unpacked = torch.where(unpacked >= 8, unpacked - 16, unpacked)

    return unpacked
