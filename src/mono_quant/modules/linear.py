"""
Quantized Linear and Conv2d module implementations.

This module provides quantized replacements for PyTorch's nn.Linear and
nn.Conv2d layers. The QuantizedLinear module stores quantized weights
and dequantizes them during forward pass for inference.
"""

import copy
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from mono_quant.core.quantizers import quantize_weight_int8, dequantize_weight, quantize_weight_int4


class QuantizedLinear(nn.Module):
    """
    A Linear layer with quantized weights for inference.

    This module wraps quantized weights (typically INT8) and handles
    dequantization during the forward pass. The weights are quantized
    once during initialization and cached for efficient inference.

    Args:
        in_features: Number of input features.
        out_features: Number of output features.
        bias: If True, adds a learnable bias to the output. Default is True.
        dtype: Quantization dtype. Only torch.qint8 is supported currently.
        symmetric: If True, use symmetric quantization (zero_point = 0).
                   If False, use asymmetric quantization. Default is False.

    Attributes:
        in_features: Number of input features.
        out_features: Number of output features.
        dtype: Quantization dtype.
        symmetric: Whether symmetric quantization was used.
        _quantized_weight: Cached quantized weight tensor.
        bias: Optional bias parameter.

    Examples:
        >>> import torch.nn as nn
        >>> from mono_quant.modules import QuantizedLinear
        >>> layer = QuantizedLinear(128, 64, bias=True, symmetric=False)
        >>> x = torch.randn(32, 128)
        >>> y = layer(x)
        >>> assert y.shape == (32, 64)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        dtype: torch.dtype = torch.qint8,
        symmetric: bool = False,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dtype = dtype
        self.symmetric = symmetric

        # Store quantized weight (cached after quantization)
        self._quantized_weight: Optional[torch.Tensor] = None

        # Bias is stored as-is (not quantized)
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

        # Placeholder for the original weight shape
        # Used to initialize the module before loading quantized weights
        self._weight_shape = (out_features, in_features)

    @classmethod
    def from_linear(
        cls,
        module: nn.Linear,
        symmetric: bool = False,
    ) -> "QuantizedLinear":
        """
        Create a QuantizedLinear from an existing nn.Linear module.

        This factory method quantizes the weights from the source Linear
        module and creates a new QuantizedLinear instance.

        Args:
            module: Source nn.Linear module to quantize.
            symmetric: If True, use symmetric quantization. Default is False.

        Returns:
            A new QuantizedLinear instance with quantized weights.

        Examples:
            >>> import torch.nn as nn
            >>> from mono_quant.modules import QuantizedLinear
            >>> linear = nn.Linear(128, 64)
            >>> q_linear = QuantizedLinear.from_linear(linear, symmetric=False)
        """
        # Create new QuantizedLinear with same configuration
        q_linear = cls(
            in_features=module.in_features,
            out_features=module.out_features,
            bias=module.bias is not None,
            symmetric=symmetric,
        )

        # Quantize the weight
        q_linear._quantized_weight = quantize_weight_int8(
            module.weight.data, symmetric=symmetric, axis=0
        )

        # Copy bias if present
        if module.bias is not None:
            q_linear.bias.data = module.bias.data.clone()

        return q_linear

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with dequantized weights.

        Args:
            input: Input tensor of shape (*, in_features).

        Returns:
            Output tensor of shape (*, out_features).
        """
        # Lazy quantization: if weight hasn't been set, this is likely an error
        # The module should be loaded with quantized weights before use
        if self._quantized_weight is None:
            raise RuntimeError(
                "QuantizedLinear has no quantized weights. "
                "Use from_linear() or set _quantized_weight directly."
            )

        # Dequantize weight for computation
        # Note: In production, specialized kernels can avoid this dequantization
        # by computing directly with quantized values (e.g., using int8 gemm)
        weight = dequantize_weight(self._quantized_weight)

        # Compute linear transformation
        output = F.linear(input, weight, self.bias)

        return output

    @property
    def weight(self) -> Optional[torch.Tensor]:
        """
        Return the dequantized weight (for compatibility).

        This property provides compatibility with code that expects
        a weight attribute on Linear modules.

        Returns:
            Dequantized weight tensor, or None if not quantized yet.
        """
        if self._quantized_weight is None:
            return None
        return dequantize_weight(self._quantized_weight)

    def extra_repr(self) -> str:
        """Return extra representation string."""
        return (
            f"in_features={self.in_features}, "
            f"out_features={self.out_features}, "
            f"dtype={self.dtype}, "
            f"symmetric={self.symmetric}, "
            f"bias={self.bias is not None}"
        )

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        """
        Custom serialization to include quantized weight metadata.

        This method saves the quantized weight along with its scale and
        zero_point information, enabling proper reconstruction during loading.

        Args:
            destination: State dictionary to save to.
            prefix: Prefix for keys in state dict.
            keep_vars: Whether to keep variables as-is.
        """
        # Call parent implementation first
        super()._save_to_state_dict(destination, prefix, keep_vars)

        # Save quantized weight metadata if present
        if self._quantized_weight is not None:
            # Save the int8 representation
            destination[prefix + "_quantized_weight_int8"] = (
                self._quantized_weight.int_repr()
            )
            # Save per-channel scales
            destination[prefix + "_scale"] = (
                self._quantized_weight.q_per_channel_scales()
            )
            # Save per-channel zero_points
            destination[prefix + "_zero_point"] = (
                self._quantized_weight.q_per_channel_zero_points()
            )
            # Save axis
            destination[prefix + "_axis"] = torch.tensor(
                self._quantized_weight.q_per_channel_axis()
            )

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict,
        missing_keys, unexpected_keys, error_msgs
    ):
        """
        Custom deserialization to restore quantized weights.

        This method reconstructs the quantized weight from saved metadata.

        Args:
            state_dict: State dictionary to load from.
            prefix: Prefix for keys in state dict.
            local_metadata: Local metadata.
            strict: Whether to enforce strict key matching.
            missing_keys: List of missing keys (will be populated).
            unexpected_keys: List of unexpected keys (will be populated).
            error_msgs: List of error messages (will be populated).
        """
        # Check if we have quantized weight metadata
        quantized_key = prefix + "_quantized_weight_int8"
        scale_key = prefix + "_scale"
        zero_point_key = prefix + "_zero_point"
        axis_key = prefix + "_axis"

        # If all quantized metadata is present, reconstruct the quantized weight
        if all(k in state_dict for k in [quantized_key, scale_key, zero_point_key, axis_key]):
            # Extract metadata from state_dict
            int8_data = state_dict.pop(quantized_key)
            scale = state_dict.pop(scale_key)
            zero_point = state_dict.pop(zero_point_key)
            axis = state_dict.pop(axis_key).item()

            # Convert int8_data back to the original int8 dtype
            # int_repr() returns an int8 tensor, we need to keep it as int8
            int8_data = int8_data.to(torch.int8)

            # Reconstruct quantized weight using dequantize-requantize approach
            # This ensures the quantized tensor is reconstructed correctly

            # For per-channel quantization, apply scale/zp per channel
            # Convert to int32 to avoid overflow
            int8_data_int32 = int8_data.to(torch.int32)

            # Dequantize: float_value = (int8_value - zero_point) * scale
            # We need to properly reshape scale/zero_point for broadcasting
            # based on the weight shape and axis

            # Get the number of dimensions in the weight
            ndim = int8_data.dim()

            # Reshape scale and zero_point to broadcast correctly
            # For axis=0 (per-channel along first dimension):
            # - Linear: (out_features, in_features) -> scale/zero_point: (out_features, 1)
            # - Conv2d: (out_channels, in_channels, h, w) -> scale/zero_point: (out_channels, 1, 1, 1)

            # Create shape for expanding scale/zero_point
            shape = [1] * ndim
            shape[axis] = scale.shape[0]

            scale_expanded = scale.reshape(shape).to(torch.float)
            zero_point_expanded = zero_point.reshape(shape).to(torch.float)

            # Dequantize to float
            dequantized = (int8_data_int32.to(torch.float) - zero_point_expanded) * scale_expanded

            # Now requantize with proper scale/zero_point
            self._quantized_weight = torch.quantize_per_channel(
                dequantized,
                scales=scale,
                zero_points=zero_point.int(),
                axis=axis,
                dtype=torch.qint8
            )

        # Call parent implementation for other parameters (bias, etc.)
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs
        )


def quantize_linear_module(
    module: nn.Linear,
    dtype: torch.dtype = torch.qint8,
    symmetric: bool = False,
) -> QuantizedLinear:
    """
    Quantize an nn.Linear module to a QuantizedLinear.

    This function creates a new QuantizedLinear module with quantized
    weights copied from the source Linear module. The original module
    is not modified.

    Args:
        module: Source nn.Linear module to quantize.
        dtype: Target quantization dtype. Only torch.qint8 is supported.
        symmetric: If True, use symmetric quantization (zero_point = 0).
                   If False, use asymmetric quantization. Default is False.

    Returns:
        A new QuantizedLinear instance with quantized weights.

    Raises:
        TypeError: If module is not an nn.Linear instance.

    Examples:
        >>> import torch.nn as nn
        >>> from mono_quant.modules import quantize_linear_module
        >>> linear = nn.Linear(128, 64)
        >>> q_linear = quantize_linear_module(linear, symmetric=False)
        >>> assert isinstance(q_linear, QuantizedLinear)
    """
    if not isinstance(module, nn.Linear):
        raise TypeError(
            f"Expected nn.Linear module, got {type(module).__name__}. "
            f"quantize_linear_module only supports nn.Linear layers."
        )

    # Create new QuantizedLinear with same configuration
    q_module = QuantizedLinear(
        in_features=module.in_features,
        out_features=module.out_features,
        bias=module.bias is not None,
        dtype=dtype,
        symmetric=symmetric,
    )

    # Quantize the weight using per-channel quantization (axis=0 for output channels)
    q_module._quantized_weight = quantize_weight_int8(
        module.weight.data, symmetric=symmetric, axis=0
    )

    # Copy bias if present (bias is not quantized)
    if module.bias is not None:
        q_module.bias.data = module.bias.data.clone()

    return q_module


class QuantizedConv2d(nn.Module):
    """
    A Conv2d layer with quantized weights for inference.

    This module wraps quantized weights (typically INT8) and handles
    dequantization during the forward pass. The weights are quantized
    once during initialization and cached for efficient inference.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        kernel_size: Size of the convolving kernel.
        stride: Stride of the convolution.
        padding: Zero-padding added to both sides of the input.
        dilation: Spacing between kernel elements.
        groups: Number of blocked connections from input to output.
        bias: If True, adds a learnable bias to the output.
        padding_mode: 'zeros', 'reflect', 'replicate' or 'circular'.
        dtype: Quantization dtype (torch.qint8 only supported).
        symmetric: If True, use symmetric quantization (zero_point = 0).

    Attributes:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        kernel_size: Size of the convolving kernel.
        _quantized_weight: Cached quantized weight tensor.
        bias: Optional bias parameter.

    Examples:
        >>> import torch.nn as nn
        >>> from mono_quant.modules import QuantizedConv2d
        >>> conv = QuantizedConv2d(3, 64, kernel_size=3)
        >>> x = torch.randn(1, 3, 32, 32)
        >>> y = conv(x)
        >>> assert y.shape == (1, 64, 30, 30)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, Tuple[int, int]] = 0,
        dilation: Union[int, Tuple[int, int]] = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = 'zeros',
        dtype: torch.dtype = torch.qint8,
        symmetric: bool = False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.padding_mode = padding_mode
        self.dtype = dtype
        self.symmetric = symmetric

        # Store quantized weight (cached after quantization)
        self._quantized_weight: Optional[torch.Tensor] = None

        # Bias is stored as-is (not quantized)
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter("bias", None)

        # Placeholder for the original weight shape
        # Shape: (out_channels, in_channels, kernel_h, kernel_w)
        if isinstance(kernel_size, int):
            kernel_h = kernel_w = kernel_size
        else:
            kernel_h, kernel_w = kernel_size

        self._weight_shape = (
            out_channels,
            in_channels // groups,
            kernel_h,
            kernel_w
        )

    @classmethod
    def from_conv2d(
        cls,
        module: nn.Conv2d,
        symmetric: bool = False,
    ) -> "QuantizedConv2d":
        """
        Create a QuantizedConv2d from an existing nn.Conv2d module.

        This factory method quantizes the weights from the source Conv2d
        module and creates a new QuantizedConv2d instance.

        Args:
            module: Source nn.Conv2d module to quantize.
            symmetric: If True, use symmetric quantization. Default is False.

        Returns:
            A new QuantizedConv2d instance with quantized weights.

        Examples:
            >>> import torch.nn as nn
            >>> from mono_quant.modules import QuantizedConv2d
            >>> conv = nn.Conv2d(3, 64, kernel_size=3)
            >>> q_conv = QuantizedConv2d.from_conv2d(conv, symmetric=False)
        """
        # Create new QuantizedConv2d with same configuration
        q_conv = cls(
            in_channels=module.in_channels,
            out_channels=module.out_channels,
            kernel_size=module.kernel_size,
            stride=module.stride,
            padding=module.padding,
            dilation=module.dilation,
            groups=module.groups,
            bias=module.bias is not None,
            padding_mode=module.padding_mode,
            symmetric=symmetric,
        )

        # Quantize the weight using per-channel quantization (axis=0 for output channels)
        q_conv._quantized_weight = quantize_weight_int8(
            module.weight.data, symmetric=symmetric, axis=0
        )

        # Copy bias if present
        if module.bias is not None:
            q_conv.bias.data = module.bias.data.clone()

        return q_conv

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with dequantized weights.

        Args:
            input: Input tensor of shape (N, C_in, H, W).

        Returns:
            Output tensor of shape (N, C_out, H_out, W_out).
        """
        # Lazy quantization check
        if self._quantized_weight is None:
            raise RuntimeError(
                "QuantizedConv2d has no quantized weights. "
                "Use from_conv2d() or set _quantized_weight directly."
            )

        # Dequantize weight for computation
        weight = dequantize_weight(self._quantized_weight)

        # Compute convolution
        output = F.conv2d(
            input, weight, self.bias, self.stride,
            self.padding, self.dilation, self.groups
        )

        return output

    @property
    def weight(self) -> Optional[torch.Tensor]:
        """
        Return the dequantized weight (for compatibility).

        This property provides compatibility with code that expects
        a weight attribute on Conv2d modules.

        Returns:
            Dequantized weight tensor, or None if not quantized yet.
        """
        if self._quantized_weight is None:
            return None
        return dequantize_weight(self._quantized_weight)

    def extra_repr(self) -> str:
        """Return extra representation string."""
        return (
            f"in_channels={self.in_channels}, "
            f"out_channels={self.out_channels}, "
            f"kernel_size={self.kernel_size}, "
            f"dtype={self.dtype}, "
            f"symmetric={self.symmetric}, "
            f"bias={self.bias is not None}"
        )

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        """
        Custom serialization to include quantized weight metadata.

        This method saves the quantized weight along with its scale and
        zero_point information, enabling proper reconstruction during loading.

        Args:
            destination: State dictionary to save to.
            prefix: Prefix for keys in state dict.
            keep_vars: Whether to keep variables as-is.
        """
        # Call parent implementation first
        super()._save_to_state_dict(destination, prefix, keep_vars)

        # Save quantized weight metadata if present
        if self._quantized_weight is not None:
            # Save the int8 representation
            destination[prefix + "_quantized_weight_int8"] = (
                self._quantized_weight.int_repr()
            )
            # Save per-channel scales
            destination[prefix + "_scale"] = (
                self._quantized_weight.q_per_channel_scales()
            )
            # Save per-channel zero_points
            destination[prefix + "_zero_point"] = (
                self._quantized_weight.q_per_channel_zero_points()
            )
            # Save axis
            destination[prefix + "_axis"] = torch.tensor(
                self._quantized_weight.q_per_channel_axis()
            )

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict,
        missing_keys, unexpected_keys, error_msgs
    ):
        """
        Custom deserialization to restore quantized weights.

        This method reconstructs the quantized weight from saved metadata.

        Args:
            state_dict: State dictionary to load from.
            prefix: Prefix for keys in state dict.
            local_metadata: Local metadata.
            strict: Whether to enforce strict key matching.
            missing_keys: List of missing keys (will be populated).
            unexpected_keys: List of unexpected keys (will be populated).
            error_msgs: List of error messages (will be populated).
        """
        # Check if we have quantized weight metadata
        quantized_key = prefix + "_quantized_weight_int8"
        scale_key = prefix + "_scale"
        zero_point_key = prefix + "_zero_point"
        axis_key = prefix + "_axis"

        # If all quantized metadata is present, reconstruct the quantized weight
        if all(k in state_dict for k in [quantized_key, scale_key, zero_point_key, axis_key]):
            # Extract metadata from state_dict
            int8_data = state_dict.pop(quantized_key)
            scale = state_dict.pop(scale_key)
            zero_point = state_dict.pop(zero_point_key)
            axis = state_dict.pop(axis_key).item()

            # Convert int8_data back to the original int8 dtype
            # int_repr() returns an int8 tensor, we need to keep it as int8
            int8_data = int8_data.to(torch.int8)

            # Reconstruct quantized weight using dequantize-requantize approach
            # This ensures the quantized tensor is reconstructed correctly

            # For per-channel quantization, apply scale/zp per channel
            # Convert to int32 to avoid overflow
            int8_data_int32 = int8_data.to(torch.int32)

            # Dequantize: float_value = (int8_value - zero_point) * scale
            # We need to properly reshape scale/zero_point for broadcasting
            # based on the weight shape and axis

            # Get the number of dimensions in the weight
            ndim = int8_data.dim()

            # Reshape scale and zero_point to broadcast correctly
            # For axis=0 (per-channel along first dimension):
            # - Linear: (out_features, in_features) -> scale/zero_point: (out_features, 1)
            # - Conv2d: (out_channels, in_channels, h, w) -> scale/zero_point: (out_channels, 1, 1, 1)

            # Create shape for expanding scale/zero_point
            shape = [1] * ndim
            shape[axis] = scale.shape[0]

            scale_expanded = scale.reshape(shape).to(torch.float)
            zero_point_expanded = zero_point.reshape(shape).to(torch.float)

            # Dequantize to float
            dequantized = (int8_data_int32.to(torch.float) - zero_point_expanded) * scale_expanded

            # Now requantize with proper scale/zero_point
            self._quantized_weight = torch.quantize_per_channel(
                dequantized,
                scales=scale,
                zero_points=zero_point.int(),
                axis=axis,
                dtype=torch.qint8
            )

        # Call parent implementation for other parameters (bias, etc.)
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs
        )


def quantize_conv2d_module(
    module: nn.Conv2d,
    dtype: torch.dtype = torch.qint8,
    symmetric: bool = False,
) -> QuantizedConv2d:
    """
    Quantize an nn.Conv2d module's weights using per-channel quantization.

    This function creates a QuantizedConv2d module with INT8 quantized weights.
    The weights are quantized per output channel (axis=0), which is the
    standard approach for Conv2d layers. Bias is preserved but not quantized.

    Args:
        module: Source nn.Conv2d module to quantize.
        dtype: Target quantization dtype. Only torch.qint8 is supported.
        symmetric: If True, use symmetric quantization (zero_point = 0).
                   If False, use asymmetric quantization. Default is False.

    Returns:
        A new QuantizedConv2d instance with INT8 quantized weights.

    Raises:
        TypeError: If module is not an nn.Conv2d instance.

    Examples:
        >>> import torch.nn as nn
        >>> from mono_quant.modules import quantize_conv2d_module
        >>> conv = nn.Conv2d(3, 64, kernel_size=3)
        >>> q_conv = quantize_conv2d_module(conv, symmetric=False)
        >>> assert isinstance(q_conv, QuantizedConv2d)
        >>> assert q_conv._quantized_weight.dtype == torch.qint8
    """
    if not isinstance(module, nn.Conv2d):
        raise TypeError(
            f"Expected nn.Conv2d module, got {type(module).__name__}. "
            f"quantize_conv2d_module only supports nn.Conv2d layers."
        )

    # Use the factory method to create QuantizedConv2d
    return QuantizedConv2d.from_conv2d(module, symmetric=symmetric)


class QuantizedLinearInt4(nn.Module):
    """
    A Linear layer with INT4 quantized weights for inference.

    This module stores weights in packed INT4 format (2 values per int8 byte)
    and dequantizes them during the forward pass. Group-wise scaling is used
    to maintain accuracy at 4-bit precision.

    Args:
        in_features: Number of input features.
        out_features: Number of output features.
        packed_weight: Packed INT4 weights stored as int8 tensor.
        scales: Per-group scale factors for dequantization.
        zero_points: Per-group zero-points for dequantization.
        group_size: Number of channels per group. Default is 128.
        bias: Optional bias tensor. Default is None.

    Attributes:
        in_features: Number of input features.
        out_features: Number of output features.
        group_size: Number of channels per quantization group.
        bias: Optional bias parameter (not quantized).

    Examples:
        >>> import torch
        >>> import torch.nn as nn
        >>> from mono_quant.modules.linear import quantize_linear_module_int4
        >>> linear = nn.Linear(128, 256)
        >>> q_linear = quantize_linear_module_int4(linear, group_size=128)
        >>> x = torch.randn(32, 128)
        >>> y = q_linear(x)
        >>> assert y.shape == (32, 256)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        packed_weight: torch.Tensor,
        scales: torch.Tensor,
        zero_points: torch.Tensor,
        group_size: int = 128,
        bias: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.group_size = group_size

        # Register as buffers (not parameters) since these are fixed quantization values
        self.register_buffer("_packed_weight", packed_weight)
        self.register_buffer("_scales", scales)
        self.register_buffer("_zero_points", zero_points)

        # Bias is stored as a parameter if provided
        if bias is not None:
            self.bias = nn.Parameter(bias.clone())
        else:
            self.register_parameter("bias", None)

    @classmethod
    def from_float(
        cls,
        module: nn.Linear,
        group_size: int = 128,
        symmetric: bool = True,
    ) -> "QuantizedLinearInt4":
        """
        Create a QuantizedLinearInt4 from an existing nn.Linear module.

        This factory method quantizes the weights from the source Linear
        module and creates a new QuantizedLinearInt4 instance.

        Args:
            module: Source nn.Linear module to quantize.
            group_size: Number of channels per group. Default is 128.
            symmetric: If True, use symmetric quantization. Default is True.

        Returns:
            A new QuantizedLinearInt4 instance with quantized weights.

        Examples:
            >>> import torch.nn as nn
            >>> from mono_quant.modules.linear import QuantizedLinearInt4
            >>> linear = nn.Linear(128, 256)
            >>> q_linear = QuantizedLinearInt4.from_float(linear, group_size=128)
        """
        # Local import to avoid circular dependencies
        from mono_quant.core.quantizers import quantize_weight_int4

        # Get weight from module
        weight = module.weight.data

        # Quantize to INT4 with group-wise scaling
        packed, scales, zero_points = quantize_weight_int4(
            weight, group_size=group_size, symmetric=symmetric, axis=0
        )

        # Create new QuantizedLinearInt4 instance
        q_linear = cls(
            in_features=module.in_features,
            out_features=module.out_features,
            packed_weight=packed,
            scales=scales,
            zero_points=zero_points,
            group_size=group_size,
            bias=module.bias.data.clone() if module.bias is not None else None,
        )

        return q_linear

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with dequantized weights.

        The packed INT4 weights are unpacked and dequantized using per-group
        scale and zero-point values before computing the linear transformation.

        Args:
            x: Input tensor of shape (*, in_features).

        Returns:
            Output tensor of shape (*, out_features).
        """
        # Local imports to avoid circular dependencies
        from mono_quant.core.mappers import _unpack_int8_to_int4

        # Unpack and dequantize weights
        weight = self._dequantize_weight()

        # Compute linear transformation
        output = F.linear(x, weight, self.bias)

        return output

    def _dequantize_weight(self) -> torch.Tensor:
        """
        Dequantize the packed INT4 weights back to float32.

        Unpacks the INT4 values from int8 storage and applies per-group
        dequantization using scale and zero-point.

        Returns:
            Dequantized weight tensor with shape (out_features, in_features).
        """
        from mono_quant.core.mappers import _unpack_int8_to_int4

        # Calculate total number of INT4 values
        num_elements = self.out_features * self.in_features

        # Unpack INT4 values from int8 storage
        int4_values = _unpack_int8_to_int4(self._packed_weight, num_elements)

        # Reshape to (out_features, in_features)
        weight = int4_values.reshape(self.out_features, self.in_features).float()

        # Apply per-group dequantization
        num_groups = self._scales.shape[0]

        for g in range(num_groups):
            start_idx = g * self.group_size
            end_idx = min(start_idx + self.group_size, self.out_features)

            # Get scale and zero_point for this group
            scale = self._scales[g].item()
            zp = self._zero_points[g].item()

            # Dequantize: (int4 - zero_point) * scale
            weight[start_idx:end_idx, :] = (weight[start_idx:end_idx, :] - zp) * scale

        return weight

    @property
    def weight(self) -> torch.Tensor:
        """
        Return the dequantized weight (for compatibility).

        This property provides compatibility with code that expects
        a weight attribute on Linear modules.

        Returns:
            Dequantized weight tensor.
        """
        return self._dequantize_weight()

    def extra_repr(self) -> str:
        """Return extra representation string."""
        return (
            f"in_features={self.in_features}, "
            f"out_features={self.out_features}, "
            f"group_size={self.group_size}, "
            f"bias={self.bias is not None}"
        )


def quantize_linear_module_int4(
    module: nn.Linear,
    group_size: int = 128,
    symmetric: bool = True,
) -> QuantizedLinearInt4:
    """
    Quantize an nn.Linear module to a QuantizedLinearInt4.

    This function creates a new QuantizedLinearInt4 module with INT4
    quantized weights copied from the source Linear module. The original
    module is not modified.

    INT4 quantization provides 2x additional compression over INT8 by
    storing weights in packed format (2 INT4 values per int8 byte).

    Args:
        module: Source nn.Linear module to quantize.
        group_size: Number of channels per quantization group. Default is 128.
        symmetric: If True, use symmetric quantization (zero_point = 0).
                   If False, use asymmetric quantization. Default is True.

    Returns:
        A new QuantizedLinearInt4 instance with quantized weights.

    Raises:
        TypeError: If module is not an nn.Linear instance.

    Examples:
        >>> import torch.nn as nn
        >>> from mono_quant.modules.linear import quantize_linear_module_int4
        >>> linear = nn.Linear(128, 256)
        >>> q_linear = quantize_linear_module_int4(linear, group_size=128)
        >>> assert isinstance(q_linear, QuantizedLinearInt4)
        >>> # Forward pass
        >>> x = torch.randn(32, 128)
        >>> y = q_linear(x)
        >>> assert y.shape == (32, 256)
    """
    if not isinstance(module, nn.Linear):
        raise TypeError(
            f"Expected nn.Linear module, got {type(module).__name__}. "
            f"quantize_linear_module_int4 only supports nn.Linear layers."
        )

    return QuantizedLinearInt4.from_float(
        module, group_size=group_size, symmetric=symmetric
    )
