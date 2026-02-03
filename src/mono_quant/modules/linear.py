"""
Quantized Linear and Conv2d module implementations.

This module provides quantized replacements for PyTorch's nn.Linear and
nn.Conv2d layers. The QuantizedLinear module stores quantized weights
and dequantizes them during forward pass for inference.
"""

import copy
from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from mono_quant.core.quantizers import quantize_weight_int8, dequantize_weight


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


def quantize_conv2d_module(
    module: nn.Conv2d,
    dtype: torch.dtype = torch.qint8,
    symmetric: bool = False,
) -> nn.Conv2d:
    """
    Quantize an nn.Conv2d module's weights using per-channel quantization.

    This function creates a new nn.Conv2d module with quantized weights.
    The weights are quantized per output channel (axis=0), which is the
    standard approach for Conv2d layers. Bias is preserved but not quantized.

    Note: This returns a standard nn.Conv2d with dequantized float32 weights.
    The weights are quantized and immediately dequantized, so the module
    uses float32 weights during inference. This is a simple approach that
    doesn't require custom Conv2d implementations.

    For true quantized inference, consider using PyTorch's optimized
    quantized Conv2d variants in future phases.

    Args:
        module: Source nn.Conv2d module to quantize.
        dtype: Target quantization dtype. Only torch.qint8 is supported.
        symmetric: If True, use symmetric quantization (zero_point = 0).
                   If False, use asymmetric quantization. Default is False.

    Returns:
        A new nn.Conv2d instance with quantized (then dequantized) weights.

    Raises:
        TypeError: If module is not an nn.Conv2d instance.

    Examples:
        >>> import torch.nn as nn
        >>> from mono_quant.modules import quantize_conv2d_module
        >>> conv = nn.Conv2d(3, 64, kernel_size=3)
        >>> q_conv = quantize_conv2d_module(conv, symmetric=False)
        >>> assert isinstance(q_conv, nn.Conv2d)
    """
    if not isinstance(module, nn.Conv2d):
        raise TypeError(
            f"Expected nn.Conv2d module, got {type(module).__name__}. "
            f"quantize_conv2d_module only supports nn.Conv2d layers."
        )

    # Quantize the weight using per-channel quantization
    # For Conv2d, axis=0 corresponds to output channels
    # Shape: (out_channels, in_channels, kernel_h, kernel_w)
    q_weight = quantize_weight_int8(
        module.weight.data, symmetric=symmetric, axis=0
    )

    # Dequantize back to float32 for standard nn.Conv2d
    # In future phases, can use quantized Conv2d for true int8 inference
    weight_dequant = dequantize_weight(q_weight)

    # Create new Conv2d with same parameters
    q_module = nn.Conv2d(
        in_channels=module.in_channels,
        out_channels=module.out_channels,
        kernel_size=module.kernel_size,
        stride=module.stride,
        padding=module.padding,
        dilation=module.dilation,
        groups=module.groups,
        bias=module.bias is not None,
        padding_mode=module.padding_mode,
    )

    # Copy quantized (dequantized) weights
    q_module.weight.data = weight_dequant

    # Copy bias if present (bias is not quantized)
    if module.bias is not None:
        q_module.bias.data = module.bias.data.clone()

    return q_module
