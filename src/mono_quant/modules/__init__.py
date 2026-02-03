"""Quantized PyTorch module replacements."""

from mono_quant.modules.linear import (
    QuantizedLinear,
    quantize_linear_module,
    quantize_conv2d_module,
)

__all__ = [
    "QuantizedLinear",
    "quantize_linear_module",
    "quantize_conv2d_module",
]
