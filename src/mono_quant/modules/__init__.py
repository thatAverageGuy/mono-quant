"""Quantized PyTorch module replacements."""

from mono_quant.modules.linear import (
    QuantizedLinear,
    QuantizedLinearInt4,
    quantize_linear_module,
    quantize_linear_module_int4,
    quantize_conv2d_module,
)

__all__ = [
    "QuantizedLinear",
    "QuantizedLinearInt4",
    "quantize_linear_module",
    "quantize_linear_module_int4",
    "quantize_conv2d_module",
]
