"""Quantized PyTorch module replacements."""

from mono_quant.modules.linear import (
    QuantizedLinear,
    QuantizedLinearInt4,
    QuantizedConv2d,
    quantize_linear_module,
    quantize_linear_module_int4,
    quantize_conv2d_module,
    convert_to_pytorch_native,
)

__all__ = [
    "QuantizedLinear",
    "QuantizedLinearInt4",
    "QuantizedConv2d",
    "quantize_linear_module",
    "quantize_linear_module_int4",
    "quantize_conv2d_module",
    "convert_to_pytorch_native",
]
