"""Unified Python API for mono-quant.

This module provides the high-level Python API for model quantization.
It exposes a single `quantize()` function that dispatches to dynamic or
static quantization based on parameters, along with result types and
exception classes for error handling.

## Quick Start

    >>> from mono_quant.api import quantize
    >>> import torch.nn as nn
    >>> model = nn.Sequential(nn.Linear(128, 256), nn.ReLU())
    >>> result = quantize(model, bits=8, dynamic=True)
    >>> if result:
    ...     result.save("quantized.safetensors")

## Public API

### Quantization
- **quantize**: Unified quantization function (dispatches to dynamic/static)

### Result Types
- **QuantizationResult**: Result dataclass with model, info, and convenience methods

### Exceptions
- **MonoQuantError**: Base exception for all mono-quant errors
- **QuantizationError**: Quantization failures
- **ValidationError**: Validation failures
- **ConfigurationError**: Invalid configuration
- **InputError**: Invalid input

## Examples

Dynamic quantization (simple, no calibration):
    >>> from mono_quant.api import quantize
    >>> result = quantize(model, bits=8, dynamic=True)

Static quantization (better accuracy, needs calibration):
    >>> calibration_data = [torch.randn(32, 128) for _ in range(100)]
    >>> result = quantize(model, bits=8, calibration_data=calibration_data)
    >>> print(f"SQNR: {result.info.sqnr_db:.2f} dB")

INT4 quantization (maximum compression):
    >>> result = quantize(model, bits=4, calibration_data=calibration_data)

FP16 quantization (simple, faster):
    >>> result = quantize(model, bits=16, dynamic=True)

From file path:
    >>> result = quantize("model.pt", bits=8, dynamic=True)

Error handling with suggestions:
    >>> from mono_quant.api import QuantizationError, ConfigurationError
    >>> try:
    ...     result = quantize(model, bits=5)
    ... except ConfigurationError as e:
    ...     print(f"Error: {e}")
    ...     # The error message includes a suggestion for fixing it
"""

# Quantization function
# Exceptions
from .exceptions import (
    ConfigurationError,
    InputError,
    MonoQuantError,
    QuantizationError,
    ValidationError,
)
from .quantize import quantize

# Result types
from .result import QuantizationResult

__all__ = [
    # Quantization
    "quantize",
    # Result types
    "QuantizationResult",
    # Exceptions
    "MonoQuantError",
    "QuantizationError",
    "ValidationError",
    "ConfigurationError",
    "InputError",
]
