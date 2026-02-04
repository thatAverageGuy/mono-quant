"""
Mono Quant - Simple, reliable model quantization with minimal dependencies.

This package provides model-agnostic quantization for PyTorch models with only
torch and numpy as required dependencies.

## Public API

### Unified Quantization

- **quantize**: Unified quantization function (recommended)
    Single entry point that dispatches to dynamic or static quantization
    based on parameters. Accepts nn.Module, state_dict, or file path.

### Quantization Functions (Advanced)

- **dynamic_quantize**: Dynamic quantization without calibration data
    Automatically quantizes supported layers (nn.Linear, nn.Conv2d) to INT8 or FP16.

- **static_quantize**: Static quantization with calibration data
    Uses calibration data to determine optimal quantization parameters, then
    quantizes selected layers with validation.

### Model I/O

- **save_model**: Save quantized models to disk
    Supports Safetensors (.safetensors) and PyTorch (.pt/.pth) formats.

- **load_model**: Load quantized models from disk
    Auto-detects format based on file extension.

### Configuration

- **QuantizationConfig**: Configuration dataclass for quantization parameters
    Provides dtype, symmetric, and per_channel settings.

### Validation

- **ValidationResult**: Validation metrics dataclass
    Contains SQNR, model size, compression ratio, and test results.

- **validate_quantization**: Manual validation function
    Run validation on existing models with configurable failure behavior.

## Quick Start

Unified API (recommended):

    >>> import torch.nn as nn
    >>> from mono_quant import quantize
    >>> model = nn.Sequential(
    ...     nn.Linear(128, 256),
    ...     nn.ReLU(),
    ...     nn.Linear(256, 10)
    ... )
    >>> # Dynamic quantization (simple, no calibration needed)
    >>> result = quantize(model, bits=8, dynamic=True)
    >>> result.save("quantized.safetensors")

Static quantization (better accuracy, needs calibration):

    >>> calibration_data = [torch.randn(32, 128) for _ in range(100)]
    >>> result = quantize(model, bits=8, calibration_data=calibration_data)
    >>> print(f"SQNR: {result.info.sqnr_db:.2f} dB")
    >>> print(f"Compression: {result.info.compression_ratio:.2f}x")

Advanced API with direct access to quantization functions:

    >>> from mono_quant import static_quantize, save_model
    >>> q_model, info = static_quantize(model, calibration_data)
    >>> save_model(q_model, "quantized.safetensors")
"""

__version__ = "1.1"

# Unified API
from mono_quant.api import quantize

# Configuration
from mono_quant.config import QuantizationConfig

# Quantization functions
from mono_quant.core import dynamic_quantize
from mono_quant.core.quantizers import static_quantize, revert_to_standard_modules, dequantize_model

# Model I/O
from mono_quant.io import save_model, load_model

# Validation
from mono_quant.io.validation import ValidationResult, validate_quantization, check_accuracy_warnings

__all__ = [
    # Version
    "__version__",
    # Unified API
    "quantize",
    # Configuration
    "QuantizationConfig",
    # Quantization functions
    "dynamic_quantize",
    "static_quantize",
    "revert_to_standard_modules",
    "dequantize_model",
    # Model I/O
    "save_model",
    "load_model",
    # Validation
    "ValidationResult",
    "validate_quantization",
    "check_accuracy_warnings",
]
