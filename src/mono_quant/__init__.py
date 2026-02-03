"""
Mono Quant - Simple, reliable model quantization with minimal dependencies.

This package provides model-agnostic quantization for PyTorch models with only
torch as a required dependency.

## Public API

### Quantization Functions

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

    >>> import torch.nn as nn
    >>> from mono_quant import static_quantize, save_model
    >>> model = nn.Sequential(
    ...     nn.Linear(128, 256),
    ...     nn.ReLU(),
    ...     nn.Linear(256, 10)
    ... )
    >>> calibration_data = [torch.randn(32, 128) for _ in range(100)]
    >>> q_model, info = static_quantize(model, calibration_data)
    >>> print(f"SQNR: {info.sqnr_db:.2f} dB")
    >>> print(f"Compression: {info.compression_ratio:.2f}x")
    >>> save_model(q_model, "quantized.safetensors")

Dynamic quantization (simpler, no calibration needed):

    >>> from mono_quant import dynamic_quantize
    >>> model = nn.Sequential(nn.Linear(128, 256), nn.ReLU())
    >>> q_model, skipped = dynamic_quantize(model, dtype=torch.qint8)
"""

__version__ = "0.1.0"

# Configuration
from mono_quant.config import QuantizationConfig

# Quantization functions
from mono_quant.core import dynamic_quantize
from mono_quant.core.quantizers import static_quantize

# Model I/O
from mono_quant.io import save_model, load_model

# Validation
from mono_quant.io.validation import ValidationResult, validate_quantization, check_accuracy_warnings

__all__ = [
    # Version
    "__version__",
    # Configuration
    "QuantizationConfig",
    # Quantization functions
    "dynamic_quantize",
    "static_quantize",
    # Model I/O
    "save_model",
    "load_model",
    # Validation
    "ValidationResult",
    "validate_quantization",
    "check_accuracy_warnings",
]
