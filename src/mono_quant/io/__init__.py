"""Model input/output handling.

This module provides comprehensive I/O capabilities for quantized models:

## Save/Load Functions

- **save_model**: Save model with automatic format detection (.safetensors, .pt, .pth)
- **load_model**: Load model with automatic format detection
- **save_pytorch**: Save model in PyTorch format (.pt/.pth)
- **load_pytorch**: Load model from PyTorch format
- **save_safetensors**: Save model in Safetensors format (.safetensors)
- **load_safetensors**: Load model from Safetensors format

## Validation Metrics

- **ValidationResult**: Dataclass containing validation metrics
- **calculate_model_size**: Calculate model memory size and parameter count
- **calculate_sqnr**: Calculate Signal-to-Quantization-Noise Ratio
- **validate_quantization**: Run comprehensive validation on quantized models

## Internal Handlers

- **_prepare_model**: Prepare model from nn.Module or state_dict
- **_detect_input_format**: Detect input format (module vs state_dict vs dataloader)
- **_validate_model**: Validate model structure for quantization

## Metadata

- **_build_metadata**: Build metadata dictionary for Safetensors format

Example:
    >>> from mono_quant.io import save_model, load_model
    >>> model = nn.Linear(128, 256)
    >>> save_model(model, "quantized.safetensors")
    >>> loaded = load_model("quantized.safetensors")
"""

# Format handlers (save/load)
from mono_quant.io.formats import (
    QuantizationInfo,
    _build_metadata,
    load_model,
    load_pytorch,
    load_safetensors,
    save_model,
    save_pytorch,
    save_safetensors,
)

# Input handlers (internal)
from mono_quant.io.handlers import (
    _detect_input_format,
    _prepare_model,
    _validate_model,
)

# Validation metrics
from mono_quant.io.validation import (
    ValidationResult,
    calculate_model_size,
    calculate_sqnr,
    validate_quantization,
)

__all__ = [
    # Save/Load functions
    "save_model",
    "load_model",
    "save_pytorch",
    "load_pytorch",
    "save_safetensors",
    "load_safetensors",
    # Validation
    "ValidationResult",
    "calculate_model_size",
    "calculate_sqnr",
    "validate_quantization",
    # Metadata
    "QuantizationInfo",
    "_build_metadata",
    # Internal handlers
    "_prepare_model",
    "_detect_input_format",
    "_validate_model",
]
