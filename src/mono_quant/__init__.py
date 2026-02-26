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

__version__ = "2.0.0"

from mono_quant.api import quantize
from mono_quant.config import QuantizationConfig
from mono_quant.core import dynamic_quantize
from mono_quant.core.quantizers import (
    dequantize_model,
    revert_to_standard_modules,
    static_quantize,
)
from mono_quant.io import load_model, save_model
from mono_quant.io.validation import (
    ValidationResult,
    check_accuracy_warnings,
    validate_quantization,
)

def export_model(model, path, format=None, **kwargs):  # noqa: A002
    """Export a model to any supported format via the unified orchestrator.

    Args:
        model:  Any nn.Module (quantized or plain FP32).
        path:   Output path. Format inferred from extension if *format* is None.
        format: ``"onnx"``, ``"gptq"``, or ``"gguf"``. Optional.
        **kwargs: Format-specific options (opset, group_size, sym, architecture…).

    See Also:
        :func:`mono_quant.export.orchestrator.export_model` for full docs.
    """
    from mono_quant.export.orchestrator import export_model as _export_model
    return _export_model(model, path, format=format, **kwargs)


def list_formats():
    """Return a dict of supported export formats and their descriptions."""
    from mono_quant.export.orchestrator import list_formats as _list_formats
    return _list_formats()


def export_to_gptq(model, path, group_size=128, sym=False, **kwargs):
    """Export a model to GPTQ INT4 format (AutoGPTQ V1 / vLLM-compatible).

    Args:
        model: Any nn.Module (quantized or plain FP32).
        path: Output directory. Created if it does not exist.
        group_size: Columns per quantization group. Must divide in_features.
        sym: Symmetric quantization if True, asymmetric if False.

    Raises:
        TypeError: If model is not an nn.Module.
        ValueError: If a layer's dimensions are incompatible with group_size.
    """
    from mono_quant.export import export_to_gptq as _export_gptq
    return _export_gptq(model, path, group_size=group_size, sym=sym)


def export_to_gguf(model, path, **kwargs):
    """Export a model to GGUF format for use with llama.cpp.

    Requires optional GGUF dependencies for validation:
        pip install mono-quant[gguf]

    Args:
        model: Any nn.Module (quantized or plain FP32).
        path:  Output directory. model.gguf written inside.
        **kwargs: quantization_type, architecture, config_path, model_params.

    Raises:
        TypeError: If model is not an nn.Module.
    """
    from mono_quant.export import export_to_gguf as _export
    return _export(model, path, **kwargs)


def export_to_onnx(model, path, **kwargs):
    """Export a quantized model to ONNX format with QDQ nodes.

    Requires optional ONNX dependencies:
        pip install mono-quant[onnx]

    Args:
        model: Quantized nn.Module to export.
        path: Output path for the .onnx file.
        **kwargs: opset, dummy_input, validate — see mono_quant.export for details.

    Raises:
        ImportError: If onnx or onnxruntime is not installed.
    """
    from mono_quant.export import export_to_onnx as _export
    return _export(model, path, **kwargs)


__all__ = [
    # Version
    "__version__",
    # Unified API
    "quantize",
    "export_model",
    "list_formats",
    "export_to_onnx",
    "export_to_gptq",
    "export_to_gguf",
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
