"""Export entry points for mono-quant (ONNX, GPTQ, GGUF, unified orchestrator)."""

from pathlib import Path
from typing import Any, Optional, Union

import torch
import torch.nn as nn


def export_to_gptq(
    model: nn.Module,
    path: Union[str, Path],
    group_size: int = 128,
    sym: bool = False,
    **kwargs: Any,
) -> None:
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
    from mono_quant.export.gptq_impl import export_to_gptq_impl

    export_to_gptq_impl(model, path, group_size=group_size, sym=sym)


def export_to_gguf(
    model: nn.Module,
    path: Union[str, Path],
    quantization_type: str = "q4_k_s",
    **kwargs: Any,
) -> None:
    """Export a model to GGUF format for use with llama.cpp.

    Args:
        model:             Any nn.Module (quantized or plain FP32).
        path:              Output directory. model.gguf written inside.
        quantization_type: Quantization format. Currently only "q4_k_s".
        **kwargs:          architecture, config_path, model_params — see GGUFExporter.

    Raises:
        TypeError: If model is not an nn.Module.
    """
    from mono_quant.export.gguf_impl import export_to_gguf_impl

    export_to_gguf_impl(
        model,
        path,
        quantization_type=quantization_type,
        **{k: v for k, v in kwargs.items() if k in ("architecture", "config_path", "model_params")},
    )


def export_to_onnx(
    model: nn.Module,
    path: Union[str, Path],
    opset: int = 14,
    dummy_input: Optional[torch.Tensor] = None,
    validate: str = "none",
    **kwargs: Any,
) -> None:
    """Export a quantized PyTorch model to ONNX with QDQ nodes.

    Args:
        model: Quantized nn.Module to export.
        path: Output path for the .onnx file.
        opset: ONNX opset version. Default is 14.
        dummy_input: Optional dummy input tensor. If None, auto-inferred.
        validate: Validation level — "none", "load", or "full".
        **kwargs: Additional arguments passed to the exporter.

    Raises:
        ImportError: If onnx or onnxruntime is not installed.
    """
    try:
        import onnx  # noqa: F401
    except ImportError:
        raise ImportError(
            "ONNX export requires additional dependencies. "
            "Install them with: pip install mono-quant[onnx]"
        )

    from mono_quant.export.onnx_impl import export_to_onnx_impl

    export_to_onnx_impl(
        model=model,
        path=path,
        opset=opset,
        dummy_input=dummy_input,
        validate=validate,
        **kwargs,
    )


# Unified orchestrator — re-exported for convenience
from mono_quant.export.orchestrator import export_model, list_formats  # noqa: E402

__all__ = [
    "export_to_gptq",
    "export_to_gguf",
    "export_to_onnx",
    "export_model",
    "list_formats",
]
