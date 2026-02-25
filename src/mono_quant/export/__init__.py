"""ONNX export entry point for mono-quant."""

from pathlib import Path
from typing import Any, Optional, Union

import torch
import torch.nn as nn


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
