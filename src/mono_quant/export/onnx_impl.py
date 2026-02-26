"""Thin module-level wrapper around ONNXExporter."""

from pathlib import Path
from typing import Any, Optional, Union

import torch
import torch.nn as nn

from mono_quant.export.onnx import ONNXExporter


def export_to_onnx_impl(
    model: nn.Module,
    path: Union[str, Path],
    opset: int = 14,
    dummy_input: Optional[torch.Tensor] = None,
    validate: str = "none",
    dynamo: bool = False,
    **kwargs: Any,
) -> None:
    """Export a quantized model to ONNX format.

    Delegates to ONNXExporter.export().

    Args:
        model: Quantized nn.Module to export.
        path: Output path for the .onnx file.
        opset: ONNX opset version. Default 14.
        dummy_input: Optional dummy input tensor. Auto-inferred if None.
        validate: Validation level — "none", "load", or "full".
        dynamo: Use FX/dynamo tracing. Handles transformer models. Default False.
        **kwargs: Reserved for future use.
    """
    exporter = ONNXExporter()
    exporter.export(
        model=model,
        path=path,
        opset=opset,
        dummy_input=dummy_input,
        validate=validate,
        dynamo=dynamo,
    )
