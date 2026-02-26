"""Thin wrapper around GPTQExporter — mirrors the onnx_impl.py pattern."""

from pathlib import Path
from typing import Any, Union

import torch.nn as nn

from mono_quant.export.gptq import GPTQExporter


def export_to_gptq_impl(
    model: nn.Module,
    path: Union[str, Path],
    group_size: int = 128,
    sym: bool = False,
    **kwargs: Any,
) -> None:
    """Export a model to GPTQ INT4 format.

    Delegates to GPTQExporter.export().

    Args:
        model: Any nn.Module (quantized or plain FP32).
        path: Output directory.
        group_size: Columns per quantization group.
        sym: Symmetric quantization if True, asymmetric if False.
        **kwargs: Reserved for future use.
    """
    GPTQExporter().export(model, path, group_size=group_size, sym=sym)
