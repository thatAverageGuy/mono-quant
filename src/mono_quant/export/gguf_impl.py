"""Thin wrapper around GGUFExporter — mirrors the gptq_impl.py pattern."""

from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch.nn as nn

from mono_quant.export.gguf import GGUFExporter


def export_to_gguf_impl(
    model: nn.Module,
    path: Union[str, Path],
    quantization_type: str = "q4_k_s",
    architecture: Optional[str] = None,
    config_path: Optional[Union[str, Path]] = None,
    model_params: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> None:
    """Export a model to GGUF format.

    Delegates to GGUFExporter.export().

    Args:
        model:             Any nn.Module (quantized or plain FP32).
        path:              Output directory. model.gguf written inside.
        quantization_type: Quantization format. Currently only "q4_k_s".
        architecture:      GGUF architecture string. Auto-detected if None.
        config_path:       Path to HuggingFace config.json. Optional.
        model_params:      Dict of model hyperparameters. Overrides config_path.
        **kwargs:          Reserved for future use.
    """
    GGUFExporter().export(
        model,
        path,
        quantization_type=quantization_type,
        architecture=architecture,
        config_path=config_path,
        model_params=model_params,
    )
