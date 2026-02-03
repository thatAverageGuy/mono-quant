"""
Mono Quant - Simple, reliable model quantization with minimal dependencies.

This package provides model-agnostic quantization for PyTorch models with only
torch as a required dependency.

Quick example:
    >>> import torch.nn as nn
    >>> from mono_quant import dynamic_quantize
    >>> model = nn.Sequential(nn.Linear(128, 256), nn.ReLU(), nn.Linear(256, 10))
    >>> q_model, skipped = dynamic_quantize(model, dtype=torch.qint8)
    >>> print(f"Skipped {len(skipped)} layers: {skipped}")
"""

__version__ = "0.1.0"

from mono_quant.config import QuantizationConfig
from mono_quant.core import dynamic_quantize

__all__ = [
    "QuantizationConfig",
    "dynamic_quantize",
    "__version__",
]
