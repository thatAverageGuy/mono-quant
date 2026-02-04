"""Tests for mono-quant API."""

import torch
import torch.nn as nn

from mono_quant import QuantizationConfig, quantize


class SimpleModel(nn.Module):
    """Simple model for testing."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 5)

    def forward(self, x):
        return self.linear(x)


def test_quantize_dynamic_int8():
    """Test dynamic INT8 quantization."""
    model = SimpleModel()
    result = quantize(model, bits=8, dynamic=True)

    assert result.success
    assert result.model is not None
    assert result.info is not None
    # Note: original_size_mb and other validation metrics are None for dynamic quantization
    # because validation doesn't run automatically. Use static quantization for full metrics.


def test_quantize_fp16():
    """Test FP16 quantization."""
    model = SimpleModel()
    result = quantize(model, bits=16, dynamic=True)

    assert result.success
    assert result.model is not None


def test_quantization_config():
    """Test QuantizationConfig."""
    config = QuantizationConfig(
        dtype=torch.qint8,
        symmetric=True,
        per_channel=True
    )

    assert config.dtype == torch.qint8
    assert config.symmetric is True
    assert config.per_channel is True


def test_version():
    """Test that version is defined."""
    import mono_quant
    assert mono_quant.__version__ is not None
    assert len(mono_quant.__version__.split('.')) >= 2


def test_public_api_exports():
    """Test that public API is exported."""
    import mono_quant

    # Check key exports
    assert hasattr(mono_quant, 'quantize')
    assert hasattr(mono_quant, 'dynamic_quantize')
    assert hasattr(mono_quant, 'static_quantize')
    assert hasattr(mono_quant, 'save_model')
    assert hasattr(mono_quant, 'load_model')
    assert hasattr(mono_quant, 'QuantizationConfig')
