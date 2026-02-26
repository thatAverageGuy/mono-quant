"""Tests for the unified export orchestrator (T-026)."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from mono_quant.export.orchestrator import (
    FORMAT_MAP,
    _detect_format,
    export_model,
    list_formats,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

class _SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(8, 4)

    def forward(self, x):
        return self.linear(x)


@pytest.fixture()
def fp32_model():
    return _SimpleModel()


# ---------------------------------------------------------------------------
# _detect_format
# ---------------------------------------------------------------------------

def test_detect_format_from_onnx_extension():
    assert _detect_format("model.onnx") == "onnx"
    assert _detect_format(Path("some/dir/model.ONNX")) == "onnx"


def test_detect_format_from_gguf_extension():
    assert _detect_format("model.gguf") == "gguf"
    assert _detect_format(Path("out/model.GGUF")) == "gguf"


def test_detect_format_gptq_no_extension():
    # Bare directory → gptq
    assert _detect_format("./gptq_dir/") == "gptq"
    assert _detect_format("gptq_dir") == "gptq"
    assert _detect_format(Path("some/output")) == "gptq"


def test_detect_format_unknown_raises_ValueError():
    with pytest.raises(ValueError, match="unrecognised extension"):
        _detect_format("model.bin")


# ---------------------------------------------------------------------------
# list_formats
# ---------------------------------------------------------------------------

def test_list_formats_returns_all_three():
    fmts = list_formats()
    assert set(fmts.keys()) == {"onnx", "gptq", "gguf"}
    # Each value is a non-empty description string
    for key, desc in fmts.items():
        assert isinstance(desc, str) and desc


def test_list_formats_is_independent_copy():
    fmts = list_formats()
    fmts["new_format"] = "should not appear"
    assert "new_format" not in FORMAT_MAP


# ---------------------------------------------------------------------------
# export_model — dispatch
# ---------------------------------------------------------------------------

def test_export_model_dispatch_onnx(fp32_model, tmp_path):
    out = str(tmp_path / "model.onnx")
    with patch(
        "mono_quant.export.orchestrator._export_onnx"
    ) as mock_onnx:
        # Patch validate_export_pre to return empty list
        with patch(
            "mono_quant.export.orchestrator.validate_export_pre",
            return_value=[],
        ):
            export_model(fp32_model, out)
    mock_onnx.assert_called_once()
    args, kwargs = mock_onnx.call_args
    assert args[0] is fp32_model
    assert args[1] == out


def test_export_model_dispatch_gptq(fp32_model, tmp_path):
    out = str(tmp_path / "gptq_dir")
    with patch(
        "mono_quant.export.orchestrator._export_gptq"
    ) as mock_gptq:
        with patch(
            "mono_quant.export.orchestrator.validate_export_pre",
            return_value=[],
        ):
            export_model(fp32_model, out, format="gptq")
    mock_gptq.assert_called_once()


def test_export_model_dispatch_gguf(fp32_model, tmp_path):
    out = str(tmp_path / "gguf_dir")
    with patch(
        "mono_quant.export.orchestrator._export_gguf"
    ) as mock_gguf:
        with patch(
            "mono_quant.export.orchestrator.validate_export_pre",
            return_value=[],
        ):
            export_model(fp32_model, out, format="gguf")
    mock_gguf.assert_called_once()


def test_export_model_unknown_format_raises(fp32_model, tmp_path):
    with patch(
        "mono_quant.export.orchestrator.validate_export_pre",
        return_value=[],
    ):
        with pytest.raises(ValueError, match="Unknown export format"):
            export_model(fp32_model, str(tmp_path / "out"), format="awq")


def test_export_model_non_module_raises(tmp_path):
    with pytest.raises(TypeError, match="nn.Module"):
        export_model({"not": "a module"}, str(tmp_path / "out"), format="onnx")


# ---------------------------------------------------------------------------
# result.export → orchestrator
# ---------------------------------------------------------------------------

def test_result_export_calls_orchestrator(fp32_model, tmp_path):
    from mono_quant.api import quantize

    result = quantize(fp32_model, bits=8, dynamic=True)
    out = str(tmp_path / "model.onnx")

    with patch(
        "mono_quant.export.orchestrator._export_onnx"
    ) as mock_onnx:
        with patch(
            "mono_quant.export.orchestrator.validate_export_pre",
            return_value=[],
        ):
            result.export(out)

    mock_onnx.assert_called_once()
