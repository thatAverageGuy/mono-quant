"""Tests for export pre/post validation (T-028)."""

import pytest
import torch
import torch.nn as nn

from mono_quant.export.common.validators import (
    ExportWarning,
    validate_export_post,
    validate_export_pre,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

class _FP32Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(8, 4)

    def forward(self, x):
        return self.linear(x)


class _FP16Model(nn.Module):
    """Model with FP16 weights — counts as 'quantized'."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(8, 4).half()

    def forward(self, x):
        return self.linear(x)


# ---------------------------------------------------------------------------
# ExportWarning dataclass
# ---------------------------------------------------------------------------

def test_export_warning_dataclass():
    w = ExportWarning(level="warning", message="test msg", check="test_check")
    assert w.level == "warning"
    assert w.message == "test msg"
    assert w.check == "test_check"


def test_export_warning_error_level():
    w = ExportWarning(level="error", message="bad thing", check="bad_check")
    assert w.level == "error"


# ---------------------------------------------------------------------------
# validate_export_pre
# ---------------------------------------------------------------------------

def test_validate_pre_no_quantized_layers_warns():
    model = _FP32Model()  # all float32 — no quantized layers
    warnings = validate_export_pre(model, None, "onnx")
    checks = [w.check for w in warnings]
    assert "no_quantized_layers" in checks


def test_validate_pre_fp16_model_no_warning():
    model = _FP16Model()
    warnings = validate_export_pre(model, None, "onnx")
    checks = [w.check for w in warnings]
    assert "no_quantized_layers" not in checks


def test_validate_pre_int8_gptq_errors():
    """INT8 model exported to GPTQ should produce an error-level warning."""
    model = _FP32Model()

    # Fake info with dtype = qint8
    class _FakeInfo:
        dtype = torch.qint8

    warnings = validate_export_pre(model, _FakeInfo(), "gptq")
    error_checks = [w.check for w in warnings if w.level == "error"]
    assert "int8_gptq_mismatch" in error_checks


def test_validate_pre_non_gptq_no_int8_error():
    model = _FP32Model()

    class _FakeInfo:
        dtype = torch.qint8

    warnings = validate_export_pre(model, _FakeInfo(), "onnx")
    checks = [w.check for w in warnings]
    assert "int8_gptq_mismatch" not in checks


def test_validate_pre_non_module_returns_empty():
    warnings = validate_export_pre({"not": "a module"}, None, "onnx")
    assert warnings == []


# ---------------------------------------------------------------------------
# validate_export_post dispatch
# ---------------------------------------------------------------------------

def test_validate_post_dispatches_onnx(tmp_path):
    from unittest.mock import patch

    p = tmp_path / "model.onnx"
    p.write_bytes(b"dummy")

    with patch(
        "mono_quant.export.common.validators.validate_onnx_model"
    ) as mock_fn:
        validate_export_post(str(p), "onnx")
    mock_fn.assert_called_once_with(str(p), level=mock_fn.call_args[1]["level"])


def test_validate_post_dispatches_gptq(tmp_path):
    from unittest.mock import patch

    with patch(
        "mono_quant.export.common.validators.validate_gptq_checkpoint_structure"
    ) as mock_fn:
        validate_export_post(str(tmp_path), "gptq")
    mock_fn.assert_called_once_with(str(tmp_path))


def test_validate_post_dispatches_gguf(tmp_path):
    from unittest.mock import patch

    p = tmp_path / "model.gguf"
    p.write_bytes(b"dummy")

    with patch(
        "mono_quant.export.common.validators.validate_gguf_checkpoint"
    ) as mock_fn:
        validate_export_post(str(p), "gguf")
    mock_fn.assert_called_once_with(str(p))


def test_validate_post_unknown_format_raises():
    with pytest.raises(ValueError, match="Unknown format"):
        validate_export_post("/some/path", "awq")
