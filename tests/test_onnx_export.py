"""Tests for ONNX export functionality.

All tests in this module are skipped automatically if onnx is not installed.
Install test dependencies with: pip install mono-quant[onnx]
"""

import builtins
import sys
import warnings
from pathlib import Path

import pytest
import torch
import torch.nn as nn

onnx = pytest.importorskip("onnx", reason="onnx not installed; pip install mono-quant[onnx]")
pytest.importorskip("onnxruntime", reason="onnxruntime not installed; pip install mono-quant[onnx]")

from mono_quant import dynamic_quantize, export_to_onnx
from mono_quant.modules.linear import QuantizedLinearInt4


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_int8_model() -> nn.Module:
    """Small INT8 dynamically-quantized model for tests."""
    model = nn.Sequential(
        nn.Linear(32, 16),
        nn.ReLU(),
        nn.Linear(16, 8),
    )
    q_model, _ = dynamic_quantize(model)
    return q_model


def _make_int4_model() -> nn.Module:
    """Model with a single INT4 layer.

    out_features=128 >= group_size=128 satisfies INT4 quantization constraint.
    """
    linear = nn.Linear(128, 128)
    q_int4 = QuantizedLinearInt4.from_float(linear, group_size=128)
    return nn.Sequential(q_int4)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_export_to_onnx_int8_linear(tmp_path: Path) -> None:
    """INT8 model exports without raising an exception."""
    q_model = _make_int8_model()
    out = tmp_path / "model.onnx"
    export_to_onnx(q_model, out)  # must not raise


def test_export_onnx_file_exists(tmp_path: Path) -> None:
    """Output .onnx file is created at the specified path."""
    q_model = _make_int8_model()
    out = tmp_path / "model.onnx"
    export_to_onnx(q_model, out)
    assert out.exists(), f"Expected {out} to exist after export"


def test_export_onnx_validate_load(tmp_path: Path) -> None:
    """validate='load' passes onnx.checker.check_model without error."""
    q_model = _make_int8_model()
    out = tmp_path / "model.onnx"
    export_to_onnx(q_model, out, validate="load")  # must not raise


def test_export_onnx_qdq_nodes_present(tmp_path: Path) -> None:
    """Exported ONNX graph contains QuantizeLinear and DequantizeLinear nodes."""
    q_model = _make_int8_model()
    out = tmp_path / "model.onnx"
    export_to_onnx(q_model, out)

    model_proto = onnx.load(str(out))
    op_types = {node.op_type for node in model_proto.graph.node}
    assert "QuantizeLinear" in op_types, f"QuantizeLinear not found. Ops: {op_types}"
    assert "DequantizeLinear" in op_types, f"DequantizeLinear not found. Ops: {op_types}"


def test_export_onnx_opset_default_14(tmp_path: Path) -> None:
    """Default export uses opset 14."""
    q_model = _make_int8_model()
    out = tmp_path / "model.onnx"
    export_to_onnx(q_model, out)

    model_proto = onnx.load(str(out))
    opset_versions = {op.version for op in model_proto.opset_import}
    assert 14 in opset_versions, f"Opset 14 not found. Got: {opset_versions}"


def test_export_onnx_int4_warning(tmp_path: Path) -> None:
    """INT4 model emits a UserWarning mentioning INT4 when opset < 21."""
    model = _make_int4_model()
    out = tmp_path / "model.onnx"

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        export_to_onnx(model, out, opset=14)

    int4_warnings = [w for w in caught if "INT4" in str(w.message)]
    assert int4_warnings, (
        "Expected a UserWarning about INT4 layers, but none was emitted. "
        f"All warnings: {[str(w.message) for w in caught]}"
    )


def test_infer_dummy_input_embedding_model_returns_long() -> None:
    """BF-014/Bug3a: _infer_dummy_input returns LongTensor when model has Embedding first."""
    from mono_quant.export.onnx import ONNXExporter

    class EmbeddingFirst(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(100, 32)
            self.fc = nn.Linear(32, 10)

        def forward(self, x):
            return self.fc(self.embed(x))

    dummy = ONNXExporter()._infer_dummy_input(EmbeddingFirst())
    assert dummy.dtype == torch.long, (
        f"Expected torch.long for embedding model, got {dummy.dtype}"
    )


def test_onnx_export_tracing_error_is_graceful(tmp_path: Path) -> None:
    """BF-014/Bug3b: tracing failures raise a RuntimeError with a user-actionable hint.

    Covers both RuntimeError (wrong tensor dtype) and TypeError (unexpected
    kwargs in complex model forwards) — both are caught and re-raised with hint.
    """
    from mono_quant.export.onnx import ONNXExporter

    class RequiresLong(nn.Module):
        """Forward only works with LongTensor (embedding model)."""
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(100, 32)
            self.fc = nn.Linear(32, 10)

        def forward(self, x):
            return self.fc(self.embed(x))

    class RaisesTypeError(nn.Module):
        """Forward raises TypeError (simulates complex model with unexpected kwargs)."""
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(100, 32)

        def forward(self, x):
            raise TypeError("unexpected keyword argument 'position_ids'")

    exporter = ONNXExporter()
    out = tmp_path / "bad.onnx"

    # RuntimeError path (wrong dtype)
    float_dummy = torch.zeros(1, 16, dtype=torch.float32)
    with pytest.raises(RuntimeError, match="dummy_input"):
        exporter.export(RequiresLong(), out, dummy_input=float_dummy)

    # TypeError path (complex model forward)
    long_dummy = torch.zeros(1, 16, dtype=torch.long)
    with pytest.raises(RuntimeError, match="dummy_input"):
        exporter.export(RaisesTypeError(), out, dummy_input=long_dummy)


def test_export_onnx_raises_without_onnx_installed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A helpful ImportError is raised when onnx is not available."""
    q_model = _make_int8_model()

    orig_import = builtins.__import__

    def mock_no_onnx(name: str, *args, **kwargs):
        if name == "onnx":
            raise ImportError("No module named 'onnx'")
        return orig_import(name, *args, **kwargs)

    # Remove cached onnx modules so the mock triggers on re-import
    to_remove = [k for k in list(sys.modules) if k == "onnx" or k.startswith("onnx.")]
    for k in to_remove:
        monkeypatch.delitem(sys.modules, k, raising=False)

    monkeypatch.setattr(builtins, "__import__", mock_no_onnx)

    with pytest.raises(ImportError, match=r"pip install mono-quant\[onnx\]"):
        export_to_onnx(q_model, str(tmp_path / "model.onnx"))
