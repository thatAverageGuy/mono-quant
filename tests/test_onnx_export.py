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


def test_export_onnx_dynamo_mlp_succeeds(tmp_path: Path) -> None:
    """T-040: dynamo=True exports a simple MLP to a valid ONNX file."""
    pytest.importorskip("onnxscript", reason="onnxscript not installed; pip install mono-quant[onnx]")
    q_model = _make_int8_model()
    out = tmp_path / "model_dynamo.onnx"
    from mono_quant.export.onnx import ONNXExporter

    exporter = ONNXExporter()
    exporter.export(q_model, out, dynamo=True)
    assert out.exists(), f"Expected {out} to exist after dynamo export"


def test_export_onnx_dynamo_embedding_model_succeeds(tmp_path: Path) -> None:
    """T-040: dynamo=True handles Embedding+Linear model with LongTensor input."""
    pytest.importorskip("onnxscript", reason="onnxscript not installed; pip install mono-quant[onnx]")
    from mono_quant import dynamic_quantize
    from mono_quant.export.onnx import ONNXExporter

    class EmbeddingModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(100, 32)
            self.fc = nn.Linear(32, 10)

        def forward(self, x):
            return self.fc(self.embed(x))

    model = EmbeddingModel()
    q_model, _ = dynamic_quantize(model)
    out = tmp_path / "embed_dynamo.onnx"
    dummy = torch.zeros(1, 16, dtype=torch.long)

    exporter = ONNXExporter()
    exporter.export(q_model, out, dummy_input=dummy, dynamo=True)
    assert out.exists(), f"Expected {out} to exist after dynamo export"


def test_export_onnx_dynamo_qdq_nodes_present(tmp_path: Path) -> None:
    """T-040: dynamo-exported flat MLP contains QDQ nodes (Gemm path — named weights)."""
    pytest.importorskip("onnxscript", reason="onnxscript not installed; pip install mono-quant[onnx]")
    q_model = _make_int8_model()
    out = tmp_path / "model_dynamo_qdq.onnx"
    from mono_quant.export.onnx import ONNXExporter

    exporter = ONNXExporter()
    exporter.export(q_model, out, dynamo=True)

    model_proto = onnx.load(str(out))
    op_types = {node.op_type for node in model_proto.graph.node}
    assert "QuantizeLinear" in op_types, f"QuantizeLinear not found in dynamo graph. Ops: {op_types}"
    assert "DequantizeLinear" in op_types, f"DequantizeLinear not found in dynamo graph. Ops: {op_types}"


def test_build_dynamo_name_map_recovers_transposed_weight() -> None:
    """T-041: _build_dynamo_name_map matches val_N anonymous initializer to named param.

    Simulates the MatMul(x, w.T) dynamo lowering pattern: constructs a minimal ONNX
    proto where a weight is stored transposed under an anonymous name, then verifies
    the function correctly identifies the original parameter name and transpose flag.
    """
    import numpy as np
    from onnx import TensorProto, helper, numpy_helper

    from mono_quant.export.common.qdq_inserter import _build_dynamo_name_map

    # Simple model with a known square weight.
    model = nn.Linear(8, 8, bias=False)
    weight_np = model.weight.detach().float().numpy()  # [8, 8]

    # Build a minimal ONNX proto with the weight stored TRANSPOSED as "val_7".
    weight_t = weight_np.T.astype(np.float32)
    init_transposed = numpy_helper.from_array(weight_t, name="val_7")
    # Also add a small constant that should be ignored (not weight-like).
    init_small = numpy_helper.from_array(np.zeros((4,), dtype=np.float32), name="val_2")

    X = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 8])
    Y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 8])
    node = helper.make_node("MatMul", inputs=["x", "val_7"], outputs=["y"])
    graph = helper.make_graph(
        [node], "test_graph", [X], [Y], initializer=[init_transposed, init_small]
    )
    proto = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 14)])

    name_map = _build_dynamo_name_map(model, proto)

    assert "val_7" in name_map, f"Expected 'val_7' in name_map, got keys: {list(name_map)}"
    param_name, is_transposed = name_map["val_7"]
    assert param_name == "weight", f"Expected param 'weight', got '{param_name}'"
    assert is_transposed is True, "Expected is_transposed=True for w.T stored as val_7"
    # Small constant must NOT appear in the map.
    assert "val_2" not in name_map, "Small 1-D constant should not be in name_map"


def test_export_onnx_dynamo_qdq_explicit_transpose(tmp_path: Path) -> None:
    """T-041: QDQ nodes inserted for a model that uses weight.T in forward.

    This exercises the val_N recovery path: the explicit weight.t() call causes
    dynamo to constant-fold the transposed weight into an anonymous initializer.
    _build_dynamo_name_map recovers the name and insert_qdq_nodes wraps it.
    """
    pytest.importorskip("onnxscript", reason="onnxscript not installed; pip install mono-quant[onnx]")
    from mono_quant import dynamic_quantize
    from mono_quant.export.onnx import ONNXExporter

    class TransposeMatMulModel(nn.Module):
        """Uses explicit weight.t() in MatMul — forces val_N naming in dynamo export."""

        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(32, 32, bias=False)

        def forward(self, x):
            # F.linear calls this internally; writing it explicitly ensures dynamo
            # stores the transposed weight as a constant (not a Gemm with transB=1).
            return torch.matmul(x, self.proj.weight.t())

    model = TransposeMatMulModel()
    q_model, _ = dynamic_quantize(model)
    out = tmp_path / "model_transpose_dynamo.onnx"
    dummy = torch.randn(1, 32)

    exporter = ONNXExporter()
    exporter.export(q_model, out, dummy_input=dummy, dynamo=True)

    assert out.exists(), f"Expected {out} to exist after export"
    model_proto = onnx.load(str(out))
    op_types = {node.op_type for node in model_proto.graph.node}
    assert "QuantizeLinear" in op_types, (
        f"QuantizeLinear not found — name map recovery may have failed. Ops: {op_types}"
    )
    assert "DequantizeLinear" in op_types, (
        f"DequantizeLinear not found — name map recovery may have failed. Ops: {op_types}"
    )


def test_export_onnx_hf_model_use_cache_restored(tmp_path: Path) -> None:
    """BF-017: config.use_cache is restored after export — even on failure."""
    from mono_quant.export.onnx import ONNXExporter
    from types import SimpleNamespace

    class FakeHFModel(nn.Module):
        """Simulates a HuggingFace model with config.use_cache."""
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(use_cache=True)
            self.fc = nn.Linear(8, 4)

        def forward(self, x):
            return self.fc(x)

    model = FakeHFModel()
    exporter = ONNXExporter()
    out = tmp_path / "hf_model.onnx"

    exporter.export(model, out)

    # config.use_cache must be restored to original True after export
    assert model.config.use_cache is True, (
        "config.use_cache must be restored to True after successful export"
    )


def test_export_onnx_hf_model_use_cache_restored_on_failure(tmp_path: Path) -> None:
    """BF-017: config.use_cache is restored even when tracing fails."""
    from mono_quant.export.onnx import ONNXExporter
    from types import SimpleNamespace

    class FailingHFModel(nn.Module):
        """Model that raises TypeError during forward — simulates complex HF model."""
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(use_cache=True)
            self.embed = nn.Embedding(100, 32)

        def forward(self, x):
            raise TypeError("unexpected keyword argument")

    model = FailingHFModel()
    exporter = ONNXExporter()
    out = tmp_path / "fail.onnx"
    dummy = torch.zeros(1, 16, dtype=torch.long)

    with pytest.raises(RuntimeError):
        exporter.export(model, out, dummy_input=dummy)

    # config.use_cache must be restored even on failure
    assert model.config.use_cache is True, (
        "config.use_cache must be restored to True even when export fails"
    )


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
