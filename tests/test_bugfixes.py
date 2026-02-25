"""Tests for audit bug fixes BF-002, BF-003, BF-004, BF-009, BF-013 and T-030."""

import io
import os
import tempfile

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# T-030: export_to_onnx stub raises NotImplementedError
# ---------------------------------------------------------------------------

def test_export_to_onnx_raises_not_implemented():
    """T-030: export_to_onnx must raise NotImplementedError, not AttributeError."""
    import mono_quant

    assert hasattr(mono_quant, "export_to_onnx"), "export_to_onnx must be exported"

    model = nn.Linear(4, 4)
    try:
        mono_quant.export_to_onnx(model, "dummy.onnx")
        assert False, "Should have raised NotImplementedError"
    except NotImplementedError:
        pass  # expected
    except AttributeError as e:
        assert False, f"Got AttributeError instead of NotImplementedError: {e}"


# ---------------------------------------------------------------------------
# BF-004: CLI exit code paths use raise SystemExit, not Context.exit()
# ---------------------------------------------------------------------------

def test_cli_quantize_no_calibration_exits_code_2():
    """BF-004: Static quantize without --calibration must exit with code 2, not TypeError."""
    from click.testing import CliRunner

    from mono_quant.cli.commands import quantize_cmd

    # Create a minimal dummy model file
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        dummy_path = f.name
    try:
        torch.save(nn.Linear(4, 4).state_dict(), dummy_path)
        runner = CliRunner()
        # --static (not --dynamic) without --calibration should exit 2
        result = runner.invoke(quantize_cmd, ["--model", dummy_path], obj={})
        assert result.exit_code == 2, (
            f"Expected exit code 2, got {result.exit_code}. Output: {result.output}"
        )
        # Must NOT be a TypeError (which would come from Context.exit() bug)
        if result.exception is not None:
            assert not isinstance(result.exception, TypeError), (
                f"Got TypeError — Context.exit() class method bug not fixed: {result.exception}"
            )
    finally:
        os.unlink(dummy_path)


def test_cli_calibrate_exits_code_1():
    """BF-004: calibrate_cmd stub must exit with code 1, not TypeError."""
    from click.testing import CliRunner

    from mono_quant.cli.commands import calibrate_cmd

    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        model_path = f.name
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        data_path = f.name
    try:
        torch.save({}, model_path)
        torch.save({}, data_path)
        runner = CliRunner()
        result = runner.invoke(calibrate_cmd, [model_path, data_path], obj={})
        assert result.exit_code == 1, (
            f"Expected exit code 1, got {result.exit_code}. Output: {result.output}"
        )
        if result.exception is not None:
            assert not isinstance(result.exception, TypeError), (
                f"Got TypeError — Context.exit() class method bug not fixed: {result.exception}"
            )
    finally:
        os.unlink(model_path)
        os.unlink(data_path)


# ---------------------------------------------------------------------------
# BF-002: result.save() and _build_metadata with CoreQuantizationInfo
# ---------------------------------------------------------------------------

def test_build_metadata_with_core_quantization_info():
    """BF-002: _build_metadata must work with core.quantizers.QuantizationInfo."""
    from mono_quant.core.quantizers import QuantizationInfo
    from mono_quant.io.formats import _build_metadata

    info = QuantizationInfo(
        selected_layers=["linear"],
        skipped_layers=[],
        calibration_samples_used=100,
        dtype=torch.qint8,
        symmetric=True,
    )
    metadata = _build_metadata(quantization_info=info)

    assert "quantization_dtype" in metadata
    assert "scheme" in metadata
    assert metadata["scheme"] == "symmetric"
    assert "per_channel" in metadata
    assert "selected_layers" in metadata
    assert "bits" in metadata
    assert metadata["bits"] == "8"
    assert "calibration_samples" in metadata
    assert metadata["calibration_samples"] == "100"


def test_result_save_completes_without_error():
    """BF-002: result.save() must not raise AttributeError."""
    from mono_quant import quantize

    model = nn.Linear(10, 5)
    result = quantize(model, bits=8, dynamic=True)
    assert result.success

    with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as f:
        path = f.name
    try:
        result.save(path)  # must not raise AttributeError
        assert os.path.exists(path)
        assert os.path.getsize(path) > 0
    finally:
        if os.path.exists(path):
            os.unlink(path)


# ---------------------------------------------------------------------------
# BF-003: INT4 symmetric quantization round-trip
# ---------------------------------------------------------------------------

def test_int4_symmetric_quantized_values_in_range():
    """BF-003: quantized INT4 values must be in [-8, 7] after removing spurious -8."""
    from mono_quant.core.quantizers import quantize_weight_int4
    from mono_quant.core.mappers import _unpack_int8_to_int4

    weight = torch.randn(256, 128)
    packed, scales, zero_points = quantize_weight_int4(weight, symmetric=True)

    # Unpack to verify raw int4 values
    unpacked = _unpack_int8_to_int4(packed, num_elements=weight.numel())
    assert unpacked.min().item() >= -8, f"Min value {unpacked.min().item()} < -8"
    assert unpacked.max().item() <= 7, f"Max value {unpacked.max().item()} > 7"


def test_int4_symmetric_round_trip_cosine_similarity():
    """BF-003: INT4 symmetric quantize→dequantize must preserve weight direction."""
    from mono_quant.modules.linear import QuantizedLinearInt4

    torch.manual_seed(42)
    linear = nn.Linear(128, 256)
    q = QuantizedLinearInt4.from_float(linear, symmetric=True)

    dq_weight = q.weight
    cos_sim = F.cosine_similarity(
        dq_weight.flatten().unsqueeze(0),
        linear.weight.data.flatten().unsqueeze(0),
    )
    assert cos_sim.item() > 0.9, (
        f"Cosine similarity {cos_sim.item():.4f} < 0.9 — "
        "INT4 symmetric formula may still be wrong"
    )


def test_int4_symmetric_round_trip_max_error():
    """BF-003: INT4 symmetric max reconstruction error must be < 20% of max_abs."""
    from mono_quant.modules.linear import QuantizedLinearInt4

    torch.manual_seed(42)
    linear = nn.Linear(128, 256)
    q = QuantizedLinearInt4.from_float(linear, symmetric=True)

    dq_weight = q.weight
    max_abs = linear.weight.data.abs().max().item()
    max_err = (dq_weight - linear.weight.data).abs().max().item()
    assert max_err < 0.2 * max_abs, (
        f"Max error {max_err:.4f} > 20% of max_abs {max_abs:.4f}"
    )


# ---------------------------------------------------------------------------
# BF-009: dequantize_model does not crash on qint8 buffers
# ---------------------------------------------------------------------------

def test_dequantize_model_qint8_buffer_no_crash():
    """BF-009: dequantize_model must not crash when model has a qint8 buffer."""
    from mono_quant import dequantize_model

    model = nn.Linear(4, 4)
    # Register a qint8 buffer manually (simulates a quantized running stat)
    q_buf = torch.quantize_per_tensor(
        torch.randn(4), scale=0.1, zero_point=0, dtype=torch.qint8
    )
    model.register_buffer("quantized_running_stat", q_buf)

    # Must not raise RuntimeError
    dq_model = dequantize_model(model)

    buf = dict(dq_model.named_buffers())["quantized_running_stat"]
    assert buf.dtype == torch.float32, (
        f"Buffer dtype is {buf.dtype}, expected float32"
    )


def test_dequantize_model_non_quantized_passthrough():
    """BF-009: dequantize_model must leave non-quantized buffers unchanged."""
    from mono_quant import dequantize_model

    model = nn.BatchNorm1d(4)
    # BatchNorm has float32 running_mean/running_var buffers
    dq_model = dequantize_model(model)

    for name, buf in dq_model.named_buffers():
        if buf is not None:
            assert buf.dtype in (torch.float32, torch.int64), (
                f"Buffer {name} has unexpected dtype {buf.dtype}"
            )
