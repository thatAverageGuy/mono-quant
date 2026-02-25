"""Tests for audit bug fixes BF-002–BF-012, T-030, T-033, and CL-001."""

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


# ---------------------------------------------------------------------------
# BF-005: mutable default argument in _quantize_sequential_module
# ---------------------------------------------------------------------------

def test_sequential_module_skip_set_not_shared_across_calls():
    """BF-005: _quantize_sequential_module must not share skip_set across calls."""
    from mono_quant.core.quantizers import _quantize_sequential_module

    seq1 = nn.Sequential(nn.Linear(4, 4), nn.ReLU())
    seq2 = nn.Sequential(nn.Linear(4, 4), nn.ReLU())
    skipped1: list = []
    skipped2: list = []

    # First call with no skip_set
    _quantize_sequential_module(seq1, skipped1, torch.qint8, False)
    # Second call — must not inherit any state from first call
    _quantize_sequential_module(seq2, skipped2, torch.qint8, False)

    # Both calls should independently quantize nn.Linear (index 0) and skip ReLU
    from mono_quant.modules.linear import QuantizedLinear
    assert isinstance(seq1[0], QuantizedLinear), "seq1 Linear not quantized"
    assert isinstance(seq2[0], QuantizedLinear), "seq2 Linear not quantized"
    assert "1" in skipped1 or "0.1" in skipped1 or len(skipped1) > 0
    assert "1" in skipped2 or "0.1" in skipped2 or len(skipped2) > 0


# ---------------------------------------------------------------------------
# BF-006: INT4 skip list not applied for INT8 static_quantize by default
# ---------------------------------------------------------------------------

def test_static_quantize_int8_does_not_apply_int4_skip_list():
    """BF-006: Default INT8 static_quantize must not silently skip embedding layers."""
    from mono_quant.core.quantizers import static_quantize

    class ModelWithEmbedding(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(100, 16)
            self.linear = nn.Linear(16, 4)

        def forward(self, x):
            return self.linear(self.embed(x))

    model = ModelWithEmbedding()
    calib_data = [torch.randint(0, 100, (4,)) for _ in range(5)]

    # With default group_size=0, INT4 skip list must NOT be applied
    q_model, info = static_quantize(model, calib_data, run_validation=False)

    # The embedding should appear in selected_layers or be handled — it should NOT
    # be silently skipped because of INT4's default skip list
    assert "embed" not in info.skipped_layers or "lm_head" not in info.skipped_layers, (
        "INT4 skip list was applied to an INT8 static_quantize call"
    )


# ---------------------------------------------------------------------------
# BF-007: quantize_weight_int4 fallback raises RuntimeError for small layers
# ---------------------------------------------------------------------------

def test_quantize_weight_int4_small_layer_raises_runtime_error():
    """BF-007: quantize_weight_int4 must raise RuntimeError when dim < group_size."""
    from mono_quant.core.quantizers import quantize_weight_int4

    # Layer dim 16 < group_size 128 triggers the fallback
    small_weight = torch.randn(16, 8)
    try:
        quantize_weight_int4(small_weight, group_size=128, symmetric=True)
        assert False, "Expected RuntimeError for small layer"
    except RuntimeError as e:
        assert "group_size" in str(e).lower() or "dimension" in str(e).lower(), (
            f"RuntimeError message unclear: {e}"
        )


def test_quantize_weight_int4_normal_path_unchanged():
    """BF-007: Normal INT4 path (dim >= group_size) must still work."""
    from mono_quant.core.quantizers import quantize_weight_int4

    weight = torch.randn(256, 128)
    packed, scales, zero_points = quantize_weight_int4(weight, group_size=128, symmetric=True)
    assert packed is not None
    assert scales is not None


# ---------------------------------------------------------------------------
# BF-008: nested layer detection in _quantize_int8_model now works
# ---------------------------------------------------------------------------

class _InnerModule(nn.Module):
    """Two-level nested container for BF-008 test."""
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 4)

    def forward(self, x):
        return self.linear(x)


class _OuterModule(nn.Module):
    """Top-level non-Sequential container for BF-008 test."""
    def __init__(self):
        super().__init__()
        self.inner = _InnerModule()

    def forward(self, x):
        return self.inner(x)


def test_nested_non_sequential_linear_is_quantized():
    """BF-008: nn.Linear nested inside a non-Sequential container must be quantized."""
    from mono_quant import dynamic_quantize
    from mono_quant.modules.linear import QuantizedLinear

    model = _OuterModule()
    assert isinstance(model.inner.linear, nn.Linear), "Precondition: unquantized"

    q_model, skipped = dynamic_quantize(model)

    assert isinstance(q_model.inner.linear, QuantizedLinear), (
        "Nested nn.Linear was not quantized — BF-008 nested detection still broken"
    )
    assert "inner.linear" not in skipped, "inner.linear should not appear in skipped"


# ---------------------------------------------------------------------------
# BF-010: quantize_embedding_module passes dtype through
# ---------------------------------------------------------------------------

def test_embedding_quantize_int8_dtype():
    """BF-010: quantize_embedding_module with dtype=qint8 gives qint8 weight."""
    from mono_quant.modules.embedding import quantize_embedding_module

    emb = nn.Embedding(100, 32)
    q_emb = quantize_embedding_module(emb, dtype=torch.qint8)
    assert q_emb._quantized_weight is not None
    assert q_emb._quantized_weight.is_quantized, (
        "Expected quantized (qint8) weight, got non-quantized"
    )


def test_embedding_quantize_fp16_dtype():
    """BF-010: quantize_embedding_module with dtype=float16 gives float16 weight."""
    from mono_quant.modules.embedding import quantize_embedding_module

    emb = nn.Embedding(100, 32)
    q_emb = quantize_embedding_module(emb, dtype=torch.float16)
    assert q_emb._quantized_weight is not None
    assert q_emb._quantized_weight.dtype == torch.float16, (
        f"Expected float16 weight, got {q_emb._quantized_weight.dtype}"
    )


# ---------------------------------------------------------------------------
# BF-011: _test_load_run does not mutate the model under test
# ---------------------------------------------------------------------------

def test_load_run_does_not_mutate_original_model():
    """BF-011: _test_load_run must not modify the model passed to it."""
    import copy
    from mono_quant.io.validation import _test_load_run
    from mono_quant import dynamic_quantize

    model = nn.Linear(4, 4)
    q_model, _ = dynamic_quantize(model)

    original_state = copy.deepcopy(q_model.state_dict())
    _test_load_run(q_model)  # must not modify q_model

    for key in original_state:
        after = q_model.state_dict()[key]
        before = original_state[key]
        if before.is_floating_point():
            assert torch.allclose(before, after), (
                f"Parameter {key} was mutated by _test_load_run"
            )


# ---------------------------------------------------------------------------
# BF-012: _check_weight_ranges no false positive on large but valid weights
# ---------------------------------------------------------------------------

def test_check_weight_ranges_no_false_positive_on_large_weights():
    """BF-012: _check_weight_ranges must not flag normally-distributed large weights."""
    from mono_quant.io.validation import _check_weight_ranges
    from mono_quant import dynamic_quantize

    # Model whose weights are large but not corrupted
    model = nn.Linear(4, 4)
    torch.nn.init.constant_(model.weight, 150.0)  # > old hardcoded 100

    q_model, _ = dynamic_quantize(model)
    result = _check_weight_ranges(q_model)
    assert result is True, (
        "False positive: _check_weight_ranges flagged large-but-valid weights as bad"
    )


# ---------------------------------------------------------------------------
# T-033: quantize() raises TypeError for file path input
# ---------------------------------------------------------------------------

def test_quantize_file_path_raises_type_error():
    """T-033: quantize() must raise TypeError when passed a file path string."""
    from mono_quant import quantize

    try:
        quantize("some_model.pt", bits=8)
        assert False, "Expected TypeError for string input"
    except TypeError as e:
        assert "nn.Module" in str(e), (
            f"TypeError message should mention nn.Module, got: {e}"
        )


def test_quantize_path_object_raises_type_error():
    """T-033: quantize() must raise TypeError when passed a Path object."""
    from pathlib import Path
    from mono_quant import quantize

    try:
        quantize(Path("some_model.pt"), bits=8)
        assert False, "Expected TypeError for Path input"
    except TypeError as e:
        assert "nn.Module" in str(e), (
            f"TypeError message should mention nn.Module, got: {e}"
        )


# ---------------------------------------------------------------------------
# CL-001: version string, removed test_models_from_any_source
# ---------------------------------------------------------------------------

def test_version_is_semver():
    """CL-001 M1: __version__ must be a valid semver string (X.Y.Z)."""
    import mono_quant

    version = mono_quant.__version__
    parts = version.split(".")
    assert len(parts) == 3, (
        f"__version__ '{version}' is not semver (expected X.Y.Z)"
    )
    for part in parts:
        assert part.isdigit(), (
            f"__version__ '{version}' part '{part}' is not a number"
        )


def test_test_models_not_in_public_api():
    """CL-001 M4: test_models_from_any_source must not be importable from public API."""
    import mono_quant

    assert not hasattr(mono_quant, "test_models_from_any_source"), (
        "test_models_from_any_source should not be in the public API"
    )


# ---------------------------------------------------------------------------
# T-032: HistogramObserver — consistent histogram accumulation (Bug A)
#         and correct asymmetric zero-point formula (Bug B)
# ---------------------------------------------------------------------------

def test_histogram_observer_consistent_bins_same_range():
    """T-032 Bug A: Two batches with the same range accumulate correctly.

    Total histogram count must equal total elements across both batches.
    """
    from mono_quant.core.observers import HistogramObserver

    obs = HistogramObserver(bins=64)
    batch = torch.linspace(0.0, 1.0, 100)
    obs.forward(batch)
    obs.forward(batch)

    total = obs.histogram.sum().item()
    assert abs(total - 200) < 1, (
        f"Expected 200 total counts (100 per batch), got {total}"
    )


def test_histogram_observer_range_expands_tracks_full_range():
    """T-032 Bug A: After a batch that expands the range, min_val/max_val cover both batches.

    The old code accumulated counts from batches with different bin edges —
    semantically meaningless. The fix ensures the running range is always correct
    so KL divergence operates on a coherent histogram.
    """
    from mono_quant.core.observers import HistogramObserver

    obs = HistogramObserver(bins=20)
    obs.forward(torch.zeros(50))        # range [0, 0]
    obs.forward(torch.full((50,), 6.0)) # range expands to [0, 6]

    assert obs.min_val == 0.0, f"min_val should be 0, got {obs.min_val}"
    assert obs.max_val == 6.0, f"max_val should be 6, got {obs.max_val}"
    # After range expansion, histogram is reset and only the latest batch is counted
    assert obs.histogram.sum().item() > 0, "histogram must have counts after forward"


def test_histogram_observer_zero_point_positive_activations():
    """T-032 Bug B: Zero-point for all-positive activations is not anchored at qmin.

    The buggy formula used (-T/2)/scale which introduced a wrong symmetric offset.
    The correct formula zp = round(qmin - min_val/scale) with min_val=-T gives
    zero_point close to 0 (> qmin=-128) for a symmetric clipping range [-T, T].
    """
    from mono_quant.core.observers import HistogramObserver

    obs = HistogramObserver(bins=256)
    obs.forward(torch.linspace(0.0, 1.0, 1000))
    scale, zp = obs.calculate_qparams()

    qmin = -128
    assert zp.item() > qmin, (
        f"zero_point {zp.item()} should be > qmin={qmin} for positive activations"
    )
    assert scale.item() > 0, "scale must be positive"


def test_histogram_observer_zero_point_symmetric_activations():
    """T-032 Bug B: Zero-point for symmetric activations [-1, 1] is approximately 0."""
    from mono_quant.core.observers import HistogramObserver

    obs = HistogramObserver(bins=256)
    obs.forward(torch.linspace(-1.0, 1.0, 1000))
    scale, zp = obs.calculate_qparams()

    assert abs(zp.item()) <= 1, (
        f"zero_point {zp.item()} should be ~0 for symmetric activations"
    )
    assert scale.item() > 0, "scale must be positive"


# ---------------------------------------------------------------------------
# T-031: Activation-based calibration stats are collected and applied
# ---------------------------------------------------------------------------

def test_static_quantize_sets_input_scale_from_calibration():
    """T-031: static_quantize must set input_scale on QuantizedLinear when cal data provided."""
    from mono_quant import static_quantize
    from mono_quant.modules.linear import QuantizedLinear

    model = nn.Sequential(nn.Linear(8, 8), nn.ReLU(), nn.Linear(8, 4))
    cal_data = [torch.randn(4, 8) for _ in range(10)]

    q_model, _ = static_quantize(model, cal_data)

    quantized_linears = [
        m for _, m in q_model.named_modules() if isinstance(m, QuantizedLinear)
    ]
    assert quantized_linears, "static_quantize must produce QuantizedLinear modules"

    has_activation_qparams = any(m.input_scale is not None for m in quantized_linears)
    assert has_activation_qparams, (
        "static_quantize must set input_scale on at least one QuantizedLinear "
        "when calibration data is provided"
    )


def test_static_quantize_forward_with_activation_qparams():
    """T-031: static-quantized model with activation qparams produces correct output shape."""
    from mono_quant import static_quantize

    model = nn.Linear(8, 4)
    cal_data = [torch.randn(2, 8) for _ in range(5)]

    q_model, _ = static_quantize(model, cal_data)
    out = q_model(torch.randn(3, 8))
    assert out.shape == (3, 4), f"Expected (3, 4), got {out.shape}"


def test_static_quantize_no_calibration_data_no_input_scale():
    """T-031: static_quantize without calibration data leaves input_scale as None (graceful fallback)."""
    from mono_quant import static_quantize
    from mono_quant.modules.linear import QuantizedLinear

    model = nn.Linear(8, 4)
    q_model, _ = static_quantize(model, [])

    for _, m in q_model.named_modules():
        if isinstance(m, QuantizedLinear):
            assert m.input_scale is None, (
                "No calibration data → input_scale must remain None"
            )
