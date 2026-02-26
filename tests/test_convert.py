"""Tests for result.convert() and monoquant convert CLI command (T-029)."""

import warnings

import pytest
import torch
import torch.nn as nn
from click.testing import CliRunner

from mono_quant import quantize
from mono_quant.api.result import QuantizationResult
from mono_quant.cli.main import cli


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

class _SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(16, 8)

    def forward(self, x):
        return self.linear(x)


@pytest.fixture()
def int8_result():
    model = _SimpleModel()
    return quantize(model, bits=8, dynamic=True)


@pytest.fixture()
def runner():
    return CliRunner()


# ---------------------------------------------------------------------------
# result.convert()
# ---------------------------------------------------------------------------

def test_convert_int8_to_int4_returns_result(int8_result):
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        new_result = int8_result.convert(bits=4)
    assert isinstance(new_result, QuantizationResult)


def test_convert_int8_to_fp16_returns_result(int8_result):
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        new_result = int8_result.convert(bits=16)
    assert isinstance(new_result, QuantizationResult)


def test_convert_result_is_quantization_result(int8_result):
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        new_result = int8_result.convert(bits=8)
    assert hasattr(new_result, "model")
    assert hasattr(new_result, "info")
    assert isinstance(new_result.model, nn.Module)


def test_convert_emits_sqnr_warning(int8_result):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        int8_result.convert(bits=4)

    user_warnings = [w for w in caught if issubclass(w.category, UserWarning)]
    assert len(user_warnings) >= 1
    msg = str(user_warnings[0].message)
    assert "convert" in msg.lower() or "re-quantization" in msg.lower()


def test_convert_does_not_modify_original(int8_result):
    original_id = id(int8_result.model)
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        int8_result.convert(bits=4)
    assert id(int8_result.model) == original_id


# ---------------------------------------------------------------------------
# CLI convert command
# ---------------------------------------------------------------------------

@pytest.fixture()
def saved_int8_model(tmp_path):
    model = _SimpleModel()
    result = quantize(model, bits=8, dynamic=True)
    p = tmp_path / "model_int8.pt"
    torch.save(result.model, str(p))
    return str(p)


def test_cli_convert_dispatches(runner, saved_int8_model, tmp_path):
    out = str(tmp_path / "model_int4.pt")
    result = runner.invoke(
        cli,
        ["convert", saved_int8_model, out, "--bits", "4"],
    )
    assert result.exit_code == 0, result.output
    assert tmp_path.joinpath("model_int4.pt").exists()
