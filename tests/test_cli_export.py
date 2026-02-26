"""Tests for unified CLI export command (T-027)."""

from pathlib import Path
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn
from click.testing import CliRunner

from mono_quant.cli.main import cli


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
def saved_model(tmp_path):
    """Write a simple model to disk with torch.save; return the path."""
    model = _SimpleModel()
    p = tmp_path / "model.pt"
    torch.save(model, str(p))
    return str(p)


@pytest.fixture()
def runner():
    return CliRunner()


# ---------------------------------------------------------------------------
# --list-formats
# ---------------------------------------------------------------------------

def test_export_cmd_list_formats(runner):
    result = runner.invoke(cli, ["export", "--list-formats"])
    assert result.exit_code == 0
    assert "onnx" in result.output
    assert "gptq" in result.output
    assert "gguf" in result.output


# ---------------------------------------------------------------------------
# Dispatch by format
# ---------------------------------------------------------------------------

def test_export_cmd_onnx_dispatches(runner, saved_model, tmp_path):
    out = str(tmp_path / "model.onnx")
    with patch("mono_quant.export.orchestrator._export_onnx") as mock_fn:
        with patch("mono_quant.export.orchestrator.validate_export_pre", return_value=[]):
            result = runner.invoke(
                cli, ["export", "-m", saved_model, "-o", out, "--format", "onnx"]
            )
    assert result.exit_code == 0, result.output
    mock_fn.assert_called_once()


def test_export_cmd_gptq_dispatches(runner, saved_model, tmp_path):
    out = str(tmp_path / "gptq_dir")
    with patch("mono_quant.export.orchestrator._export_gptq") as mock_fn:
        with patch("mono_quant.export.orchestrator.validate_export_pre", return_value=[]):
            result = runner.invoke(
                cli, ["export", "-m", saved_model, "-o", out, "--format", "gptq"]
            )
    assert result.exit_code == 0, result.output
    mock_fn.assert_called_once()


def test_export_cmd_gguf_dispatches(runner, saved_model, tmp_path):
    out = str(tmp_path / "gguf_dir")
    with patch("mono_quant.export.orchestrator._export_gguf") as mock_fn:
        with patch("mono_quant.export.orchestrator.validate_export_pre", return_value=[]):
            result = runner.invoke(
                cli, ["export", "-m", saved_model, "-o", out, "--format", "gguf"]
            )
    assert result.exit_code == 0, result.output
    mock_fn.assert_called_once()


# ---------------------------------------------------------------------------
# Auto-detect from extension
# ---------------------------------------------------------------------------

def test_export_cmd_autodetect_onnx(runner, saved_model, tmp_path):
    out = str(tmp_path / "model.onnx")
    with patch("mono_quant.export.orchestrator._export_onnx") as mock_fn:
        with patch("mono_quant.export.orchestrator.validate_export_pre", return_value=[]):
            result = runner.invoke(cli, ["export", "-m", saved_model, "-o", out])
    assert result.exit_code == 0, result.output
    mock_fn.assert_called_once()


def test_export_cmd_autodetect_gguf(runner, saved_model, tmp_path):
    out = str(tmp_path / "model.gguf")
    with patch("mono_quant.export.orchestrator._export_gguf") as mock_fn:
        with patch("mono_quant.export.orchestrator.validate_export_pre", return_value=[]):
            result = runner.invoke(cli, ["export", "-m", saved_model, "-o", out])
    assert result.exit_code == 0, result.output
    mock_fn.assert_called_once()


# ---------------------------------------------------------------------------
# Error cases
# ---------------------------------------------------------------------------

def test_export_cmd_unknown_format_exits_nonzero(runner, saved_model, tmp_path):
    out = str(tmp_path / "model.bin")
    result = runner.invoke(
        cli,
        ["export", "-m", saved_model, "-o", out, "--format", "awq"],
        catch_exceptions=False,
    )
    # Click rejects invalid choice before reaching our code
    assert result.exit_code != 0


def test_export_cmd_missing_format_and_unknown_ext_exits_nonzero(runner, saved_model, tmp_path):
    out = str(tmp_path / "model.bin")
    with patch("mono_quant.export.orchestrator.validate_export_pre", return_value=[]):
        result = runner.invoke(cli, ["export", "-m", saved_model, "-o", out])
    assert result.exit_code != 0
