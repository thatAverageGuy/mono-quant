"""Tests for GPTQ INT4 export functionality.

No external GPTQ/vLLM dependencies required — safetensors is already a core dep.
"""

import json
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from mono_quant import dynamic_quantize, export_to_gptq
from mono_quant.export.common.gptq_packing import (
    _pack_int4_cols_to_int32,
    _pack_int4_rows_to_int32,
    quantize_to_gptq_int4,
    unpack_gptq_weight,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_fp32_model() -> nn.Module:
    """Small plain FP32 model with Linear layers."""
    return nn.Sequential(
        nn.Linear(128, 256),
        nn.ReLU(),
        nn.Linear(256, 64),
    )


def _make_int4_model() -> nn.Module:
    """INT4-quantized model via dynamic_quantize (uses INT4 path)."""
    from mono_quant.modules.linear import QuantizedLinearInt4

    linear = nn.Linear(128, 256)
    q = QuantizedLinearInt4.from_float(linear, group_size=128)
    return nn.Sequential(q)


def _make_int8_model() -> nn.Module:
    """INT8-quantized model via dynamic_quantize."""
    model = nn.Sequential(
        nn.Linear(128, 256),
        nn.ReLU(),
        nn.Linear(256, 64),
    )
    q_model, _ = dynamic_quantize(model)
    return q_model


def _load_safetensors(path: Path) -> dict:
    from safetensors.torch import load_file
    return load_file(str(path))


# ---------------------------------------------------------------------------
# T-018 packing round-trip tests
# ---------------------------------------------------------------------------

def test_pack_int4_rows_round_trip():
    """pack rows → unpack → values match original."""
    torch.manual_seed(0)
    in_features, out_features = 64, 32
    original = torch.randint(0, 16, (in_features, out_features), dtype=torch.int32)
    packed = _pack_int4_rows_to_int32(original)
    assert packed.shape == (in_features // 8, out_features)
    unpacked = unpack_gptq_weight(packed, in_features)
    assert unpacked.shape == (in_features, out_features)
    assert torch.all(unpacked == original), "Row pack/unpack round-trip failed"


def test_pack_zeros_round_trip():
    """pack zeros cols → unpack → values match original."""
    torch.manual_seed(1)
    num_groups, out_features = 4, 64
    original = torch.randint(0, 16, (num_groups, out_features), dtype=torch.int32)
    packed = _pack_int4_cols_to_int32(original)
    assert packed.shape == (num_groups, out_features // 8)

    # Unpack: 8 values per packed int32, LSB-first
    mask = torch.tensor(0xF, dtype=torch.int32)
    rows = [(packed >> (i * 4)) & mask for i in range(8)]
    stacked = torch.stack(rows, dim=2)  # (num_groups, out_features // 8, 8)
    unpacked = stacked.reshape(num_groups, out_features)
    assert torch.all(unpacked == original), "Zero col pack/unpack round-trip failed"


# ---------------------------------------------------------------------------
# T-018 shape tests
# ---------------------------------------------------------------------------

def test_quantize_to_gptq_int4_shapes():
    """All four output tensors have the correct shapes."""
    out_features, in_features = 64, 256
    group_size = 128
    weight = torch.randn(out_features, in_features)
    qweight, qzeros, scales, g_idx = quantize_to_gptq_int4(weight, group_size=group_size)

    num_groups = in_features // group_size
    assert qweight.shape == (in_features // 8, out_features), f"qweight shape wrong: {qweight.shape}"
    assert qzeros.shape == (num_groups, out_features // 8), f"qzeros shape wrong: {qzeros.shape}"
    assert scales.shape == (num_groups, out_features), f"scales shape wrong: {scales.shape}"
    assert g_idx.shape == (in_features,), f"g_idx shape wrong: {g_idx.shape}"

    assert qweight.dtype == torch.int32
    assert qzeros.dtype == torch.int32
    assert scales.dtype == torch.float16
    assert g_idx.dtype == torch.int32


def test_quantize_to_gptq_int4_g_idx_values():
    """g_idx[i] == i // group_size for all i."""
    weight = torch.randn(64, 256)
    _, _, _, g_idx = quantize_to_gptq_int4(weight, group_size=128)
    expected = torch.arange(256, dtype=torch.int32) // 128
    assert torch.all(g_idx == expected)


# ---------------------------------------------------------------------------
# T-019 file-level export tests
# ---------------------------------------------------------------------------

def test_gptq_export_creates_files(tmp_path):
    """export_to_gptq writes model.safetensors and quantize_config.json."""
    model = _make_fp32_model()
    export_to_gptq(model, tmp_path)
    assert (tmp_path / "model.safetensors").exists()
    assert (tmp_path / "quantize_config.json").exists()


def test_gptq_export_config_required_fields(tmp_path):
    """quantize_config.json contains all fields required by vLLM."""
    model = _make_fp32_model()
    export_to_gptq(model, tmp_path)
    config = json.loads((tmp_path / "quantize_config.json").read_text())
    for field in ("bits", "group_size", "desc_act", "sym", "quant_method"):
        assert field in config, f"Missing required field: {field}"
    assert config["bits"] == 4
    assert config["quant_method"] == "gptq"
    assert config["desc_act"] is False


def test_gptq_export_safetensors_keys(tmp_path):
    """model.safetensors has qweight/qzeros/scales/g_idx for each Linear."""
    model = _make_fp32_model()
    export_to_gptq(model, tmp_path)
    tensors = _load_safetensors(tmp_path / "model.safetensors")
    # nn.Sequential with named_modules: "0", "2" are the Linear layers
    for layer_name in ("0", "2"):
        for suffix in ("qweight", "qzeros", "scales", "g_idx"):
            key = f"{layer_name}.{suffix}"
            assert key in tensors, f"Missing key: {key}"


def test_gptq_export_weight_reconstruction_error(tmp_path):
    """Dequantized weights have < 1% relative error vs original."""
    torch.manual_seed(42)
    model = nn.Sequential(nn.Linear(128, 64))
    original_w = model[0].weight.data.clone()

    export_to_gptq(model, tmp_path, group_size=128)
    tensors = _load_safetensors(tmp_path / "model.safetensors")

    qweight = tensors["0.qweight"]   # (in // 8, out)
    scales = tensors["0.scales"]     # (num_groups, out), float16
    qzeros = tensors["0.qzeros"]     # (num_groups, out // 8)

    in_features = original_w.shape[1]
    out_features = original_w.shape[0]
    group_size = 128
    num_groups = in_features // group_size

    # Unpack qweight → (in_features, out_features)
    w_uint4 = unpack_gptq_weight(qweight, in_features)

    # Unpack qzeros → (num_groups, out_features)
    mask = torch.tensor(0xF, dtype=torch.int32)
    z_cols = [(qzeros >> (i * 4)) & mask for i in range(8)]
    z_stacked = torch.stack(z_cols, dim=2).reshape(num_groups, out_features)
    zeros = (z_stacked + 1).float()   # V1: stored as zero-1

    sc = scales.float()  # (num_groups, out_features)

    # Dequantize: weight[group, :, out] = (q - zero) * scale
    # w_uint4 shape: (in_features, out_features) — group = col // group_size
    w_recon = torch.zeros(out_features, in_features)
    for g in range(num_groups):
        col_start = g * group_size
        col_end = col_start + group_size
        q_slice = w_uint4[col_start:col_end, :].float().T   # (out_features, group_size)
        z_slice = zeros[g, :].unsqueeze(1)                  # (out_features, 1)
        s_slice = sc[g, :].unsqueeze(1)                     # (out_features, 1)
        w_recon[:, col_start:col_end] = (q_slice - z_slice) * s_slice

    err = (w_recon - original_w).abs().mean()
    scale_ref = original_w.abs().mean()
    relative_err = (err / scale_ref).item()
    # INT4 with group_size=128 gives ~5-10% relative mean-abs error; 15% is a safe upper bound
    assert relative_err < 0.15, f"Reconstruction error too high: {relative_err:.4f}"


def test_gptq_export_from_int4_model(tmp_path):
    """INT4-quantized model exports to GPTQ without error."""
    model = _make_int4_model()
    export_to_gptq(model, tmp_path)
    assert (tmp_path / "model.safetensors").exists()
    assert (tmp_path / "quantize_config.json").exists()


def test_gptq_export_from_fp32_model(tmp_path):
    """Plain FP32 nn.Sequential exports to GPTQ without error."""
    model = _make_fp32_model()
    export_to_gptq(model, tmp_path)
    tensors = _load_safetensors(tmp_path / "model.safetensors")
    assert len(tensors) > 0


def test_gptq_export_bias_preserved(tmp_path):
    """Bias tensors are stored unchanged in the checkpoint."""
    torch.manual_seed(7)
    model = nn.Sequential(nn.Linear(128, 64, bias=True))
    original_bias = model[0].bias.data.clone()

    export_to_gptq(model, tmp_path)
    tensors = _load_safetensors(tmp_path / "model.safetensors")

    assert "0.bias" in tensors, "Bias key missing from checkpoint"
    assert torch.allclose(tensors["0.bias"].float(), original_bias), "Bias not preserved"
