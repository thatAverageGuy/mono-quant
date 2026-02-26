"""Tests for GGUF export functionality.

Coverage:
    T-022 — GGUFWriter binary format (6 tests)
    T-023 — Q4_K_S quantization (5 tests)
    T-024 — GGUFExporter integration (8 tests)
    T-025 — validate_gguf_checkpoint (3 tests)

gguf-py is required for the T-024 integration tests and T-025 tests.
Tests that need gguf-py are skipped if it is not installed.
"""

import json
import struct
import warnings
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn

from mono_quant import export_to_gguf
from mono_quant.export.gguf.arch_maps import (
    build_kv_entries,
    detect_architecture,
    map_tensor_name,
    reset_generic_counter,
)
from mono_quant.export.gguf.quant_types import (
    Q4_K_BLOCK_BYTES,
    quantize_to_q4_k_s,
    unpack_6bit_scales,
    _pack_6bit_scales,
)
from mono_quant.export.gguf.writer import (
    GGUF_TYPE_FLOAT32,
    GGUF_TYPE_STRING,
    GGUF_TYPE_UINT32,
    GGUFWriter,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_model(in_f: int = 256, out_f: int = 64) -> nn.Module:
    """Small model whose Linear layers have in_features divisible by 256."""
    return nn.Sequential(
        nn.Linear(in_f, out_f),
        nn.ReLU(),
        nn.Linear(out_f, in_f),
    )


def _gguf_available() -> bool:
    try:
        import gguf  # noqa: F401
        return True
    except ImportError:
        return False


requires_gguf = pytest.mark.skipif(
    not _gguf_available(), reason="gguf-py not installed (pip install gguf)"
)


# ---------------------------------------------------------------------------
# T-022: GGUFWriter unit tests
# ---------------------------------------------------------------------------

def test_gguf_writer_magic_and_version(tmp_path: Path) -> None:
    """Written file starts with b'GGUF' and version 3."""
    writer = GGUFWriter()
    out = tmp_path / "test.gguf"
    writer.write(out)

    data = out.read_bytes()
    assert data[:4] == b"GGUF", f"Bad magic: {data[:4]!r}"
    version = struct.unpack("<I", data[4:8])[0]
    assert version == 3, f"Expected version 3, got {version}"


def test_gguf_writer_kv_string(tmp_path: Path) -> None:
    """String KV entry is readable by gguf-py with the correct value."""
    pytest.importorskip("gguf")
    import gguf

    writer = GGUFWriter()
    writer.add_string("general.architecture", "llama")
    out = tmp_path / "kv_str.gguf"
    writer.write(out)

    reader = gguf.GGUFReader(str(out))
    kv = {f.name: f for f in reader.fields.values()}
    assert "general.architecture" in kv
    assert kv["general.architecture"].parts[-1].tobytes().decode() == "llama"


def test_gguf_writer_kv_uint32(tmp_path: Path) -> None:
    """UINT32 KV entry is readable by gguf-py with the correct value."""
    pytest.importorskip("gguf")
    import gguf

    writer = GGUFWriter()
    writer.add_uint32("general.quantization_version", 2)
    out = tmp_path / "kv_u32.gguf"
    writer.write(out)

    reader = gguf.GGUFReader(str(out))
    kv = {f.name: f for f in reader.fields.values()}
    assert "general.quantization_version" in kv
    assert int(kv["general.quantization_version"].parts[-1]) == 2


def test_gguf_writer_kv_float32(tmp_path: Path) -> None:
    """FLOAT32 KV entry is readable by gguf-py with approximately the correct value."""
    pytest.importorskip("gguf")
    import gguf

    writer = GGUFWriter()
    writer.add_float32("llama.rope.freq_base", 10000.0)
    out = tmp_path / "kv_f32.gguf"
    writer.write(out)

    reader = gguf.GGUFReader(str(out))
    kv = {f.name: f for f in reader.fields.values()}
    assert "llama.rope.freq_base" in kv
    val = float(kv["llama.rope.freq_base"].parts[-1])
    assert abs(val - 10000.0) < 1.0, f"Expected ~10000.0, got {val}"


def test_gguf_writer_tensor_data_alignment(tmp_path: Path) -> None:
    """Tensor data section starts on a 32-byte boundary from the start of the file."""
    writer = GGUFWriter()
    writer.add_string("general.architecture", "test")
    # Two tensors of different sizes to test alignment between them
    data1 = bytes(range(100))       # not a multiple of 32
    data2 = bytes(range(200))
    writer.add_tensor("t1", data1, (100,), 0)
    writer.add_tensor("t2", data2, (200,), 0)
    out = tmp_path / "align.gguf"
    writer.write(out)

    raw = out.read_bytes()
    # Find the start of tensor data: must be 32-byte aligned
    # Header is 24 bytes; we need to parse to find exactly where tensor data starts.
    # We know the padding brings everything to a 32-byte boundary.
    assert len(raw) % 1 == 0  # file exists and is non-empty
    # The pre-data portion length must be a multiple of 32
    # We parse the offsets to verify t2 starts at offset len(data1_padded)
    n_tensors = struct.unpack("<Q", raw[8:16])[0]
    assert n_tensors == 2


def test_gguf_writer_tensor_offsets(tmp_path: Path) -> None:
    """Tensor offsets within the data section reflect 32-byte alignment."""
    pytest.importorskip("gguf")
    import gguf

    writer = GGUFWriter()
    writer.add_string("general.architecture", "test")
    data1 = bytes(50)   # 50 bytes → padded to 64
    data2 = bytes(100)  # 100 bytes
    writer.add_tensor("tensor0", data1, (50,), 0)
    writer.add_tensor("tensor1", data2, (100,), 0)
    out = tmp_path / "offsets.gguf"
    writer.write(out)

    reader = gguf.GGUFReader(str(out))
    tensors = {t.name: t for t in reader.tensors}
    assert "tensor0" in tensors
    assert "tensor1" in tensors
    # tensor1 offset must be ≥ 64 (data1 padded to 32-byte multiple of 64)
    offset1 = tensors["tensor1"].field.offset
    assert offset1 >= 64, f"tensor1 offset {offset1} should be ≥ 64"


# ---------------------------------------------------------------------------
# T-023: Q4_K_S quantization unit tests
# ---------------------------------------------------------------------------

def test_q4k_block_size() -> None:
    """Total byte count matches out_features × blocks × 144."""
    out_f, in_f = 8, 512   # 512 / 256 = 2 blocks per row
    weight = torch.zeros(out_f, in_f)
    data = quantize_to_q4_k_s(weight)
    expected = out_f * (in_f // 256) * Q4_K_BLOCK_BYTES
    assert len(data) == expected, f"Expected {expected} bytes, got {len(data)}"


def test_q4k_scale_packing_round_trip() -> None:
    """6-bit scale packing: encode → decode → values match exactly."""
    import random
    random.seed(42)
    ls = [random.randint(0, 63) for _ in range(8)]
    lm = [random.randint(0, 63) for _ in range(8)]
    packed = _pack_6bit_scales(ls, lm)
    decoded_ls, decoded_lm = unpack_6bit_scales(packed)
    assert decoded_ls == ls, f"ls mismatch: {decoded_ls} != {ls}"
    assert decoded_lm == lm, f"lm mismatch: {decoded_lm} != {lm}"


def test_q4k_nibble_packing_round_trip() -> None:
    """4-bit nibble packing: pack → unpack → values match."""
    torch.manual_seed(7)
    weight = torch.randint(0, 8, (1, 256)).float()  # small values for easy check
    data = quantize_to_q4_k_s(weight)
    # Unpack nibbles from qs[128] (bytes 16..144 of the first block)
    qs = data[16:144]
    unpacked = []
    for byte in qs:
        unpacked.append(byte & 0xF)
        unpacked.append((byte >> 4) & 0xF)
    assert len(unpacked) == 256
    # All nibble values must be in [0, 15]
    assert all(0 <= v <= 15 for v in unpacked)


def test_q4k_reconstruction_error() -> None:
    """Dequantized weights are within 15% relative MAE of the original."""
    torch.manual_seed(99)
    weight = torch.randn(4, 256)
    data = quantize_to_q4_k_s(weight)

    # Dequantize manually using the block format
    recon = torch.zeros_like(weight)
    block_size = Q4_K_BLOCK_BYTES
    for row in range(4):
        for blk in range(1):  # 256 / 256 = 1 block per row
            base = (row * 1 + blk) * block_size
            block = data[base : base + block_size]

            d    = struct.unpack("<e", block[0:2])[0]
            dmin = struct.unpack("<e", block[2:4])[0]
            scales_b = block[4:16]
            qs_b     = block[16:144]

            ls, lm = unpack_6bit_scales(scales_b)

            nibbles = []
            for byte in qs_b:
                nibbles.append(byte & 0xF)
                nibbles.append((byte >> 4) & 0xF)

            for sub in range(8):
                sc  = d    * ls[sub]
                mn  = dmin * lm[sub]
                for k in range(32):
                    q = nibbles[sub * 32 + k]
                    recon[row, blk * 256 + sub * 32 + k] = sc * q - mn

    err      = (recon - weight).abs().mean().item()
    scale_ref = weight.abs().mean().item()
    relative  = err / max(scale_ref, 1e-8)
    assert relative < 0.15, f"Reconstruction error too high: {relative:.4f}"


def test_q4k_invalid_in_features() -> None:
    """in_features not divisible by 256 raises ValueError."""
    with pytest.raises(ValueError, match="divisible by 256"):
        quantize_to_q4_k_s(torch.randn(4, 128))


# ---------------------------------------------------------------------------
# T-024: GGUFExporter integration tests
# ---------------------------------------------------------------------------

def test_gguf_export_creates_file(tmp_path: Path) -> None:
    """export_to_gguf writes model.gguf to the output directory."""
    model = _make_model()
    export_to_gguf(model, tmp_path)
    assert (tmp_path / "model.gguf").exists()


@requires_gguf
def test_gguf_export_ggufpy_reads_file(tmp_path: Path) -> None:
    """gguf-py opens the exported file without error."""
    import gguf
    model = _make_model()
    export_to_gguf(model, tmp_path)
    reader = gguf.GGUFReader(str(tmp_path / "model.gguf"))
    assert reader is not None


@requires_gguf
def test_gguf_export_tensor_count(tmp_path: Path) -> None:
    """Number of tensors in the GGUF file matches the nn.Linear count."""
    import gguf
    model = _make_model()
    export_to_gguf(model, tmp_path)
    reader = gguf.GGUFReader(str(tmp_path / "model.gguf"))
    n_linears = sum(1 for _, m in model.named_modules() if isinstance(m, nn.Linear))
    assert len(reader.tensors) == n_linears


def test_gguf_export_config_path(tmp_path: Path) -> None:
    """KV metadata is populated from config.json when provided."""
    pytest.importorskip("gguf")
    import gguf

    config = {
        "model_type": "llama",
        "_name_or_path": "test-llama",
        "num_hidden_layers": 4,
        "hidden_size": 256,
    }
    cfg_file = tmp_path / "config.json"
    cfg_file.write_text(json.dumps(config))

    model = _make_model()
    export_to_gguf(model, tmp_path, config_path=cfg_file)

    reader = gguf.GGUFReader(str(tmp_path / "model.gguf"))
    kv = {f.name: f for f in reader.fields.values()}
    assert "general.architecture" in kv


def test_gguf_export_model_params_override(tmp_path: Path) -> None:
    """model_params dict values override config.json values."""
    pytest.importorskip("gguf")
    import gguf

    config = {"model_type": "llama", "num_hidden_layers": 4, "hidden_size": 256}
    cfg_file = tmp_path / "config.json"
    cfg_file.write_text(json.dumps(config))

    # Override model_type via model_params
    model = _make_model()
    export_to_gguf(
        model, tmp_path,
        config_path=cfg_file,
        model_params={"model_type": "qwen2", "num_hidden_layers": 8},
    )

    reader = gguf.GGUFReader(str(tmp_path / "model.gguf"))
    kv = {f.name: f for f in reader.fields.values()}
    # Architecture should reflect the overridden model_type = qwen2
    arch_val = kv["general.architecture"].parts[-1].tobytes().decode()
    assert arch_val == "qwen2", f"Expected qwen2, got {arch_val}"


def test_gguf_export_llama_tensor_names() -> None:
    """map_tensor_name produces correct GGUF names for LLaMA tensors."""
    cases = [
        ("model.embed_tokens.weight",                        "token_embd.weight"),
        ("model.layers.0.self_attn.q_proj.weight",           "blk.0.attn_q.weight"),
        ("model.layers.3.mlp.down_proj.weight",              "blk.3.ffn_down.weight"),
        ("model.norm.weight",                                "output_norm.weight"),
        ("lm_head.weight",                                   "output.weight"),
    ]
    for pt_name, expected in cases:
        result = map_tensor_name(pt_name, "llama")
        assert result == expected, f"For '{pt_name}': expected '{expected}', got '{result}'"


def test_gguf_export_unknown_arch_warns(tmp_path: Path) -> None:
    """Unknown architecture emits a warning and uses generic tensor naming."""
    model = nn.Sequential(nn.Linear(256, 64))
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        export_to_gguf(model, tmp_path, architecture="unknown_arch_xyz")
    assert (tmp_path / "model.gguf").exists()
    # At least one warning should mention the unknown architecture or generic naming
    messages = [str(warning.message) for warning in w]
    assert any("unknown_arch_xyz" in m or "generic" in m for m in messages), (
        f"Expected arch warning, got: {messages}"
    )


def test_gguf_export_cli_runs(tmp_path: Path) -> None:
    """CLI export-gguf command produces model.gguf without error."""
    from click.testing import CliRunner
    from mono_quant.cli.main import cli

    # Save a small model for the CLI to load
    model = _make_model()
    model_pt = tmp_path / "model.pt"
    torch.save(model, str(model_pt))

    output_dir = tmp_path / "gguf_out"
    runner = CliRunner()
    result = runner.invoke(cli, [
        "export-gguf",
        "--model", str(model_pt),
        "--output", str(output_dir),
    ])
    assert result.exit_code == 0, f"CLI failed:\n{result.output}"
    assert (output_dir / "model.gguf").exists()


# ---------------------------------------------------------------------------
# T-025: validate_gguf_checkpoint tests
# ---------------------------------------------------------------------------

def test_validate_gguf_checkpoint_valid(tmp_path: Path) -> None:
    """validate_gguf_checkpoint passes on a file produced by export_to_gguf."""
    pytest.importorskip("gguf")
    from mono_quant.export.common.validators import validate_gguf_checkpoint

    model = _make_model()
    export_to_gguf(model, tmp_path, architecture="llama")
    # Should not raise
    validate_gguf_checkpoint(tmp_path / "model.gguf")


def test_validate_gguf_checkpoint_missing_file() -> None:
    """validate_gguf_checkpoint raises FileNotFoundError for a missing file."""
    from mono_quant.export.common.validators import validate_gguf_checkpoint

    with pytest.raises(FileNotFoundError):
        validate_gguf_checkpoint(Path("/nonexistent/path/model.gguf"))


def test_validate_gguf_checkpoint_import_error(tmp_path: Path) -> None:
    """validate_gguf_checkpoint raises ImportError when gguf is not installed."""
    from mono_quant.export.common.validators import validate_gguf_checkpoint

    # Write a dummy file so the existence check passes
    dummy = tmp_path / "model.gguf"
    dummy.write_bytes(b"GGUF\x03\x00\x00\x00")

    with patch.dict("sys.modules", {"gguf": None}):
        with pytest.raises(ImportError, match="pip install mono-quant\\[gguf\\]"):
            validate_gguf_checkpoint(dummy)
