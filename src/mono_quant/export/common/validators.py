"""Export validation utilities (ONNX, GPTQ, GGUF)."""

import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, List, Optional, Union


class ValidationLevel(str, Enum):
    NONE = "none"
    LOAD = "load"
    FULL = "full"


# ---------------------------------------------------------------------------
# T-028 — Pre/post export validation
# ---------------------------------------------------------------------------

@dataclass
class ExportWarning:
    """A single pre-export validation finding."""

    level: str      # "error" | "warning"
    message: str
    check: str      # short identifier, e.g. "no_quantized_layers"


def validate_export_pre(
    model: Any,
    info: Any,
    format: str,  # noqa: A002
) -> List[ExportWarning]:
    """Run pre-export compatibility checks.

    Args:
        model:  The nn.Module about to be exported.
        info:   Optional ``QuantizationInfo`` (may be None).
        format: Target export format — ``"onnx"``, ``"gptq"``, or ``"gguf"``.

    Returns:
        List of :class:`ExportWarning`. Empty means all checks passed.
    """
    import torch
    import torch.nn as nn

    findings: List[ExportWarning] = []

    if not isinstance(model, nn.Module):
        return findings  # type error handled by orchestrator

    # Check 1: model has at least one quantized layer.
    # Two valid cases:
    #   a) INT8/INT4: model contains QuantizedLinear/QuantizedConv2d/etc. instances.
    #      (QuantizedLinear stores its weight as a plain tensor attribute, NOT as
    #      nn.Parameter, so checking parameter dtypes would always return False.)
    #   b) FP16: all parameters were cast to float16 in-place; no wrapper modules used.
    from mono_quant.modules.linear import (
        QuantizedLinear, QuantizedConv2d, QuantizedLinearInt4,
    )
    from mono_quant.modules.embedding import QuantizedEmbedding
    _QUANTIZED_TYPES = (QuantizedLinear, QuantizedConv2d, QuantizedLinearInt4, QuantizedEmbedding)
    has_quantized = (
        any(isinstance(m, _QUANTIZED_TYPES) for m in model.modules())
        or any(p.dtype == torch.float16 for p in model.parameters())
    )
    if not has_quantized:
        findings.append(ExportWarning(
            level="warning",
            message="Model has no quantized parameters — exporting a plain FP32 model.",
            check="no_quantized_layers",
        ))

    # Check 2: GPTQ requires INT4-style quantization; warn for INT8-only models
    if format == "gptq" and info is not None:
        dtype = getattr(info, "dtype", None)
        if dtype == torch.qint8:
            findings.append(ExportWarning(
                level="error",
                message=(
                    "GPTQ export targets INT4; the model was quantized to INT8. "
                    "The GPTQExporter will re-quantize weights to INT4 automatically, "
                    "but accuracy may be lower than a direct INT4 quantization."
                ),
                check="int8_gptq_mismatch",
            ))

    # Check 3: ONNX + INT4 dtype + opset < 21 → warn about representation fallback
    if format == "onnx" and info is not None:
        dtype = getattr(info, "dtype", None)
        # INT4 is represented in mono-quant as qint8 with 4-bit range — no direct signal,
        # so we check for bits=4 via info fields if available.
        bits = getattr(info, "bits", None)
        if bits == 4:
            findings.append(ExportWarning(
                level="warning",
                message=(
                    "ONNX INT4 requires opset 21+ for native INT4 representation. "
                    "Lower opsets fall back to INT8 representation in the ONNX graph."
                ),
                check="onnx_int4_opset",
            ))

    return findings


def validate_export_post(path: Union[str, Path], format: str) -> None:  # noqa: A002
    """Run post-export structural validation.

    Dispatches to the format-specific validator.

    Args:
        path:   Path to the exported artefact (file or directory).
        format: One of ``"onnx"``, ``"gptq"``, ``"gguf"``.

    Raises:
        ValueError: If format is unknown.
    """
    if format == "onnx":
        validate_onnx_model(path, level=ValidationLevel.LOAD)
    elif format == "gptq":
        validate_gptq_checkpoint_structure(path)
    elif format == "gguf":
        validate_gguf_checkpoint(path)
    else:
        raise ValueError(f"Unknown format for post-export validation: {format!r}")


def validate_onnx_model(
    path: Union[str, Path],
    level: Union[ValidationLevel, str] = ValidationLevel.NONE,
) -> None:
    """Validate an exported ONNX model.

    Args:
        path: Path to the ONNX model file.
        level: Validation level — none, load, or full.

    Raises:
        ImportError: If onnx/onnxruntime is not installed for the requested level.
        ValueError: If the ONNX checker rejects the model.
    """
    level = ValidationLevel(level)

    if level == ValidationLevel.NONE:
        return

    try:
        import onnx
    except ImportError:
        raise ImportError(
            "Validation requires onnx. Install with: pip install mono-quant[onnx]"
        )

    model_proto = onnx.load(str(path))
    onnx.checker.check_model(model_proto)

    if level == ValidationLevel.FULL:
        try:
            import onnxruntime as ort
        except ImportError:
            raise ImportError(
                "Full validation requires onnxruntime. "
                "Install with: pip install mono-quant[onnx]"
            )

        import numpy as np

        # Map ONNX type strings to numpy dtypes. Fallback: float32.
        _ONNX_TO_NP = {
            "tensor(float)":   np.float32,
            "tensor(float16)": np.float16,
            "tensor(double)":  np.float64,
            "tensor(int64)":   np.int64,
            "tensor(int32)":   np.int32,
            "tensor(int16)":   np.int16,
            "tensor(int8)":    np.int8,
            "tensor(uint8)":   np.uint8,
            "tensor(bool)":    np.bool_,
        }

        session = ort.InferenceSession(str(path))
        inputs = session.get_inputs()
        input_shape = inputs[0].shape
        # Replace dynamic (symbolic) dims with 1
        concrete_shape = [d if isinstance(d, int) and d > 0 else 1 for d in input_shape]
        input_dtype = _ONNX_TO_NP.get(inputs[0].type, np.float32)
        dummy = np.zeros(concrete_shape, dtype=input_dtype)
        session.run(None, {inputs[0].name: dummy})


def validate_gguf_checkpoint(path: Union[str, Path]) -> None:
    """Validate a GGUF checkpoint using gguf-py (Level 1 automated check).

    Checks:
    - File exists
    - gguf-py can open and parse the file without error
    - File contains at least one tensor
    - general.architecture KV key is present

    Args:
        path: Path to the .gguf file.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file fails structural validation.
        ImportError: If gguf is not installed (with install hint).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"GGUF file not found: {path}")

    try:
        import gguf  # noqa: F401
    except ImportError:
        raise ImportError(
            "GGUF validation requires gguf-py. "
            "Install with: pip install mono-quant[gguf]"
        )

    try:
        reader = gguf.GGUFReader(str(path))
    except Exception as e:
        raise ValueError(f"gguf-py failed to parse {path}: {e}") from e

    if len(reader.tensors) == 0:
        raise ValueError(f"GGUF file contains no tensors: {path}")

    kv_keys = {field.name for field in reader.fields.values()}
    if "general.architecture" not in kv_keys:
        raise ValueError(
            f"GGUF file missing required KV key 'general.architecture': {path}"
        )


_REQUIRED_CONFIG_FIELDS = {"bits", "group_size", "desc_act", "sym", "quant_method"}


def validate_gptq_checkpoint_structure(path: Union[str, Path]) -> None:
    """Validate the structure of a GPTQ checkpoint directory.

    Pure Python, no vLLM required.

    Checks:
    - Directory exists
    - model.safetensors and quantize_config.json are present
    - quantize_config.json contains all required fields
    - model.safetensors contains at least one .qweight key

    Args:
        path: Path to the GPTQ checkpoint directory.

    Raises:
        FileNotFoundError: If the directory or required files are missing.
        ValueError: If required config fields are absent or no .qweight key found.
    """
    path = Path(path)

    if not path.is_dir():
        raise FileNotFoundError(f"GPTQ checkpoint directory not found: {path}")

    safetensors_path = path / "model.safetensors"
    config_path = path / "quantize_config.json"

    if not safetensors_path.exists():
        raise FileNotFoundError(f"model.safetensors not found in {path}")
    if not config_path.exists():
        raise FileNotFoundError(f"quantize_config.json not found in {path}")

    config = json.loads(config_path.read_text())
    missing = _REQUIRED_CONFIG_FIELDS - config.keys()
    if missing:
        raise ValueError(f"quantize_config.json is missing required fields: {missing}")

    # Inspect safetensors header without loading full tensors
    from safetensors import safe_open

    with safe_open(str(safetensors_path), framework="pt", device="cpu") as f:
        keys = list(f.keys())

    qweight_keys = [k for k in keys if k.endswith(".qweight")]
    if not qweight_keys:
        raise ValueError(
            f"model.safetensors contains no .qweight keys — "
            f"found keys: {keys[:10]}{'...' if len(keys) > 10 else ''}"
        )
