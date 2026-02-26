"""Unified export orchestrator for mono-quant.

Dispatches to the ONNX, GPTQ, or GGUF exporters from a single call.
"""

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch.nn as nn

from mono_quant.export.common.validators import validate_export_pre, validate_export_post


# ---------------------------------------------------------------------------
# Format registry
# ---------------------------------------------------------------------------

FORMAT_MAP: Dict[str, str] = {
    "onnx": "ONNX with QDQ nodes (.onnx file)",
    "gptq": "GPTQ INT4 AutoGPTQ V1 format (directory)",
    "gguf": "GGUF v3 binary for llama.cpp (.gguf file)",
}

# Extension → format mapping (lower-case suffix, including the dot)
_EXTENSION_MAP: Dict[str, str] = {
    ".onnx": "onnx",
    ".gguf": "gguf",
    # No entry for gptq — it is the fallback when nothing else matches.
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _detect_format(path: Union[str, Path]) -> str:
    """Infer export format from *path*.

    Rules:
    - ``.onnx`` suffix → ``"onnx"``
    - ``.gguf`` suffix → ``"gguf"``
    - Path ends with ``/`` or ``\\``, or has no recognised suffix → ``"gptq"``

    Returns:
        One of ``"onnx"``, ``"gptq"``, ``"gguf"``.

    Raises:
        ValueError: If the suffix is non-empty and not in the known map.
    """
    p = Path(str(path))
    suffix = p.suffix.lower()

    if suffix in _EXTENSION_MAP:
        return _EXTENSION_MAP[suffix]

    # Unknown non-empty suffix (e.g. .safetensors, .bin) → error.
    if suffix and suffix not in ("",):
        raise ValueError(
            f"Cannot detect export format from path {str(path)!r} "
            f"(unrecognised extension {suffix!r}). "
            f"Pass --format explicitly, or use a .onnx / .gguf extension, "
            f"or a bare directory path for GPTQ."
        )

    # No extension (or trailing separator) → default to gptq directory export.
    return "gptq"


def list_formats() -> Dict[str, str]:
    """Return a copy of the supported format registry.

    Returns:
        dict mapping format name → human-readable description.
    """
    return dict(FORMAT_MAP)


# ---------------------------------------------------------------------------
# Main dispatch
# ---------------------------------------------------------------------------

def export_model(
    model: nn.Module,
    path: Union[str, Path],
    format: Optional[str] = None,  # noqa: A002  (shadowing built-in intentionally)
    info: Any = None,
    **options: Any,
) -> None:
    """Export *model* to the requested format.

    Args:
        model:  Any ``nn.Module`` (quantized or plain FP32).
        path:   Destination path. Interpretation depends on format:
                - ONNX: path to ``.onnx`` file.
                - GPTQ: output directory.
                - GGUF: output directory (``model.gguf`` written inside).
        format: Export format — ``"onnx"``, ``"gptq"``, or ``"gguf"``.
                If *None*, inferred from *path* extension.
        info:   Optional ``QuantizationInfo`` for pre-validation heuristics.
        **options:
            ONNX: ``opset`` (int, default 14), ``validate`` (str, default "none"),
                  ``dummy_input`` (Tensor | None).
            GPTQ: ``group_size`` (int, default 128), ``sym`` (bool, default False).
            GGUF: ``quantization_type`` (str), ``architecture`` (str | None),
                  ``config_path`` (str | Path | None),
                  ``model_params`` (dict | None).
            Shared: ``validate_post`` (bool, default False) — run post-export
                  structural validation after the exporter finishes.

    Raises:
        ValueError: If *format* is not in ``FORMAT_MAP`` or cannot be inferred.
        TypeError:  If *model* is not an ``nn.Module``.
        ImportError: If optional dependencies for the requested format are absent.
    """
    if not isinstance(model, nn.Module):
        raise TypeError(
            f"export_model requires an nn.Module, got {type(model).__name__!r}"
        )

    # Resolve format
    if format is None:
        fmt = _detect_format(path)
    else:
        fmt = format.lower()
        if fmt not in FORMAT_MAP:
            raise ValueError(
                f"Unknown export format {fmt!r}. "
                f"Supported: {', '.join(FORMAT_MAP)}"
            )

    # Pre-export validation
    warnings = validate_export_pre(model, info, fmt)
    for w in warnings:
        level_prefix = "ERROR" if w.level == "error" else "WARNING"
        print(f"[export:{level_prefix}] {w.message}", file=sys.stderr)

    # Dispatch
    if fmt == "onnx":
        _export_onnx(model, path, **options)
    elif fmt == "gptq":
        _export_gptq(model, path, **options)
    elif fmt == "gguf":
        _export_gguf(model, path, **options)

    # Post-export validation
    validate_post = options.pop("validate_post", False)
    if validate_post:
        validate_export_post(path, fmt)


# ---------------------------------------------------------------------------
# Format-specific dispatch helpers
# ---------------------------------------------------------------------------

def _export_onnx(model: nn.Module, path: Union[str, Path], **options: Any) -> None:
    try:
        import onnx  # noqa: F401
    except ImportError:
        raise ImportError(
            "ONNX export requires additional dependencies. "
            "Install with: pip install mono-quant[onnx]"
        )
    from mono_quant.export.onnx_impl import export_to_onnx_impl

    export_to_onnx_impl(
        model=model,
        path=path,
        opset=options.get("opset", 14),
        dummy_input=options.get("dummy_input", None),
        validate=options.get("validate", "none"),
    )


def _export_gptq(model: nn.Module, path: Union[str, Path], **options: Any) -> None:
    from mono_quant.export.gptq import GPTQExporter

    GPTQExporter().export(
        model,
        path,
        group_size=options.get("group_size", 128),
        sym=options.get("sym", False),
    )


def _export_gguf(model: nn.Module, path: Union[str, Path], **options: Any) -> None:
    from mono_quant.export.gguf.exporter import GGUFExporter

    GGUFExporter().export(
        model,
        path,
        quantization_type=options.get("quantization_type", "q4_k_s"),
        architecture=options.get("architecture", None),
        config_path=options.get("config_path", None),
        model_params=options.get("model_params", None),
    )


__all__ = [
    "FORMAT_MAP",
    "export_model",
    "list_formats",
    "validate_export_pre",
    "validate_export_post",
    "_detect_format",  # exported for tests
]
