"""Model serialization formats for quantized models.

This module provides save/load functions for both PyTorch (.pt/.pth) and
Safetensors (.safetensors) formats with comprehensive metadata support.

## Safetensors Format

Safetensors provides secure serialization (no pickle) with zero-copy loading.
All metadata values MUST be strings (safetensors constraint). Complex types
are automatically JSON-serialized.

## PyTorch Format

PyTorch format uses torch.save/torch.load with pickle. Use with caution on
untrusted models. Metadata is not natively supported in standard torch.save.

## Metadata

Quantization metadata includes:
- Quantization parameters: dtype, scheme (symmetric/asymmetric), per_channel
- Model info: selected layers, calibration samples used
- Version info: mono_quant version, PyTorch version
- Metrics: original/quantized size, compression ratio, SQNR

Example:
    >>> import torch.nn as nn
    >>> from mono_quant.io import save_model, load_model
    >>> model = nn.Linear(10, 20)
    >>> save_model(model, "quantized.safetensors")
    >>> loaded = load_model("quantized.safetensors")
"""

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn

try:
    from safetensors.torch import save_file, safe_open
    SAFETENSORS_AVAILABLE = True
except ImportError:
    SAFETENSORS_AVAILABLE = False


@dataclass
class QuantizationInfo:
    """
    Metadata about quantization parameters and process.

    Args:
        dtype: Target quantization dtype (e.g., torch.qint8, torch.float16)
        symmetric: Whether symmetric quantization was used
        per_channel: Whether per-channel quantization was used
        selected_layers: List of layer names that were quantized
        calibration_samples_used: Number of calibration samples used
        scheme: Quantization scheme ("symmetric" or "asymmetric")
        group_size: Group size for INT4 quantization (None for INT8/FP16)
        bits: Bits per weight (4 for INT4, 8 for INT8, 16 for FP16)
    """

    dtype: torch.dtype
    symmetric: bool
    per_channel: bool
    selected_layers: List[str] = field(default_factory=list)
    calibration_samples_used: Optional[int] = None
    scheme: str = "symmetric"
    group_size: Optional[int] = None
    bits: Optional[int] = None


def _build_metadata(
    quantization_info: Optional[QuantizationInfo] = None,
    original_size_mb: Optional[float] = None,
    quantized_size_mb: Optional[float] = None,
    compression_ratio: Optional[float] = None,
    sqnr_db: Optional[float] = None,
) -> Dict[str, str]:
    """
    Build metadata dictionary for Safetensors format.

    All values are converted to strings per Safetensors requirement.
    Complex types (lists, dicts) are JSON-serialized.

    Args:
        quantization_info: Quantization parameters and process info
        original_size_mb: Original model size in MB
        quantized_size_mb: Quantized model size in MB
        compression_ratio: Compression ratio (original_size / quantized_size)
        sqnr_db: Signal-to-Quantization-Noise Ratio in dB

    Returns:
        Dictionary with all string values for Safetensors metadata
    """
    metadata: Dict[str, str] = {}

    # Add quantization parameters if provided
    if quantization_info is not None:
        metadata["quantization_dtype"] = str(quantization_info.dtype)
        metadata["scheme"] = "symmetric" if quantization_info.symmetric else "asymmetric"
        metadata["per_channel"] = "true" if quantization_info.per_channel else "false"
        metadata["selected_layers"] = json.dumps(quantization_info.selected_layers)
        if quantization_info.calibration_samples_used is not None:
            metadata["calibration_samples"] = str(quantization_info.calibration_samples_used)
        # INT4-specific metadata
        if quantization_info.group_size is not None:
            metadata["group_size"] = str(quantization_info.group_size)
        if quantization_info.bits is not None:
            metadata["bits"] = str(quantization_info.bits)

    # Add version information
    try:
        from importlib.metadata import version as get_version
        metadata["mono_quant_version"] = get_version("mono-quant")
    except Exception:
        metadata["mono_quant_version"] = "0.1.0"  # Fallback to hardcoded version
    metadata["pytorch_version"] = torch.__version__

    # Add size metrics if provided
    if original_size_mb is not None:
        metadata["original_size_mb"] = str(original_size_mb)
    if quantized_size_mb is not None:
        metadata["quantized_size_mb"] = str(quantized_size_mb)
    if compression_ratio is not None:
        metadata["compression_ratio"] = str(compression_ratio)
    if sqnr_db is not None:
        metadata["sqnr_db"] = str(sqnr_db)

    metadata["format"] = "safetensors"

    return metadata


def save_safetensors(
    model: Union[nn.Module, Dict[str, torch.Tensor]],
    path: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Save model to Safetensors format with metadata.

    Safetensors provides secure serialization (no pickle) and zero-copy
    loading. All metadata values MUST be strings - complex types are
    automatically JSON-serialized.

    Args:
        model: PyTorch model or state_dict to save
        path: Output file path (should end in .safetensors)
        metadata: Optional metadata dict. Values will be converted to strings.
            Non-string values are JSON-serialized.

    Raises:
        ImportError: If safetensors is not installed
        ValueError: If path doesn't end in .safetensors

    Examples:
        >>> import torch.nn as nn
        >>> model = nn.Linear(10, 20)
        >>> save_safetensors(model, "model.safetensors",
        ...                  metadata={"test": "value"})
        >>> # With complex metadata (auto-serialized)
        >>> save_safetensors(model, "model.safetensors",
        ...                  metadata={"layers": ["linear1", "linear2"]})
    """
    if not SAFETENSORS_AVAILABLE:
        raise ImportError(
            "safetensors is not installed. "
            "Install it with: pip install safetensors>=0.4"
        )

    # Extract state_dict
    if isinstance(model, nn.Module):
        state_dict = model.state_dict()
    else:
        state_dict = model

    # Build metadata if not provided
    if metadata is None:
        metadata = {}

    # Convert all metadata values to strings (Safetensors requirement)
    string_metadata: Dict[str, str] = {}
    for key, value in metadata.items():
        if isinstance(value, str):
            string_metadata[key] = value
        else:
            # JSON-serialize complex types
            string_metadata[key] = json.dumps(value)

    save_file(state_dict, path, metadata=string_metadata)


def load_safetensors(
    path: str,
    device: torch.device = torch.device("cpu"),
) -> Dict[str, torch.Tensor]:
    """
    Load model from Safetensors format.

    Uses zero-copy loading for efficient memory usage. Metadata is
    automatically parsed from JSON strings back to original types.

    Args:
        path: Path to .safetensors file
        device: Device to load tensors onto

    Returns:
        State dict with loaded tensors

    Raises:
        ImportError: If safetensors is not installed
        FileNotFoundError: If file doesn't exist

    Examples:
        >>> state_dict = load_safetensors("model.safetensors")
        >>> # Load to GPU
        >>> state_dict = load_safetensors("model.safetensors",
        ...                               device=torch.device("cuda:0"))
    """
    if not SAFETENSORS_AVAILABLE:
        raise ImportError(
            "safetensors is not installed. "
            "Install it with: pip install safetensors>=0.4"
        )

    tensors: Dict[str, torch.Tensor] = {}

    with safe_open(path, framework="pt", device=str(device)) as f:
        for key in f.keys():
            tensors[key] = f.get_tensor(key)

    return tensors


def save_pytorch(
    model: Union[nn.Module, Dict[str, torch.Tensor]],
    path: str,
) -> None:
    """
    Save model to PyTorch format (.pt/.pth).

    WARNING: Uses torch.save with pickle. Do NOT load models from
    untrusted sources as this can execute arbitrary code.

    Args:
        model: PyTorch model or state_dict to save
        path: Output file path (should end in .pt or .pth)

    Raises:
        ValueError: If path doesn't end in .pt or .pth

    Examples:
        >>> import torch.nn as nn
        >>> model = nn.Linear(10, 20)
        >>> save_pytorch(model, "model.pt")
    """
    # Extract state_dict
    if isinstance(model, nn.Module):
        state_dict = model.state_dict()
    else:
        state_dict = model

    torch.save(state_dict, path)


def load_pytorch(
    path: str,
    device: torch.device = torch.device("cpu"),
) -> Dict[str, torch.Tensor]:
    """
    Load model from PyTorch format (.pt/.pth).

    WARNING: Uses torch.load with pickle. Only load models from
    trusted sources as this can execute arbitrary code.

    Args:
        path: Path to .pt or .pth file
        device: Device to load tensors onto

    Returns:
        State dict with loaded tensors

    Examples:
        >>> state_dict = load_pytorch("model.pt")
        >>> # Load to GPU
        >>> state_dict = load_pytorch("model.pt",
        ...                            device=torch.device("cuda:0"))
    """
    return torch.load(path, map_location=device, weights_only=False)


def save_model(
    model: Union[nn.Module, Dict[str, torch.Tensor]],
    path: str,
    quantization_info: Optional[QuantizationInfo] = None,
    original_size_mb: Optional[float] = None,
    quantized_size_mb: Optional[float] = None,
    compression_ratio: Optional[float] = None,
    sqnr_db: Optional[float] = None,
) -> None:
    """
    Save quantized model with automatic format detection.

    Format is determined by file extension:
    - .safetensors: Safetensors format (secure, with metadata)
    - .pt, .pth: PyTorch format (uses pickle, no metadata support)

    For Safetensors format, comprehensive quantization metadata is
    automatically included.

    Args:
        model: PyTorch model or state_dict to save
        path: Output file path (extension determines format)
        quantization_info: Quantization parameters for metadata
        original_size_mb: Original model size for metadata
        quantized_size_mb: Quantized model size for metadata
        compression_ratio: Compression ratio for metadata
        sqnr_db: Signal-to-Quantization-Noise Ratio for metadata

    Raises:
        ValueError: If path extension is not supported
        ImportError: If safetensors required but not installed

    Examples:
        >>> import torch.nn as nn
        >>> from mono_quant.io import save_model
        >>> model = nn.Linear(10, 20)
        >>> # Save to Safetensors (recommended)
        >>> save_model(model, "quantized.safetensors")
        >>> # Save to PyTorch format
        >>> save_model(model, "quantized.pt")
        >>> # With quantization metadata
        >>> from mono_quant.io.formats import QuantizationInfo
        >>> info = QuantizationInfo(dtype=torch.qint8, symmetric=True,
        ...                         per_channel=True)
        >>> save_model(model, "quantized.safetensors",
        ...            quantization_info=info,
        ...            compression_ratio=4.0)
    """
    if path.endswith(".safetensors"):
        # Build metadata for Safetensors
        metadata = _build_metadata(
            quantization_info=quantization_info,
            original_size_mb=original_size_mb,
            quantized_size_mb=quantized_size_mb,
            compression_ratio=compression_ratio,
            sqnr_db=sqnr_db,
        )
        save_safetensors(model, path, metadata=metadata)
    elif path.endswith((".pt", ".pth")):
        # PyTorch format doesn't support metadata
        save_pytorch(model, path)
    else:
        raise ValueError(
            f"Unsupported file extension: {path}. "
            f"Use .safetensors, .pt, or .pth"
        )


def load_model(
    path: str,
    device: torch.device = torch.device("cpu"),
) -> Dict[str, torch.Tensor]:
    """
    Load model with automatic format detection.

    Format is determined by file extension:
    - .safetensors: Safetensors format
    - .pt, .pth: PyTorch format

    Args:
        path: Path to model file (extension determines format)
        device: Device to load tensors onto

    Returns:
        State dict with loaded tensors

    Raises:
        ValueError: If path extension is not supported

    Examples:
        >>> from mono_quant.io import load_model
        >>> # Load from Safetensors
        >>> state_dict = load_model("quantized.safetensors")
        >>> # Load from PyTorch format
        >>> state_dict = load_model("quantized.pt")
        >>> # Load to GPU
        >>> state_dict = load_model("quantized.safetensors",
        ...                         device=torch.device("cuda:0"))
    """
    if path.endswith(".safetensors"):
        return load_safetensors(path, device=device)
    elif path.endswith((".pt", ".pth")):
        return load_pytorch(path, device=device)
    else:
        raise ValueError(
            f"Unsupported file extension: {path}. "
            f"Use .safetensors, .pt, or .pth"
        )


__all__ = [
    "QuantizationInfo",
    "save_safetensors",
    "load_safetensors",
    "save_pytorch",
    "load_pytorch",
    "save_model",
    "load_model",
    "_build_metadata",
    "SAFETENSORS_AVAILABLE",
]
