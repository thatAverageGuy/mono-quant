"""Export validation utilities (ONNX and GPTQ)."""

import json
from enum import Enum
from pathlib import Path
from typing import Union


class ValidationLevel(str, Enum):
    NONE = "none"
    LOAD = "load"
    FULL = "full"


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

        session = ort.InferenceSession(str(path))
        inputs = session.get_inputs()
        input_shape = inputs[0].shape
        # Replace dynamic (symbolic) dims with 1
        concrete_shape = [d if isinstance(d, int) and d > 0 else 1 for d in input_shape]
        dummy = np.zeros(concrete_shape, dtype=np.float32)
        session.run(None, {inputs[0].name: dummy})


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
