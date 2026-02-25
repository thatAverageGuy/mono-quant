"""ONNX model validation utilities."""

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
