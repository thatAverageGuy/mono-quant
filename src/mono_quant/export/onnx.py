"""ONNXExporter — orchestrates the full ONNX export flow."""

import json
import tempfile
import warnings
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch
import torch.nn as nn

from mono_quant.export.base import BaseExporter
from mono_quant.export.common.validators import ValidationLevel, validate_onnx_model


class ONNXExporter(BaseExporter):
    """Exports a quantized mono-quant model to ONNX with QDQ nodes.

    Export flow:
        1. Collect quantization params from quantized modules.
        2. Revert model to standard FP32 nn.Modules.
        3. torch.onnx.export → raw ONNX graph (FP32 weights).
        4. Insert QuantizeLinear/DequantizeLinear node pairs.
        5. Attach quantization metadata to model doc_string.
        6. Save final .onnx file.
        7. Optional: validate via onnx.checker / onnxruntime.
    """

    def export(
        self,
        model: nn.Module,
        path: Union[str, Path],
        opset: int = 14,
        dummy_input: Optional[torch.Tensor] = None,
        validate: Union[str, ValidationLevel] = "none",
    ) -> None:
        """Run the full export pipeline.

        Args:
            model: Quantized nn.Module.
            path: Destination .onnx file path.
            opset: ONNX opset version. Default 14.
            dummy_input: Optional dummy input tensor. Auto-inferred if None.
            validate: Validation level — "none", "load", or "full".

        Raises:
            ImportError: If onnx is not installed.
            RuntimeError: If dummy input cannot be inferred.
            TypeError: If model is not an nn.Module.
        """
        try:
            import onnx
        except ImportError:
            raise ImportError(
                "ONNX export requires additional dependencies. "
                "Install them with: pip install mono-quant[onnx]"
            )

        from mono_quant.core.quantizers import revert_to_standard_modules
        from mono_quant.export.common.qdq_inserter import (
            collect_quantization_params,
            insert_qdq_nodes,
        )

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Step 1: Validate model type
        self.validate_compatibility(model)

        # Step 2: Collect quantization params (before reverting)
        qparams = collect_quantization_params(model, opset=opset)

        # Warn about INT4 layers that can't be QDQ-encoded at this opset
        has_int4 = any(p.is_int4 for p in qparams.values())
        if has_int4 and opset < 21:
            warnings.warn(
                f"INT4 quantized layers detected but opset={opset} < 21. "
                "INT4 QDQ nodes require opset>=21. "
                "INT4 layers will be exported with FP32 dequantized weights.",
                UserWarning,
                stacklevel=3,
            )

        # Only insert QDQ nodes for INT8 layers
        qdq_qparams = {k: v for k, v in qparams.items() if not v.is_int4}

        # Step 3: Revert to standard FP32 modules (non-inplace copy)
        fp32_model = revert_to_standard_modules(model, inplace=False)
        fp32_model.eval()

        # Step 4: Infer dummy input if not provided
        if dummy_input is None:
            dummy_input = self._infer_dummy_input(fp32_model)

        # Step 5: Export to temporary ONNX file
        tmp_file = tempfile.NamedTemporaryFile(suffix=".onnx", delete=False)
        tmp_path = Path(tmp_file.name)
        tmp_file.close()

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                torch.onnx.export(
                    fp32_model,
                    (dummy_input,),
                    str(tmp_path),
                    dynamo=False,
                    opset_version=opset,
                    input_names=["input"],
                    output_names=["output"],
                    dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
                )

            # Step 6: Load proto, insert QDQ nodes
            model_proto = onnx.load(str(tmp_path))
            if qdq_qparams:
                model_proto = insert_qdq_nodes(model_proto, qdq_qparams)

            # Step 7: Attach metadata
            metadata = self.build_metadata(model, opset=opset, has_int4=has_int4)
            model_proto.doc_string = json.dumps(metadata)

            # Step 8: Save
            onnx.save(model_proto, str(path))

        finally:
            tmp_path.unlink(missing_ok=True)

        # Step 9: Optional validation
        validate_onnx_model(path, level=ValidationLevel(validate))

    def validate_compatibility(self, model: nn.Module) -> None:
        """Verify model is an nn.Module."""
        if not isinstance(model, nn.Module):
            raise TypeError(
                f"Expected nn.Module, got {type(model).__name__}. "
                "Pass a quantized PyTorch model."
            )

    def build_metadata(self, model: nn.Module, **kwargs: Any) -> Dict[str, Any]:
        """Build JSON metadata dict stored in the ONNX model's doc_string."""
        from mono_quant import __version__

        return {
            "mono_quant_version": __version__,
            "export_format": "onnx_qdq",
            "opset": kwargs.get("opset", 14),
            "has_int4_layers": kwargs.get("has_int4", False),
            "library": "mono-quant",
        }

    def _infer_dummy_input(self, model: nn.Module) -> torch.Tensor:
        """Generate a dummy input tensor from the first Linear or Conv2d layer."""
        for module in model.modules():
            if isinstance(module, nn.Linear):
                return torch.zeros(1, module.in_features)
            if isinstance(module, nn.Conv2d):
                return torch.zeros(1, module.in_channels, 32, 32)
        raise RuntimeError(
            "Cannot infer dummy input: no Linear or Conv2d layer found. "
            "Please provide dummy_input explicitly."
        )

    def _detect_int4_quantization(self, model: nn.Module) -> bool:
        """Return True if model contains any QuantizedLinearInt4 modules."""
        from mono_quant.modules.linear import QuantizedLinearInt4

        return any(isinstance(m, QuantizedLinearInt4) for m in model.modules())
