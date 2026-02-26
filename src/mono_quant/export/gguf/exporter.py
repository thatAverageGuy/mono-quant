"""GGUFExporter — exports any nn.Module to GGUF format for llama.cpp."""

import json
import warnings
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch.nn as nn

from mono_quant.export.base import BaseExporter


class GGUFExporter(BaseExporter):
    """Export any mono-quant (or plain FP32) model to a GGUF v3 file.

    Export flow:
        1. validate_compatibility  — must be nn.Module
        2. resolve_config          — merge config.json + model_params dict
        3. detect_architecture     — from config or user arg
        4. revert_to_standard_modules  — produce FP32 nn.Linear layers
        5. build_kv_metadata       — HF config → GGUF KV entries
        6. Quantize each nn.Linear weight → Q4_K_S bytes
        7. Map each PyTorch tensor name → GGUF name
        8. GGUFWriter.write(path / "model.gguf")
    """

    def export(
        self,
        model: nn.Module,
        path: Union[str, Path],
        quantization_type: str = "q4_k_s",
        architecture: Optional[str] = None,
        config_path: Optional[Union[str, Path]] = None,
        model_params: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """Run the full GGUF export pipeline.

        Args:
            model:             Any nn.Module (quantized or plain FP32).
            path:              Output directory. model.gguf written inside.
            quantization_type: "q4_k_s" (only supported value in v2.0).
            architecture:      GGUF architecture string. Auto-detected from config if None.
            config_path:       Path to HuggingFace config.json. Optional.
            model_params:      Dict of model hyperparameters (same keys as HF config.json).
                               Values here override config_path values on collision.

        Raises:
            TypeError: If model is not an nn.Module.
            ValueError: If quantization_type is not supported.
        """
        from mono_quant.core.quantizers import revert_to_standard_modules
        from mono_quant.export.gguf.arch_maps import (
            build_kv_entries,
            detect_architecture,
            map_tensor_name,
            reset_generic_counter,
        )
        from mono_quant.export.gguf.quant_types import (
            GGML_TYPE_F32,
            GGML_TYPE_Q4_K,
            quantize_to_q4_k_s,
        )
        from mono_quant.export.gguf.writer import GGUFWriter

        if quantization_type != "q4_k_s":
            raise ValueError(
                f"Unsupported quantization_type '{quantization_type}'. "
                "Only 'q4_k_s' is supported in v2.0."
            )

        self.validate_compatibility(model)

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Step 1: Resolve config (config.json + model_params override)
        config = self._resolve_config(config_path, model_params)

        # Step 2: Detect architecture
        if architecture is None:
            architecture = detect_architecture(config.get("model_type"))
            if architecture == "generic":
                warnings.warn(
                    "Could not detect GGUF architecture from config. "
                    "Using 'generic' sequential tensor naming. "
                    "Pass --architecture or provide config_path/model_params with 'model_type'.",
                    stacklevel=2,
                )

        # Step 3: Revert to FP32 standard modules
        fp32_model = revert_to_standard_modules(model, inplace=False)
        fp32_model.eval()

        # Step 4: Build KV metadata
        model_name = config.get("_name_or_path", "mono-quant-export")
        kv_entries = build_kv_entries(config, architecture, quantization_type, model_name)

        # Step 5: Quantize tensors + map names
        reset_generic_counter(architecture)
        writer = GGUFWriter()

        for key, val_type, val in kv_entries:
            _add_kv(writer, key, val_type, val)

        for layer_name, module in fp32_model.named_modules():
            if not isinstance(module, nn.Linear):
                continue

            gguf_name = map_tensor_name(f"{layer_name}.weight", architecture)
            weight    = module.weight.data.float()
            out_f, in_f = weight.shape

            if in_f % 256 == 0:
                data      = quantize_to_q4_k_s(weight)
                ggml_type = GGML_TYPE_Q4_K
            else:
                warnings.warn(
                    f"Layer '{layer_name}' has in_features={in_f} not divisible by 256. "
                    "Writing as FP32 (GGML_TYPE_F32). Q4_K requires in_features % 256 == 0.",
                    stacklevel=2,
                )
                import numpy as np
                data      = weight.numpy().tobytes()
                ggml_type = GGML_TYPE_F32

            writer.add_tensor(gguf_name, data, (out_f, in_f), ggml_type)

            if module.bias is not None:
                bias_name = map_tensor_name(f"{layer_name}.bias", architecture)
                bias_data = module.bias.data.float().numpy().tobytes()
                writer.add_tensor(bias_name, bias_data, tuple(module.bias.shape), GGML_TYPE_F32)

        writer.write(path / "model.gguf")

    def validate_compatibility(self, model: nn.Module) -> None:
        """Verify model is an nn.Module."""
        if not isinstance(model, nn.Module):
            raise TypeError(
                f"Expected nn.Module, got {type(model).__name__}. Pass a PyTorch model."
            )

    def build_metadata(self, model: nn.Module, **kwargs: Any) -> Dict[str, Any]:
        """Return a metadata dict describing the export."""
        from mono_quant import __version__
        return {
            "mono_quant_version": __version__,
            "export_format": "gguf",
            **{k: str(v) for k, v in kwargs.items()},
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _resolve_config(
        self,
        config_path: Optional[Union[str, Path]],
        model_params: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Merge config.json and model_params into one dict.

        model_params values override config.json values on collision.
        """
        config: Dict[str, Any] = {}

        if config_path is not None:
            config_path = Path(config_path)
            config.update(json.loads(config_path.read_text(encoding="utf-8")))

        if model_params is not None:
            config.update(model_params)

        if not config:
            warnings.warn(
                "No config_path or model_params provided. "
                "Architecture metadata will be minimal. "
                "llama.cpp may not load the model correctly.",
                stacklevel=3,
            )

        return config


def _add_kv(writer: Any, key: str, val_type: str, val: Any) -> None:
    """Dispatch a KV entry to the appropriate GGUFWriter.add_* method."""
    if val_type == "string":
        writer.add_string(key, str(val))
    elif val_type == "uint32":
        writer.add_uint32(key, int(val))
    elif val_type == "float32":
        writer.add_float32(key, float(val))
    elif val_type == "bool":
        writer.add_bool(key, bool(val))
    else:
        warnings.warn(f"Unknown KV value type '{val_type}' for key '{key}'. Skipping.")
