"""GPTQExporter — exports a mono-quant model to AutoGPTQ V1 format."""

import json
from pathlib import Path
from typing import Any, Dict, Union

import torch.nn as nn

from mono_quant.export.base import BaseExporter


class GPTQExporter(BaseExporter):
    """Exports any mono-quant (or plain FP32) model to GPTQ INT4 safetensors.

    Export flow:
        1. validate_compatibility — confirm nn.Module
        2. revert_to_standard_modules — produce FP32 nn.Linear layers
        3. For each nn.Linear: quantize_to_gptq_int4
        4. Write model.safetensors  (qweight, qzeros, scales, g_idx, bias)
        5. Write quantize_config.json
    """

    def export(
        self,
        model: nn.Module,
        path: Union[str, Path],
        group_size: int = 128,
        sym: bool = False,
        **kwargs: Any,
    ) -> None:
        """Run the full GPTQ export pipeline.

        Args:
            model: Any nn.Module (quantized or plain FP32).
            path: Output directory. Created if it does not exist.
            group_size: Columns per quantization group. Must divide in_features.
            sym: Symmetric quantization if True, asymmetric if False.

        Raises:
            TypeError: If model is not an nn.Module.
            ValueError: If a layer's dimensions are incompatible with group_size.
        """
        from safetensors.torch import save_file

        from mono_quant.core.quantizers import revert_to_standard_modules
        from mono_quant.export.common.gptq_packing import quantize_to_gptq_int4

        self.validate_compatibility(model)

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        fp32_model = revert_to_standard_modules(model, inplace=False)
        fp32_model.eval()

        checkpoint = {}
        for name, module in fp32_model.named_modules():
            if not isinstance(module, nn.Linear):
                continue
            qw, qz, sc, gi = quantize_to_gptq_int4(
                module.weight.data, group_size=group_size, sym=sym
            )
            checkpoint[f"{name}.qweight"] = qw.contiguous()
            checkpoint[f"{name}.qzeros"] = qz.contiguous()
            checkpoint[f"{name}.scales"] = sc.contiguous()
            checkpoint[f"{name}.g_idx"] = gi.contiguous()
            if module.bias is not None:
                checkpoint[f"{name}.bias"] = module.bias.data.clone().contiguous()

        save_file(checkpoint, path / "model.safetensors")
        self._write_quantize_config(path, group_size, sym)

    def validate_compatibility(self, model: nn.Module) -> None:
        """Verify model is an nn.Module."""
        if not isinstance(model, nn.Module):
            raise TypeError(
                f"Expected nn.Module, got {type(model).__name__}. "
                "Pass a PyTorch model."
            )

    def build_metadata(self, model: nn.Module, **kwargs: Any) -> Dict[str, Any]:
        """Build metadata dict for the exported model."""
        from mono_quant import __version__

        return {
            "mono_quant_version": __version__,
            "export_format": "gptq",
            **{k: str(v) for k, v in kwargs.items()},
        }

    def _write_quantize_config(self, path: Path, group_size: int, sym: bool) -> None:
        config = {
            "bits": 4,
            "group_size": group_size,
            "desc_act": False,
            "sym": sym,
            "quant_method": "gptq",
        }
        (path / "quantize_config.json").write_text(json.dumps(config, indent=2))
