# BF-001: v1.1 — Bug Fixes and Feature Additions

## Status
DONE

## Release
v1.1 (shipped 2026-02-04, before Phase 5)

## What Was Fixed / Added

### Bug Fixes
- QuantizedConv2d was fake — returned dequantized FP32 instead of INT8
- dynamic_quantize() crashed when exclusion parameters were passed
- QuantizedLinear/QuantizedConv2d didn't serialize quantization metadata
- Quantized models couldn't be loaded without mono-quant installed

### New Features
1. **QuantizedConv2d with true INT8 weights** — real memory savings (~4x reduction)
2. **Dynamic quantization exclusion params** — modules_to_not_convert, skip_layer_types, skip_layer_names, skip_param_threshold now work on dynamic_quantize() (matching static API)
3. **PyTorch-native deployment** — convert_to_pytorch_native() + auto-conversion in save_model(); quantized models loadable without mono-quant
4. **state_dict serialization** — custom _save_to_state_dict / _load_from_state_dict for round-trip fidelity
5. **QuantizedEmbedding** — INT8/FP16 embedding quantization (INT4 blocked for accuracy)
6. **revert_to_standard_modules()** — convert QuantizedLinear/Conv2d/Embedding back to nn.* for ONNX/pruning/inspection

## Files Changed
- `src/mono_quant/modules/linear.py` — QuantizedConv2d INT8, serialization, convert_to_pytorch_native
- `src/mono_quant/modules/embedding.py` — QuantizedEmbedding (new file)
- `src/mono_quant/core/quantizers.py` — exclusion params for dynamic, revert_to_standard_modules
- `src/mono_quant/io/formats.py` — auto-conversion on save
- `src/mono_quant/__init__.py` — version bump to 1.1, new exports
- `pyproject.toml` — version bump to 1.1

## Breaking Changes
None — all changes backward compatible.
