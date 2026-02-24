# T-007: Serialization — PyTorch and Safetensors Formats

## Status
DONE

## Phase
02-03 — Phase 2: Static Quantization & I/O

## Requirements
- IO-01: Save to PyTorch format (.pt/.pth)
- IO-02: Save to Safetensors format
- IO-03: Save quantization config with model
- IO-04: Load quantized model from disk
- IO-05: Dequantize model back to FP32

## Decisions
- Auto-detect format from file extension
- Safetensors is default for new saves
- Quantization metadata saved as part of state_dict

## Files
- `src/mono_quant/io/formats.py` — save_model(), load_model()
- `src/mono_quant/io/validation.py` — format validation
