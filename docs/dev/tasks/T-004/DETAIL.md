# T-004: dynamic_quantize() Function, Public API Exports

## Status
DONE

## Phase
01-04 — Phase 1: Core Quantization Foundation

## Requirements
- QCORE-04: Dynamic quantization without calibration data

## Decisions
- dynamic_quantize() in core/quantizers.py
- Exported from src/mono_quant/__init__.py public API
- Handles Linear and Conv2d by default

## Success Criteria
- [x] dynamic_quantize(model) works end-to-end
- [x] Returns (quantized_model, QuantizationInfo) tuple
- [x] Public API exports correct

## Files
- `src/mono_quant/core/quantizers.py` — dynamic_quantize()
- `src/mono_quant/core/__init__.py` — exports
- `src/mono_quant/__init__.py` — public API
