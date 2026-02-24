# T-006: Layer Selection API, static_quantize() with Calibration

## Status
DONE

## Phase
02-02 — Phase 2: Static Quantization & I/O

## Requirements
- QCORE-05: Static quantization with calibration data
- QCORE-06: User selects which layer types to quantize

## Decisions
- modules_to_not_convert, skip_layer_types, skip_layer_names parameters
- static_quantize(model, calibration_data) as primary entry point
- Returns (quantized_model, QuantizationInfo)

## Success Criteria
- [x] static_quantize() works end-to-end with calibration data
- [x] Layer type exclusion works
- [x] Layer name exclusion works

## Files
- `src/mono_quant/core/quantizers.py` — static_quantize()
