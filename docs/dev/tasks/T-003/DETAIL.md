# T-003: Quantization Transformations, QuantizedLinear Module

## Status
DONE

## Phase
01-03 — Phase 1: Core Quantization Foundation

## Requirements
- QCORE-01: INT8 quantization with per-channel scaling
- QCORE-03: FP16 quantization

## Decisions
- QuantizedLinear subclasses nn.Linear, stores INT8 weights
- from_float() classmethod copies weights and biases from original
- FP16 handled as in-place weight cast, not a custom module

## Success Criteria
- [x] QuantizedLinear stores actual INT8 weights
- [x] forward() dequantizes to FP32 for computation
- [x] from_float() copies all module state

## Files
- `src/mono_quant/modules/linear.py` — QuantizedLinear
- `src/mono_quant/core/quantizers.py` — quantization transformation logic
