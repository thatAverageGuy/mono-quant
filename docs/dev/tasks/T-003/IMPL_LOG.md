# Implementation Log: T-003

## Summary
QuantizedLinear module that stores real INT8 weights. Quantization
transformation logic in core/quantizers.py.

## Status
DONE — 2026-02-03 | Milestone: v1.0

## Files Changed
| File | Change | What |
|------|--------|------|
| src/mono_quant/modules/linear.py | Created | QuantizedLinear with INT8 storage |
| src/mono_quant/core/quantizers.py | Created | quantize_tensor(), module replacement |

## Key Issues
- Circular import between core.quantizers and modules.linear
  Resolution: local imports inside quantize functions in core/quantizers.py
