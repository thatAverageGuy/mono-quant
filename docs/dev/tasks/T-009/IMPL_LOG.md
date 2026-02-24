# Implementation Log: T-009

## Summary
INT4 quantization with group-wise scaling. QuantizedLinearInt4 packs two 4-bit
values per byte for storage efficiency.

## Status
DONE — 2026-02-03 | Milestone: v1.0

## Files Changed
| File | Change | What |
|------|--------|------|
| src/mono_quant/modules/linear.py | Modified | QuantizedLinearInt4 class |
| src/mono_quant/core/quantizers.py | Modified | INT4 quantization path |

## Key Decisions
- Packed INT8 storage (2 INT4 per byte) chosen over float16 packing for simplicity
- group_size=128 matches GPTQ standard — chosen for future compatibility
