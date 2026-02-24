# Implementation Log: T-002

## Summary
Core quantization math: symmetric/asymmetric schemes and per-tensor/per-channel scale
and zero-point calculation. Foundation for all quantization modes.

## Status
DONE — 2026-02-03 | Milestone: v1.0

## Files Changed
| File | Change | What |
|------|--------|------|
| src/mono_quant/core/schemes.py | Created | SymmetricScheme, AsymmetricScheme |
| src/mono_quant/core/mappers.py | Created | per-tensor, per-channel mappers |

## Key Decisions
- Symmetric always sets zero_point=0, enabling faster inference
- Per-channel mapper operates over dim=0 for weight tensors
