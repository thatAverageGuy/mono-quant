# Implementation Log: T-015

## Summary
QDQ node insertion utilities. The core mechanism that converts mono-quant's
internal representation to ONNX-compatible quantized graph format.

## Status
DONE — 2026-02-04 | Milestone: v2.0/Phase 5

## Files Changed
| File | Change | What |
|------|--------|------|
| src/mono_quant/export/common/qdq_inserter.py | Created | insert_qdq_for_linear, insert_qdq_for_conv2d (558 lines) |

## Key Decisions
- Zero-point dtype: symmetric/all-zero → INT8; asymmetric with min≥0 → UINT8
- Per-channel DequantizeLinear with axis=0 for weight quantization
- Scale initializer stored as FLOAT (not FLOAT16) for runtime compatibility
