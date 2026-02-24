# Implementation Log: BF-001

## Summary
v1.1 release: fixed fake Conv2d quantization, added QuantizedEmbedding, enabled
PyTorch-native deployment (no mono-quant needed to load), added revert_to_standard_modules()
as bridge to ONNX/ecosystem exports.

## Status
DONE — 2026-02-04 | Milestone: v1.1

## Files Changed
| File | Change | What |
|------|--------|------|
| src/mono_quant/modules/linear.py | Modified | QuantizedConv2d INT8, serialization methods, convert_to_pytorch_native |
| src/mono_quant/modules/embedding.py | Created | QuantizedEmbedding |
| src/mono_quant/core/quantizers.py | Modified | Dynamic exclusion params, revert_to_standard_modules() |
| src/mono_quant/io/formats.py | Modified | Auto-convert on save_model() |
| src/mono_quant/__init__.py | Modified | Version 1.1, new exports |
| pyproject.toml | Modified | Version 1.1 |

## Testing Results
- Unit: All existing tests pass
- Integration: Save/load round-trip verified without mono-quant
- Coverage: ~1,200 lines added across 8 files/7 commits

## Impact
- revert_to_standard_modules() became the critical bridge for Phase 5 ONNX export
- PyTorch-native deployment unlocks zero-dependency model distribution
