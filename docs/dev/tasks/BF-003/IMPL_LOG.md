# Implementation Log: BF-003

## Summary
Removed the spurious `- 8` shift from the INT4 symmetric quantization formula
in `core/quantizers.py`. The shift was inverting and offsetting all weights in
symmetric INT4 quantization, producing silently wrong inference results.

## What Was Done
- Removed `- 8` from the symmetric branch of `quantize_weight_int4`
- Added three tests: range check, cosine similarity round-trip, max error bound

## How It Was Done
Single LEVEL 1 edit. The fix is one token removed from one line. The surrounding
pack/unpack infrastructure (`_pack_int4_to_int8`, `_unpack_int8_to_int4`) already
handles signed [-8, 7] values correctly via two's complement bitwise operations —
the `-8` was never needed.

Before:
```python
int4_group = torch.clamp(
    (group_weights / group_scale).round().to(torch.int32) - 8,
    qmin, qmax   # [-8, 7]
)
```
After:
```python
int4_group = torch.clamp(
    (group_weights / group_scale).round().to(torch.int32),
    qmin, qmax   # [-8, 7]
)
```

## Files Changed

| File | Change Type | What Changed |
|------|-------------|--------------|
| src/mono_quant/core/quantizers.py | Modified | Removed `- 8` from symmetric INT4 formula |
| tests/test_bugfixes.py | Modified | Three INT4 round-trip tests added |

## Testing Results
- Unit: `test_int4_symmetric_quantized_values_in_range` — values in [-8, 7]: PASS
- Unit: `test_int4_symmetric_round_trip_cosine_similarity` — cosine sim > 0.9: PASS
- Unit: `test_int4_symmetric_round_trip_max_error` — max error < 20% of max_abs: PASS
- Asymmetric path: unchanged, no regression
- All 15 tests passing

## Final State
**Status**: DONE | **Date**: 2026-02-25
