# Implementation Log: BF-007

## Summary
Fixed the `quantize_weight_int4` fallback branch that silently returned corrupt
packed/zero-point data when the layer dimension was smaller than `group_size`.
Replaced with a clear `RuntimeError` per the DETAIL.md decision.

## What Was Done
- Replaced the fallback `return` in the `dim_size < group_size` branch with a
  `raise RuntimeError(...)` that explains the constraint clearly
- Removed the misleading `logger.warning` and the entire corrupt fallback logic
  (the old code returned `q_weight.int_repr()` as `zero_point` instead of
  `q_weight.q_per_channel_zero_points()`, and used INT8 data where packed INT4
  was expected by callers)
- Added test: small weight (dim 16 < group_size 128) raises RuntimeError
- Added test: normal path (dim 256 >= group_size 128) still works correctly

## How It Was Done
Single LEVEL 1 edit. Decision between raising (Option A) vs fixing (Option B):
chose Option A per DETAIL.md — silent corruption is worse than a clear error.
The fallback was never usable (two separate bugs in its return values), and no
caller requires it given the existing skip list that avoids small layers for INT4.

Before:
```python
if dim_size < group_size:
    logger.warning(...)
    q_weight = quantize_weight_int8(...)
    scale = q_weight.q_per_channel_scales()
    zero_point = q_weight.int_repr()  # BUG: should be q_per_channel_zero_points()
    return zero_point.to(torch.int8), scale, torch.zeros_like(scale).to(torch.int32)
```

After:
```python
if dim_size < group_size:
    raise RuntimeError(
        f"INT4 quantization requires group_size <= layer dimension, but ..."
    )
```

## Files Changed

| File | Change Type | What Changed |
|------|-------------|--------------|
| src/mono_quant/core/quantizers.py | Modified | Fallback branch replaced with RuntimeError |
| tests/test_bugfixes.py | Modified | Two new tests for small-layer raise and normal path |

## Testing Results
- Unit: `test_quantize_weight_int4_small_layer_raises_runtime_error` — PASS
- Unit: `test_quantize_weight_int4_normal_path_unchanged` — PASS
- All 28 tests passing

## Final State
**Status**: DONE | **Date**: 2026-02-25
