# BF-007: Fix `quantize_weight_int4` fallback returning wrong data

## Status
TODO

## Audit Reference
C7 (Critical)

## Problem
In `core/quantizers.py`, the fallback branch of `quantize_weight_int4` (for
weights that fail the INT4 path) returns:

```python
return weight.int_repr(), scale, weight.q_per_channel_scales()
```

This has two errors:
1. **Third return value**: should be zero_points (`weight.q_per_channel_zero_points()`),
   not scales. The caller unpacks `(packed, scale, zero_point)`.
2. **Shape mismatch**: `int_repr()` returns an INT8 tensor (not packed INT4).
   The caller expects a packed INT4 tensor (where two INT4 values are stored per
   byte). Using an INT8 tensor here corrupts the downstream pack/unpack logic.

The fallback path is hit when group-wise quantization fails or when the tensor
doesn't meet INT4 preconditions. Both errors together mean the fallback silently
produces wrong quantized weights.

## Requirements
1. Third return value must be `weight.q_per_channel_zero_points()`.
2. Packed tensor must be INT4-packed (use the same packing as the normal path).
3. Scale must be compatible with the per-channel/group-wise downstream consumer.

## Decisions
- **Decision:** Fix the zero_points return value (typo/copy-paste error).
  For the packing issue: the fallback should either pack correctly using
  `_pack_int4_to_int8`, or the fallback should raise rather than silently
  return malformed data.
  Preferred: raise an explicit error. The fallback was likely a defensive stub
  that was never properly completed.
  Reason: Silent corruption is worse than a clear error. If the fallback is
  needed in the future it should be properly implemented with a test.

## Success Criteria
- [ ] Third return value is `zero_points`, not scales
- [ ] No silent return of mismatched tensor format
- [ ] If fallback raises, error message clearly explains why INT4 failed
- [ ] Existing INT4 success path is unchanged

## Dependencies
- None

## Implementation Guidance

In `src/mono_quant/core/quantizers.py`, find the fallback in `quantize_weight_int4`:

```python
# BEFORE (approximate)
return weight.int_repr(), scale, weight.q_per_channel_scales()

# OPTION A — Fix the typo and raise instead of returning corrupt data
raise RuntimeError(
    f"INT4 quantization fallback triggered for weight shape {weight.shape}. "
    "This path is not fully implemented. Use INT8 quantization instead."
)

# OPTION B — Fix both errors if fallback is needed
packed = _pack_int4_to_int8(weight.int_repr())   # pack INT8 → INT4
return packed, scale, weight.q_per_channel_zero_points()
```

Decision between A and B: prefer A (raise) unless there is a documented caller
that depends on the fallback returning valid data. Verify by searching for callers.

## Testing Requirements
- Unit: Trigger the fallback path; verify `RuntimeError` is raised with a
  clear message (if Option A chosen)
- Unit: Verify normal INT4 path still works (regression)
- Coverage target: both branches of `quantize_weight_int4`

## Open Questions
<!-- MUST be empty before implementation begins -->
