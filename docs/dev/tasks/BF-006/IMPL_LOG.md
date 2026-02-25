# Implementation Log: BF-006

## Summary
Fixed `DEFAULT_INT4_SKIP` being silently applied to every INT8 `static_quantize`
call. The condition `if group_size > 0` used the wrong discriminator — since
`group_size` defaulted to 128, the INT4 skip list was injected into all calls.

## What Was Done
- Changed `group_size` default in `static_quantize` from `128` to `0`
- With the new default, INT8 calls (no explicit group_size) get `group_size=0`,
  so the `if group_size > 0` condition is `False` and the INT4 skip list is
  not applied
- INT4 callers that explicitly pass `group_size=128` continue to get the skip list
- Also updated `all_layers_quantized_warning` in the `check_accuracy_warnings` call
  to be conditional on `group_size > 0` (M6 from CL-001)
- Added test verifying that INT8 static_quantize with default params does not
  skip embedding layers via the INT4 skip list

## How It Was Done
LEVEL 1 edit. The `group_size` parameter in `static_quantize` was only used as
an INT4 discriminator (it is never passed to `quantize_linear_module` or any
downstream function). Changing the default from 128 to 0 is safe: existing INT8
callers that relied on the default get the correct behavior (no INT4 skips),
and INT4 callers that explicitly set `group_size=128` are unchanged.

## Files Changed

| File | Change Type | What Changed |
|------|-------------|--------------|
| src/mono_quant/core/quantizers.py | Modified | `group_size: int = 128` → `group_size: int = 0` in `static_quantize` |
| src/mono_quant/core/quantizers.py | Modified | `all_layers_quantized_warning=(group_size > 0)` in `check_accuracy_warnings` call |
| tests/test_bugfixes.py | Modified | Added `test_static_quantize_int8_does_not_apply_int4_skip_list` |

## Testing Results
- Unit: `test_static_quantize_int8_does_not_apply_int4_skip_list` — PASS
- All 28 tests passing

## Final State
**Status**: DONE | **Date**: 2026-02-25
