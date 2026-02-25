# Implementation Log: BF-011

## Summary
Fixed `_test_load_run` mutating the model under test. The function was loading
a saved state_dict back into `quantized` (the live model being validated),
replacing its weights with the serialized values. Any subsequent use of the
model after validation would see potentially different weights.

## What Was Done
- Added `import copy` to `io/validation.py` imports
- Replaced `quantized.load_state_dict(loaded)` with:
  ```python
  test_model = copy.deepcopy(quantized)
  test_model.load_state_dict(loaded)
  ```
- Updated the forward pass in step 4 to use `test_model(test_input)` instead
  of `quantized(test_input)`
- Added test verifying that the original model's state_dict is unchanged after
  `_test_load_run` returns

## How It Was Done
LEVEL 1 surgical edit. `copy.deepcopy` creates a fully independent clone of the
quantized model, including all parameters and buffers. The clone receives the
loaded state_dict; the original remains untouched. The test shapes (determined
by iterating `quantized.modules()`) still use the unmodified original, which
is correct since we're just determining input dimensions.

## Files Changed

| File | Change Type | What Changed |
|------|-------------|--------------|
| src/mono_quant/io/validation.py | Modified | `import copy` added; deepcopy before load_state_dict; forward pass on test_model |
| tests/test_bugfixes.py | Modified | Added `test_load_run_does_not_mutate_original_model` |

## Testing Results
- Unit: `test_load_run_does_not_mutate_original_model` — PASS
- All 28 tests passing

## Final State
**Status**: DONE | **Date**: 2026-02-25
