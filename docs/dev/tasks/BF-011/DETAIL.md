# BF-011: Fix `_test_load_run` mutating the model under validation

## Status
TODO

## Audit Reference
H6 (High)

## Problem
In `io/validation.py`, `_test_load_run` is called to verify that a quantized
model can be saved and reloaded. It does:

```python
quantized.load_state_dict(loaded)
```

This call **overwrites the state of the model passed in** — `quantized` is the
live model being validated, not a copy. After this call, the model's weights
have been replaced with the reloaded version. If any discrepancy exists between
the saved/loaded version and the original (e.g. buffer renaming, precision
drift), the validated model is now in a corrupted state.

Additionally, if validation is run before `result.save()` finalises, the caller
receives back a model that has been modified by the validation step.

## Requirements
1. `_test_load_run` must not modify the model passed to it.
2. Reload verification must operate on a deep copy or a temporary fresh
   instance — not the live model.
3. Validation result must be based on comparing the original vs reloaded weights,
   not overwriting in place.

## Decisions
- **Decision:** Use `copy.deepcopy(quantized)` to clone the model before calling
  `load_state_dict`. Operate the reload on the clone only.
  Reason: Deepcopy is safe here — model is a PyTorch module with no external
  state. This is the standard PyTorch pattern for non-destructive validation.

## Success Criteria
- [ ] After calling `validate_quantization(model, ...)`, the model's weights
      are identical to before the call
- [ ] Reload test still verifies that save → load round-trip works
- [ ] All existing tests pass

## Dependencies
- None

## Implementation Guidance

In `src/mono_quant/io/validation.py`, in `_test_load_run`:

```python
import copy

# BEFORE
def _test_load_run(quantized, path):
    loaded = load_model(path)
    quantized.load_state_dict(loaded)   # mutates live model

# AFTER
def _test_load_run(quantized, path):
    loaded = load_model(path)
    model_copy = copy.deepcopy(quantized)
    model_copy.load_state_dict(loaded)  # operates on clone
    # compare model_copy vs quantized for drift if needed
    return model_copy
```

Adjust to fit the actual function signature and what the caller expects as
a return value.

## Testing Requirements
- Unit: Call `validate_quantization(model, ...)`; compare `model.state_dict()`
  before and after — must be byte-identical.
- Unit: Verify the reload still works (the state_dict loaded into the clone
  must be valid).
- Coverage target: `_test_load_run` reload path

## Open Questions
<!-- MUST be empty before implementation begins -->
