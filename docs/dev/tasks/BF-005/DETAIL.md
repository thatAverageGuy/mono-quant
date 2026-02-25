# BF-005: Fix mutable default argument in `_quantize_sequential_module`

## Status
TODO

## Audit Reference
C9 (Medium — silent data corruption potential)

## Problem
In `core/quantizers.py`, `_quantize_sequential_module` uses a mutable set as a
default argument:

```python
def _quantize_sequential_module(model, ..., skip_set: set = set()):
```

Python evaluates default arguments once at function definition time. Every call
that doesn't pass `skip_set` shares the same `set` object. If any call modifies
`skip_set` (e.g. by adding to it), subsequent calls see the modified state.
This can cause layers to be silently skipped that should not be.

## Requirements
1. Replace mutable default with `None` sentinel.
2. Initialise `skip_set = set()` inside the function body when `None` is passed.
3. No change to external call sites — default behaviour must be identical.

## Decisions
- **Decision:** Use `None` sentinel pattern (standard Python idiom).
  Reason: Correct, zero-risk, no interface change.

## Success Criteria
- [ ] `_quantize_sequential_module` signature uses `skip_set=None`
- [ ] First line of function body: `if skip_set is None: skip_set = set()`
- [ ] Existing tests still pass

## Dependencies
- None

## Implementation Guidance

In `src/mono_quant/core/quantizers.py`, find `_quantize_sequential_module`:

```python
# BEFORE
def _quantize_sequential_module(model, config, skip_set: set = set()):
    ...

# AFTER
def _quantize_sequential_module(model, config, skip_set=None):
    if skip_set is None:
        skip_set = set()
    ...
```

Check for any other functions in the same file with the same pattern.

## Testing Requirements
- Unit: Call `_quantize_sequential_module` twice in sequence; verify second call
  is not affected by any modifications made to `skip_set` in the first call.
- Coverage target: the initialisation branch

## Open Questions
<!-- MUST be empty before implementation begins -->
