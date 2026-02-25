# Implementation Log: BF-005

## Summary
Fixed mutable default argument `skip_set: set = set()` in
`_quantize_sequential_module`. Python evaluates default arguments once at
function definition time, so all calls sharing the default would share the
same set object — mutations in one call could affect subsequent calls.

## What Was Done
- Changed `skip_set: set = set()` to `skip_set: Optional[set] = None`
- Added `if skip_set is None: skip_set = set()` as the first statement in
  the function body (sentinel pattern)
- Updated docstring to reflect the new parameter semantics
- Added test verifying two independent calls both work correctly without
  sharing state

## How It Was Done
Single LEVEL 1 edit. The sentinel pattern (None default + internal creation)
is the standard Python idiom for mutable default arguments. The function body
already assigned to `skip_set` only via membership checks — never via mutation
— so the bug's practical impact was limited to function calls that passed the
literal default, but the pattern was still wrong and could cause issues if the
function were extended.

Before:
```python
def _quantize_sequential_module(
    ...
    skip_set: set = set(),   # shared across all default-argument calls
    ...
```

After:
```python
def _quantize_sequential_module(
    ...
    skip_set: Optional[set] = None,   # safe: None per-call
    ...
):
    if skip_set is None:
        skip_set = set()
```

## Files Changed

| File | Change Type | What Changed |
|------|-------------|--------------|
| src/mono_quant/core/quantizers.py | Modified | `skip_set: set = set()` → `skip_set: Optional[set] = None` + sentinel |
| tests/test_bugfixes.py | Modified | Added `test_sequential_module_skip_set_not_shared_across_calls` |

## Testing Results
- Unit: `test_sequential_module_skip_set_not_shared_across_calls` — PASS
- All 28 tests passing

## Final State
**Status**: DONE | **Date**: 2026-02-25
