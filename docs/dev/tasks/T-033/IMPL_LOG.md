# Implementation Log: T-033

## Summary
Fixed the file path input branch in `api/quantize.py` that always crashed.
Per DETAIL.md Option A: removed the broken file path branch entirely and
replaced with a clear `TypeError` directing users to load the model themselves.

## What Was Done
- Replaced the `if isinstance(model, (str, Path)):` branch (which set `model = state_dict`
  and then crashed in `_prepare_model`) with an unconditional `TypeError` raise
  that explains what to do instead
- Removed the now-unused `from mono_quant.io import load_model` local import
- Updated the `InputError` suggestion text for the `_prepare_model` failure path
- Added two tests: string path raises TypeError; Path object raises TypeError

## How It Was Done
Single LEVEL 1 edit. The file path feature was never functional:
`_prepare_model(state_dict)` requires an architecture argument to reconstruct
a model from a state dict. No caller in the codebase used this path. The
TypeError with a helpful message (example of how to do it correctly) is
strictly better than an opaque crash deep in `_prepare_model`.

Before:
```python
if isinstance(model, (str, Path)):
    state_dict = load_model(model_path)
    model = state_dict  # then crashes in _prepare_model
```

After:
```python
if isinstance(model, (str, Path)):
    raise TypeError(
        f"quantize() expects an nn.Module, got {type(model).__name__!r}. "
        "To quantize from a file, load the model first: ..."
    )
```

## Files Changed

| File | Change Type | What Changed |
|------|-------------|--------------|
| src/mono_quant/api/quantize.py | Modified | File path branch replaced with TypeError; unused import removed |
| tests/test_bugfixes.py | Modified | Two new TypeError tests for str and Path inputs |

## Testing Results
- Unit: `test_quantize_file_path_raises_type_error` — PASS
- Unit: `test_quantize_path_object_raises_type_error` — PASS
- All 28 tests passing

## Final State
**Status**: DONE | **Date**: 2026-02-25
