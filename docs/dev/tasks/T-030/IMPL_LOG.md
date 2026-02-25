# Implementation Log: T-030

## Summary
Corrected the falsely-DONE ONNX export phantom. T-014–T-017 were marked DONE
in original planning but `src/mono_quant/export/` never existed. Added a
`NotImplementedError` stub for `export_to_onnx()` and corrected task statuses.

## What Was Done
- Added `export_to_onnx()` stub to `src/mono_quant/__init__.py`
- Added `export_to_onnx` to `__all__`
- Updated `docs/dev/tasks/TASKS.md` — T-014–T-017 moved back to TODO with ⚠
- Updated T-014–T-017 IMPL_LOG.md files with audit correction note
- Added test verifying `NotImplementedError` (not `AttributeError`) is raised

## How It Was Done
Minimal LEVEL 1 intervention. The stub follows the existing `NotImplementedError`
pattern used in other "planned but not implemented" public APIs.

## Files Changed

| File | Change Type | What Changed |
|------|-------------|--------------|
| src/mono_quant/__init__.py | Modified | Added `export_to_onnx` stub + `__all__` entry |
| docs/dev/tasks/TASKS.md | Modified | T-014–T-017 status corrected to TODO ⚠ |
| tests/test_bugfixes.py | Created | `test_export_to_onnx_raises_not_implemented` |

## Why These Choices Were Made
- Stub preferred over removing the API reference: the function is documented in
  the public docstring. Callers get a clear `NotImplementedError` with a helpful
  message rather than `AttributeError: module has no attribute 'export_to_onnx'`.
- T-034+ reserved for actual ONNX implementation to preserve existing IMPL_LOG
  history for T-014–T-017.

## Testing Results
- Unit: 1 test passing — `export_to_onnx` raises `NotImplementedError`
- All 15 tests passing

## Final State
**Status**: DONE | **Date**: 2026-02-25
