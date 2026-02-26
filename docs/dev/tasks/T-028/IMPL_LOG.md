# Implementation Log: T-028

## Summary

Added `ExportWarning` dataclass and `validate_export_pre` / `validate_export_post` functions
to `export/common/validators.py`. Wired `validate_export_pre` into the orchestrator at module
level so tests can patch it.

## What Was Done

### validators.py additions

- `ExportWarning` dataclass: `level` (str), `message` (str), `check` (str).
- `validate_export_pre(model, info, format)` → `List[ExportWarning]`:
  - Check 1: no quantized parameters → warning `"no_quantized_layers"`
  - Check 2: INT8 model + GPTQ format → error `"int8_gptq_mismatch"`
  - Check 3: ONNX + bits==4 → warning `"onnx_int4_opset"`
- `validate_export_post(path, format)`:
  - Dispatches to `validate_onnx_model`, `validate_gptq_checkpoint_structure`,
    `validate_gguf_checkpoint`.

### orchestrator.py

- Imports `validate_export_pre`, `validate_export_post` at module level (not inside function).
- Calls `validate_export_pre` before dispatch; prints stderr lines for each warning.
- `validate_post` kwarg (default False) triggers `validate_export_post` after export.

## Files Changed

| File | Change Type | What Changed |
|------|-------------|--------------|
| `src/mono_quant/export/common/validators.py` | Modified | ExportWarning, validate_export_pre/post |
| `src/mono_quant/export/orchestrator.py` | Modified | Module-level import, pre/post hooks |
| `tests/test_export_validation.py` | Created | 11 tests |

## Testing Results

- Unit: 11/11 passing
- Coverage: all 3 pre-check paths covered; post dispatch for all 3 formats covered

## Final State

**Status**: DONE | **Date**: 2026-02-26
