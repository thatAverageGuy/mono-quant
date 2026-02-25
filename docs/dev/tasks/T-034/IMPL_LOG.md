# Implementation Log: T-034

## Summary
Created the ONNX export infrastructure: public entry point, base class,
validation utilities, and pyproject.toml optional dependencies. Replaced
the NotImplementedError stub in `__init__.py` with a lazy import.

## What Was Done
- Created `src/mono_quant/export/` package with `__init__.py`, `base.py`,
  `common/__init__.py`, and `common/validators.py`.
- Added `[project.optional-dependencies] onnx` to `pyproject.toml`.
- Replaced stub in `src/mono_quant/__init__.py` with a lazy delegation to
  `mono_quant.export.export_to_onnx`.
- Removed stale `.pyc` files from `export/__pycache__/` and `export/common/__pycache__/`.

## How It Was Done
- `export/__init__.py`: defines `export_to_onnx(model, path, opset, dummy_input, validate)`.
  Lazily imports `onnx` to validate the dependency is installed, then delegates to
  `onnx_impl.export_to_onnx_impl`. Raises `ImportError` with pip install hint if missing.
- `export/base.py`: `BaseExporter` ABC with three abstract methods: `export()`,
  `validate_compatibility()`, `build_metadata()`.
- `export/common/validators.py`: `ValidationLevel` enum (none/load/full) and
  `validate_onnx_model(path, level)` which runs `onnx.checker.check_model` for
  "load" and adds an onnxruntime inference pass for "full".
- `__init__.py` stub: delegated to `mono_quant.export` lazily so ImportError
  propagates cleanly without requiring onnx at package import time.

## Files Changed

| File | Change Type | What Changed |
|------|-------------|--------------|
| `src/mono_quant/export/__init__.py` | Created | Public `export_to_onnx()` with lazy onnx check |
| `src/mono_quant/export/base.py` | Created | `BaseExporter` ABC |
| `src/mono_quant/export/common/__init__.py` | Created | Package init (empty) |
| `src/mono_quant/export/common/validators.py` | Created | `ValidationLevel`, `validate_onnx_model()` |
| `src/mono_quant/__init__.py` | Modified | Replaced NotImplementedError stub with lazy import |
| `pyproject.toml` | Modified | Added `[onnx]` optional dependencies |

## Testing Results
- No new tests in this task (infrastructure only).
- Existing 35 tests: 35/35 passing.

## Final State
**Status**: DONE | **Date**: 2026-02-25
