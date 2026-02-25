# Implementation Log: CL-001

## Summary
Code quality cleanup addressing M1 (version semver), M2 (safetensors pin),
M3 (private names in __all__), M4 (test stub in production code),
M6 (accuracy warning fires unconditionally), and M7 (observer docs).

## What Was Done

### M1 — Version string
- `src/mono_quant/__init__.py`: `__version__ = "1.1"` → `"1.1.0"`
- `pyproject.toml`: `version = "1.1"` → `"1.1.0"`

### M2 — safetensors version
- `pyproject.toml`: `"safetensors>=0.3"` → `"safetensors>=0.4"`
  (matches the existing error message in `io/formats.py`)

### M3 — Private names in __all__
- `src/mono_quant/core/__init__.py`: Removed `_select_layers_by_type`,
  `_select_layers_by_name` from `__all__` (private functions, never for
  public use). No `__all__` found in `quantizers.py` or `observers.py`
  (those modules have no `__all__` defined).

### M4 — Remove test stub from production code
- `src/mono_quant/core/quantizers.py`: Removed `test_models_from_any_source()`
  function entirely (lines 1024–1090). The function was development scaffolding,
  never moved to the test suite, never called by any production code.
- `src/mono_quant/core/__init__.py`: Removed `test_models_from_any_source`
  from both the import and `__all__`.

### M6 — Accuracy warning fires unconditionally
- `src/mono_quant/core/quantizers.py`: Changed `all_layers_quantized_warning=True`
  to `all_layers_quantized_warning=(group_size > 0)` in the `check_accuracy_warnings`
  call inside `static_quantize`. The "all layers quantized" warning is only
  meaningful for INT4 (where skipping embeddings/norms matters). INT8 calls with
  default `group_size=0` no longer emit this spurious warning.

### M7 — Observer attachment contract
- `src/mono_quant/calibration/runner.py`: The `run_calibration` docstring already
  documents the observer attachment contract adequately. No separate
  `attach_observers`/`detach_observers` functions exist in the current codebase;
  the contract is documented inline in `static_quantize` where hooks are
  registered/removed. No additional docstring changes required.

## Files Changed

| File | Change Type | What Changed |
|------|-------------|--------------|
| src/mono_quant/__init__.py | Modified | `__version__` → `"1.1.0"` |
| pyproject.toml | Modified | `version` → `"1.1.0"`, `safetensors>=0.3` → `>=0.4` |
| src/mono_quant/core/__init__.py | Modified | Removed private exports and test stub import from `__all__` |
| src/mono_quant/core/quantizers.py | Modified | Removed `test_models_from_any_source` function; `all_layers_quantized_warning` gated on `group_size > 0` |
| tests/test_bugfixes.py | Modified | Two new CL-001 tests |

## Testing Results
- Unit: `test_version_is_semver` — PASS (`__version__ == "1.1.0"` is valid semver)
- Unit: `test_test_models_not_in_public_api` — PASS (function removed from API)
- All 28 tests passing

## Final State
**Status**: DONE | **Date**: 2026-02-25
