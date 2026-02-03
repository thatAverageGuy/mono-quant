---
phase: 04-user-interfaces
plan: 01
subsystem: api
tags: [python-api, quantization, dataclass, exceptions, unified-interface]

# Dependency graph
requires:
  - phase: 03-advanced-calibration-&-int4
    provides: static_quantize, dynamic_quantize, QuantizationInfo, save_model, load_model, validate_quantization
provides:
  - Unified quantize() function with dynamic/static dispatch
  - QuantizationResult dataclass with convenience methods (.save(), .validate(), __bool__)
  - Custom exception hierarchy with actionable suggestions
  - Public API exports (quantize, QuantizationResult, exceptions)
affects: [04-user-interfaces]

# Tech tracking
tech-stack:
  added: [tqdm]
  patterns: [unified-api-pattern, result-object-pattern, exception-with-suggestions]

key-files:
  created:
    - src/mono_quant/api/exceptions.py
    - src/mono_quant/api/result.py
    - src/mono_quant/api/quantize.py
    - src/mono_quant/api/__init__.py
  modified:
    - src/mono_quant/__init__.py
    - pyproject.toml

key-decisions:
  - "Unified quantize() function dispatches to dynamic_quantize or static_quantize based on dynamic flag"
  - "bits parameter (4/8/16) maps to dtype internally (4/8 -> qint8, 16 -> float16)"
  - "scheme parameter uses string values ('symmetric'/'asymmetric') instead of enum for simpler API"
  - "Hybrid error approach: exceptions raise, but Result also has .success flag"
  - "show_progress parameter defaults to False (silent for library use)"

patterns-established:
  - "Pattern 1: Single-entry-point API - quantize() dispatches based on parameters"
  - "Pattern 2: Result object pattern - QuantizationResult encapsulates model, info, success, errors, warnings"
  - "Pattern 3: Actionable exceptions - all exceptions support suggestion parameter"
  - "Pattern 4: Input normalization - accepts nn.Module, state_dict, or file path"

# Metrics
duration: 3min
completed: 2026-02-03
---

# Phase 4 Plan 1: Python API Summary

**Unified quantize() function with dynamic/static dispatch, QuantizationResult dataclass with convenience methods, and custom exception hierarchy with actionable suggestions**

## Performance

- **Duration:** 3 min
- **Started:** 2026-02-03T17:04:12Z
- **Completed:** 2026-02-03T17:07:25Z
- **Tasks:** 4
- **Files modified:** 6

## Accomplishments

- Created unified `quantize()` function that automatically dispatches to `dynamic_quantize()` or `static_quantize()` based on the `dynamic` parameter
- Built `QuantizationResult` dataclass with `.save()`, `.validate()`, and `__bool__()` convenience methods
- Implemented custom exception hierarchy (`MonoQuantError`, `QuantizationError`, `ValidationError`, `ConfigurationError`, `InputError`) with actionable suggestions
- Updated package exports to expose `quantize()` at top level and added `tqdm` to main dependencies

## Task Commits

Each task was committed atomically:

1. **Task 1: Create custom exception hierarchy** - `275491d` (feat)
2. **Task 2: Create QuantizationResult dataclass** - `e29038c` (feat)
3. **Task 3: Create unified quantize() function** - `6d14ab2` (feat)
4. **Task 4: Update package exports and dependencies** - `8a7b25b` (feat)

## Files Created/Modified

### Created

- `src/mono_quant/api/exceptions.py` - Custom exception hierarchy with suggestion support
- `src/mono_quant/api/result.py` - QuantizationResult dataclass with convenience methods
- `src/mono_quant/api/quantize.py` - Unified quantize() function with parameter validation and dispatch
- `src/mono_quant/api/__init__.py` - API module exports

### Modified

- `src/mono_quant/__init__.py` - Added quantize to top-level exports, updated docstring
- `pyproject.toml` - Added tqdm>=4.66 to main dependencies

## Decisions Made

- **Unified function approach:** Single `quantize()` function dispatches based on `dynamic` flag instead of separate `dynamic_quantize_api()` and `static_quantize_api()` functions
- **Parameter mapping:** `bits` (4/8/16) maps to dtype internally (4/8 -> qint8, 16 -> float16) for simpler user-facing API
- **String scheme:** `scheme` parameter uses string values ("symmetric"/"asymmetric") instead of enum for API simplicity
- **Hybrid error handling:** Exceptions raise for immediate feedback, but `QuantizationResult.success` flag also indicates status
- **Progress defaults:** `show_progress` parameter defaults to `False` for library use (silent by default)

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None - all tasks completed without issues.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- Python API foundation complete with unified `quantize()` function
- Result object pattern established for CLI integration (Plan 04-02)
- Exception hierarchy ready for CLI error formatting
- tqdm dependency added for progress bar support in both API and CLI
- Ready to proceed with Plan 04-02: CLI implementation

---
*Phase: 04-user-interfaces*
*Plan: 01*
*Completed: 2026-02-03*
