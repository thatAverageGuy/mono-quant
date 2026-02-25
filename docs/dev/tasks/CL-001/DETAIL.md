# CL-001: Code quality cleanup — version strings, exports, stale test code, observer contract

## Status
TODO

## Audit Reference
M1, M2, M3, M4, M6, M7 (Medium/Low)

## Problem
Several low-severity quality issues found in the audit:

### M1 — Version string not semver
`src/mono_quant/__init__.py`: `__version__ = "1.1"` (not semver).
`pyproject.toml`: `version = "1.1"`.
Semver requires three parts: `MAJOR.MINOR.PATCH`. Current value breaks any
tooling that strictly parses semver (e.g. release automation).

### M2 — safetensors version mismatch
`pyproject.toml`: `safetensors>=0.3`
Error message in `io/formats.py`: `"safetensors>=0.4 required"`
One of these is wrong. Align them.

### M3 — Private functions in `__all__`
`core/quantizers.py` and `core/observers.py` export private functions (names
starting with `_`) in `__all__`. These are never intended for public use and
their inclusion is a copy-paste error.

### M4 — Test code embedded in production module
`core/quantizers.py` contains `test_models_from_any_source()` — a function
with test logic (model creation, assertions) in a production source file.
It is not part of any public API and appears to be a development stub never
moved to the test suite.

### M6 — Accuracy warning fires unconditionally
`api/quantize.py`: `check_accuracy_warnings(result)` is called after
every quantization. `check_accuracy_warnings` appears to emit warnings
regardless of whether the model has any actual accuracy concern.

### M7 — Observer attachment contract undocumented
`calibration/runner.py`'s `attach_observers`/`run_calibration`/`detach_observers`
functions assume a specific internal state on the model (hooks attached in a
specific place) but this contract is nowhere documented. Adding a docstring
clarifying the expected state machine is necessary before T-031 implementation.

## Requirements
1. Fix version strings to semver (`1.1.0`).
2. Align safetensors version in pyproject.toml and error messages.
3. Remove private names from `__all__`.
4. Remove `test_models_from_any_source` from production code.
5. Fix `check_accuracy_warnings` to only warn when thresholds are exceeded.
6. Document observer attachment contract with docstrings.

## Decisions
- **Decision:** Bump version to `1.1.0` (not `2.0.0`) — no breaking changes
  in this cleanup.
- **Decision:** Set `safetensors>=0.4` in pyproject.toml (match the error
  message which likely reflects actual minimum tested version).
- **Decision:** Move `test_models_from_any_source` to `tests/` as a proper
  pytest test or delete it if it duplicates existing test coverage.

## Success Criteria
- [ ] `__version__` is semver (X.Y.Z format)
- [ ] `safetensors` pin matches error message
- [ ] No `_`-prefixed names in `__all__` in any module
- [ ] `test_models_from_any_source` removed from `core/quantizers.py`
- [ ] `check_accuracy_warnings` only warns when a real threshold is exceeded
- [ ] Observer functions have docstrings explaining attachment contract

## Dependencies
- None (pure cleanup, no logic changes)

## Implementation Guidance

### M1 — Version
```python
# src/mono_quant/__init__.py
__version__ = "1.1.0"
```
```toml
# pyproject.toml
version = "1.1.0"
```

### M2 — safetensors
```toml
# pyproject.toml
"safetensors>=0.4",
```
Or update the error message if `>=0.3` is truly the correct minimum.

### M3 — __all__ cleanup
```python
# Remove from __all__ in each module
__all__ = [name for name in __all__ if not name.startswith('_')]
```
Review each removal to ensure nothing in the public interface starts with `_`.

### M4 — Remove test function
Find `test_models_from_any_source` in `core/quantizers.py` and either:
- Delete it (if covered by `tests/test_api.py`)
- Move it to `tests/` as `test_quantize_models_from_any_source`

### M6 — Conditional accuracy warnings
In `api/quantize.py`, `check_accuracy_warnings` should check against a
threshold (e.g. SQNR < 20 dB) before appending to `result.warnings`.

### M7 — Observer docstrings
Add docstrings to `attach_observers`, `run_calibration`, `detach_observers`
in `calibration/runner.py` explaining the expected call sequence and module
state before/after each call.

## Testing Requirements
- Unit: Import `mono_quant`; verify `__version__` matches pyproject.toml
- Unit: Call `check_accuracy_warnings` with high-quality result; verify no
  spurious warning
- Coverage target: all changed code paths

## Open Questions
<!-- MUST be empty before implementation begins -->
