# Implementation Log: CL-002

## Summary
Bumped version from `1.1.0` to `2.0.0` in two files. Reflects the major scope
of changes delivered through Phase 5–8 (three export formats, unified API,
audit bug-fix wave).

## Files Changed

| File | Change Type | What Changed |
|------|-------------|--------------|
| `pyproject.toml` | Modified | `version = "1.1.0"` → `version = "2.0.0"` |
| `src/mono_quant/__init__.py` | Modified | `__version__ = "1.1.0"` → `__version__ = "2.0.0"` |

## Testing Results
- `test_version_is_semver`: passing (validates X.Y.Z format)
- Full suite: 109 passed, 9 skipped, 0 failures

## Final State
**Status**: DONE | **Date**: 2026-02-27
