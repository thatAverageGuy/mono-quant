# CONTEXT

## Current State
**Task:** T-026–T-029 — DONE (Phase 8 complete)
**Phase:** Phase 8 (Unified Export API) — COMPLETE
**Branch:** dev
**Last Commit:** pending push
**Date:** 2026-02-26

## Previous Session Summary
Phase 7 GGUF Export (T-022–T-025) completed and committed in previous session.

## Current Task State

Phase 8 Unified Export API — fully implemented, 103/103 tests passing (9 skipped: gguf-py not installed):

| Task | Description | Status |
|------|-------------|--------|
| T-026 | export/orchestrator.py + result.export() + list_formats() | DONE |
| T-027 | Unified CLI `export` command (replaces export/export-gptq/export-gguf) | DONE |
| T-028 | ExportWarning, validate_export_pre/post in validators.py | DONE |
| T-029 | result.convert(bits) + monoquant convert CLI command | DONE |

**Test suite:** 103 passed, 9 skipped (gguf-py), 0 failed

**New tests added:**
- `tests/test_export_orchestrator.py` — 12 tests
- `tests/test_cli_export.py` — 8 tests
- `tests/test_export_validation.py` — 11 tests
- `tests/test_convert.py` — 6 tests

**Breaking CLI change:** `monoquant export-gptq` and `monoquant export-gguf` removed.
Use `monoquant export --format gptq/gguf` instead. Python API unchanged.

## Next Steps

1. [ ] Follow commit procedure for T-026, T-027, T-028, T-029
2. [ ] Push to dev
3. [ ] Consider PR dev → main for v2.0

## Open Questions / Decisions Pending

- T-038 (calibration-based conversion): deferred, stub at docs/dev/tasks/T-038/DETAIL.md
- llama.cpp manual test (T-025): still requires llama.cpp binary; procedure documented

## Blockers

None.

## Quick Links

- Tasks index: docs/dev/tasks/TASKS.md
- T-026 log: docs/dev/tasks/T-026/IMPL_LOG.md
- T-027 log: docs/dev/tasks/T-027/IMPL_LOG.md
- T-028 log: docs/dev/tasks/T-028/IMPL_LOG.md
- T-029 log: docs/dev/tasks/T-029/IMPL_LOG.md
- Architecture: docs/dev/ARCHITECTURE.md
