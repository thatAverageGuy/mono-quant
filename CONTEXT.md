# CONTEXT

## Current State

**Task:** None active — 6 audit fixes complete, awaiting commit approval
**Phase:** Pre-Phase-6 — audit fixes in progress
**Branch:** dev
**Last Commit:** 828671d (AX-002: add CLAUDE.md to version control)
**Date:** 2026-02-25

## Previous Session Summary

Full correctness audit of the entire codebase (LEVEL 0) produced 17 tasks.
Task documents (DETAIL.md) created for BF-002 through BF-013, T-030 through
T-033, and CL-001. TASKS.md updated with full audit finding index.

## Current Task State

**6 tasks implemented and tested, pending commit approval:**
- T-030: DONE — `export_to_onnx` stub added; T-014–T-017 status corrected
- BF-004: DONE — 11× `_click.Context.exit(N)` → `raise SystemExit(N)` in CLI
- BF-013: DONE — `|| echo` fallback removed from CI pytest step
- BF-002: DONE — dual `QuantizationInfo` resolved; `result.save()` works
- BF-003: DONE — spurious `-8` removed from INT4 symmetric formula
- BF-009: DONE — `dequantize_model` buffer loop fixed with `register_buffer`

**Test suite:** 15/15 passing (5 original + 10 new in `test_bugfixes.py`)

**Remaining audit tasks (all TODO):**
- BF-005: mutable default argument
- BF-006: INT4 skip list injected into INT8
- BF-007: `quantize_weight_int4` fallback wrong format
- BF-008: nested layer detection always False
- BF-010: `quantize_embedding_module` drops dtype
- BF-011: `_test_load_run` mutates model under test
- BF-012: hardcoded weight range threshold
- T-031: activation-based calibration (dead code)
- T-032: HistogramObserver rewrite
- T-033: file path input fix in `quantize()`
- CL-001: code quality cleanup

## Next Steps

1. [ ] Get user approval for the 6 completed tasks
2. [ ] Commit (6 separate commits, one per task ID)
3. [ ] Continue with BF-008 (nested layer detection) — next highest severity
4. [ ] Then BF-006, BF-007, BF-005, BF-010, BF-011, BF-012
5. [ ] Then T-033, CL-001
6. [ ] Then T-031, T-032 (calibration/observer fixes — largest scope)
7. [ ] Then Phase 5 ONNX implementation (T-034+)

## Open Questions / Decisions Pending

- BF-007: Raise or fix the fallback? See DETAIL.md — lean toward raise.
- T-030: New task IDs T-034+ for actual ONNX implementation.

## Blockers

None.

## Quick Links

- Tasks index: docs/dev/tasks/TASKS.md
- Next task: docs/dev/tasks/BF-008/DETAIL.md
- Architecture: docs/dev/ARCHITECTURE.md
- Previous session: docs/dev/session_context/session_2026-02-24.md
