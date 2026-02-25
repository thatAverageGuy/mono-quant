# CONTEXT

## Current State

**Task:** None active — 15 audit fixes complete, awaiting commit approval
**Phase:** Pre-Phase-6 — all non-calibration audit fixes done
**Branch:** dev
**Last Commit:** 828671d (AX-002: add CLAUDE.md to version control)
**Date:** 2026-02-25

## Previous Session Summary

Full correctness audit of the entire codebase (LEVEL 0) produced 17 tasks.
Task documents (DETAIL.md) created for BF-002 through BF-013, T-030 through
T-033, and CL-001. TASKS.md updated with full audit finding index.

First 6 tasks committed (T-030, BF-004, BF-013, BF-002, BF-003, BF-009).
Remaining 9 tasks implemented and tested in the current session.

## Current Task State

**15 tasks implemented and tested, pending commit approval:**

Previously committed:
- T-030: DONE — `export_to_onnx` stub added; T-014–T-017 status corrected
- BF-004: DONE — 11× `_click.Context.exit(N)` → `raise SystemExit(N)` in CLI
- BF-013: DONE — `|| echo` fallback removed from CI pytest step
- BF-002: DONE — dual `QuantizationInfo` resolved; `result.save()` works
- BF-003: DONE — spurious `-8` removed from INT4 symmetric formula
- BF-009: DONE — `dequantize_model` buffer loop fixed with `register_buffer`

New (awaiting commit approval — 9 separate commits needed):
- BF-008: DONE — always-False isinstance guard removed from nested named_modules() loop
- BF-006: DONE — `group_size` default changed 128→0; INT4 skip no longer injected into INT8
- BF-007: DONE — fallback replaced with RuntimeError (was returning corrupt INT8 as INT4)
- BF-005: DONE — mutable default `skip_set=set()` → `None` sentinel
- BF-010: DONE — `dtype` threaded through `quantize_embedding_module` → `from_embedding`
- BF-011: DONE — `_test_load_run` uses `copy.deepcopy` before `load_state_dict`
- BF-012: DONE — hardcoded `> 100` replaced with 10-sigma relative outlier check
- T-033: DONE — file path input raises `TypeError` with helpful message
- CL-001: DONE — version `1.1.0`, safetensors `>=0.4`, test stub removed, M6/M7

**Test suite:** 28/28 passing (5 original + 23 in `test_bugfixes.py`)

**Remaining audit tasks:**
- T-032: HistogramObserver rewrite (C5)
- T-031: activation-based calibration (C4, depends on T-032)

## Next Steps

1. [ ] Get user approval for the 9 new completed tasks
2. [ ] Commit (9 separate commits, one per task ID)
3. [ ] Continue with T-032 (HistogramObserver) — next highest severity
4. [ ] Then T-031 (activation calibration — depends on T-032)
5. [ ] Then Phase 5 ONNX implementation (T-034+)

## Open Questions / Decisions Pending

- T-030: New task IDs T-034+ for actual ONNX implementation (post-audit).

## Blockers

None.

## Quick Links

- Tasks index: docs/dev/tasks/TASKS.md
- Next task: docs/dev/tasks/T-032/DETAIL.md
- Architecture: docs/dev/ARCHITECTURE.md
- Previous session: docs/dev/session_context/session_2026-02-24.md
