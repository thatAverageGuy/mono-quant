# CONTEXT

## Current State
**Task:** BF-015 — DONE (pending commit alongside BF-014)
**Phase:** Sequential fix run. BF-014 + BF-015 done; CL-002 → T-040 → BF-016 remain.
**Branch:** dev
**Last Commit:** 8a0152b (T-029 + docs sync)
**Date:** 2026-02-27

## Previous Session Summary
Phase 8 (T-026–T-029) fully implemented and pushed to dev. Manual testing began.
Test A1 (simple MLP ONNX) passed. Test A2 (OPT-125m ONNX) revealed 3 bugs fixed in
BF-014. Post-BF-014 investigation identified 4 more items for system stability:
BF-015, CL-002, T-040, BF-016. User approved sequential execution.

## Current Task State

**BF-014 — all done, pre-commit:**
- Code: DONE
- Tests: DONE (107 total passing at time of writing; 109 after BF-015)
- IMPL_LOG: DONE
- TASKS.md: DONE
- CHANGELOG.md: DONE
- CONTEXT.md: DONE (updated below)
- Commit: PENDING USER APPROVAL (to be committed together with BF-015)

**BF-015 — DONE:**
- Code: DONE (3 sites in quantizers.py: isinstance → type-is)
- Tests: DONE (2 new; 109 total passing, 9 skipped, 0 failures)
- IMPL_LOG: DONE
- TASKS.md: DONE (moved to Completed)
- CHANGELOG.md: DONE
- CONTEXT.md: DONE (this file)
- Commit: PENDING USER APPROVAL (commit BF-014 + BF-015 together)

## Commit Plan

BF-014 and BF-015 will be committed as TWO separate commits (one per task ID), both
pushed to dev in sequence before moving to CL-002.

Staging for BF-014 commit:
- src/mono_quant/api/quantize.py
- src/mono_quant/export/common/validators.py
- src/mono_quant/export/onnx.py
- tests/test_api.py
- tests/test_export_validation.py
- tests/test_onnx_export.py (4 new tests)
- docs/dev/tasks/BF-014/DETAIL.md
- docs/dev/tasks/BF-014/IMPL_LOG.md
- docs/dev/tasks/TASKS.md
- CHANGELOG.md
- CONTEXT.md

Staging for BF-015 commit (on top of BF-014):
- src/mono_quant/core/quantizers.py
- tests/test_bugfixes.py (2 new tests)
- docs/dev/tasks/BF-015/DETAIL.md
- docs/dev/tasks/BF-015/IMPL_LOG.md
- docs/dev/tasks/TASKS.md
- CHANGELOG.md
- CONTEXT.md

NOT committed (gitignored or excluded):
- mq_manual_test/ (diagnostic scripts)
- test_results.txt

## Manual Test Status

| Test | What | Platform | Status |
|------|------|----------|--------|
| A1 | ONNX export — simple MLP | Windows | PASSED |
| A2 | ONNX export — OPT-125m | Windows | FIXED (BF-014 + BF-015) — re-run needed |
| B | GPTQ export → vLLM load + generate | Linux (Ubuntu SSD) | NOT RUN |
| C | GGUF export → llama.cpp load + generate | Linux or Windows | NOT RUN |
| D | CLI smoke tests | Windows | NOT RUN |
| E | result.convert() Python API | Windows | NOT RUN |

## Next Steps

1. [ ] Get user approval → commit BF-014 (separate commit)
2. [ ] Get user approval → commit BF-015 (separate commit)
3. [ ] Implement CL-002 (version 1.1.0 → 2.0.0)
4. [ ] Implement T-040 (dynamo=True ONNX export)
5. [ ] Implement BF-016 (validate_onnx_model dtype fix, depends T-040)
6. [ ] Re-run test A2 after all fixes committed
7. [ ] Run tests B, C, D, E
8. [ ] Raise PR dev → main for v2.0 release

## Open Questions / Decisions Pending

- T-038 (calibration-based conversion): deferred, stub at docs/dev/tasks/T-038/DETAIL.md
- PyPI publish: not done yet, pyproject.toml is configured

## Blockers

None.

## Quick Links

- BF-014 detail: docs/dev/tasks/BF-014/DETAIL.md
- BF-014 log: docs/dev/tasks/BF-014/IMPL_LOG.md
- BF-015 detail: docs/dev/tasks/BF-015/DETAIL.md
- BF-015 log: docs/dev/tasks/BF-015/IMPL_LOG.md
- T-040 detail: docs/dev/tasks/T-040/DETAIL.md
- Tasks index: docs/dev/tasks/TASKS.md
- Architecture: docs/dev/ARCHITECTURE.md
