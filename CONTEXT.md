# CONTEXT

## Current State
**Task:** CL-002 — DONE (pending commit)
**Phase:** Sequential fix run. BF-014 + BF-015 committed; CL-002 done; T-040 → BF-016 remain.
**Branch:** dev
**Last Commit:** 51a1740 (BF-015)
**Date:** 2026-02-27

## Previous Session Summary
Phase 8 (T-026–T-029) fully implemented and pushed. Manual test A2 (OPT-125m) revealed
bugs fixed in BF-014 and BF-015. User approved sequential execution of:
BF-015 → BF-014 commit → CL-002 → T-040 → BF-016.

BF-014 and BF-015 are committed and pushed (commits 9b5f141, 51a1740).

## Current Task State

**CL-002 — all done, pre-commit:**
- Code: DONE (pyproject.toml + __init__.py: 1.1.0 → 2.0.0)
- Tests: DONE (test_version_is_semver passing; 109 total passing)
- IMPL_LOG: DONE
- TASKS.md: DONE
- CHANGELOG.md: DONE (header updated to [2.0.0] - Unreleased)
- CONTEXT.md: DONE (this file)
- Commit: PENDING USER APPROVAL

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

1. [ ] Get user approval → commit CL-002
2. [ ] Implement T-040 (dynamo=True ONNX export)
3. [ ] Implement BF-016 (validate_onnx_model dtype fix, depends T-040)
4. [ ] Re-run test A2 after all fixes
5. [ ] Run tests B, C, D, E
6. [ ] Raise PR dev → main for v2.0 release

## Open Questions / Decisions Pending

- T-038 (calibration-based conversion): deferred
- PyPI publish: not done yet, pyproject.toml configured

## Blockers

None.

## Quick Links

- T-040 detail: docs/dev/tasks/T-040/DETAIL.md
- Tasks index: docs/dev/tasks/TASKS.md
- Architecture: docs/dev/ARCHITECTURE.md
