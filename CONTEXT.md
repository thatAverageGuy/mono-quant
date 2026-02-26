# CONTEXT

## Current State
**Task:** T-040 — DONE (pending commit)
**Phase:** Sequential fix run. BF-014 + BF-015 + CL-002 committed; T-040 done; BF-016 remains.
**Branch:** dev
**Last Commit:** 8044cab (CL-002)
**Date:** 2026-02-27

## Previous Session Summary
Phase 8 (T-026–T-029) fully implemented and pushed. Manual test A2 (OPT-125m)
revealed bugs fixed in BF-014 and BF-015. Committed sequentially: BF-014 (9b5f141),
BF-015 (51a1740), CL-002 (8044cab). T-040 implemented and pending commit.

## Current Task State

**T-040 — all done, pre-commit:**
- Code: DONE (onnx.py + onnx_impl.py + orchestrator.py + pyproject.toml)
- Tests: DONE (3 new dynamo tests; 112 total passing, 9 skipped)
- IMPL_LOG: DONE
- TASKS.md: DONE
- CHANGELOG.md: DONE
- CONTEXT.md: DONE (this file)
- Commit: PENDING USER APPROVAL

## Manual Test Status

| Test | What | Platform | Status |
|------|------|----------|--------|
| A1 | ONNX export — simple MLP | Windows | PASSED |
| A2 | ONNX export — OPT-125m | Windows | FIXED (BF-014 + BF-015 + T-040) — re-run needed |
| B | GPTQ export → vLLM load + generate | Linux (Ubuntu SSD) | NOT RUN |
| C | GGUF export → llama.cpp load + generate | Linux or Windows | NOT RUN |
| D | CLI smoke tests | Windows | NOT RUN |
| E | result.convert() Python API | Windows | NOT RUN |

## Next Steps

1. [ ] Get user approval → commit T-040
2. [ ] Implement BF-016 (validate_onnx_model dtype fix)
3. [ ] Re-run test A2 after all fixes committed
4. [ ] Run tests B, C, D, E
5. [ ] Raise PR dev → main for v2.0 release

## Open Questions / Decisions Pending

- T-038 (calibration-based conversion): deferred
- PyPI publish: not done yet, pyproject.toml configured

## Blockers

None.

## Quick Links

- BF-016 detail: docs/dev/tasks/BF-016/ (TBD)
- Tasks index: docs/dev/tasks/TASKS.md
- Architecture: docs/dev/ARCHITECTURE.md
