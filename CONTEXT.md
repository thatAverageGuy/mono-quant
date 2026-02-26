# CONTEXT

## Current State
**Task:** BF-016 — DONE (pending commit)
**Phase:** Sequential fix run complete. All active tasks done. Ready for manual tests + PR.
**Branch:** dev
**Last Commit:** c6c5a19 (T-040)
**Date:** 2026-02-27

## Previous Session Summary
Sequential fix run following manual test A2 failure:
- BF-014 (9b5f141): three ONNX/dynamic-quant bugs fixed
- BF-015 (51a1740): nn.Embedding subclass quantization fixed
- CL-002 (8044cab): version bumped 1.1.0 → 2.0.0
- T-040 (c6c5a19): dynamo=True ONNX export path added
- BF-016: validate_onnx_model dtype fix — DONE, pending commit

## Current Task State

**BF-016 — all done, pre-commit:**
- Code: DONE (validators.py: dtype inferred from ONNX input spec)
- Tests: DONE (1 new; 113 total passing, 9 skipped)
- IMPL_LOG: DONE
- TASKS.md: DONE (no active tasks remain)
- CHANGELOG.md: DONE
- CONTEXT.md: DONE (this file)
- Commit: PENDING USER APPROVAL

## Active Tasks

None. All scoped tasks complete.

## Manual Test Status

| Test | What | Platform | Status |
|------|------|----------|--------|
| A1 | ONNX export — simple MLP | Windows | PASSED |
| A2 | ONNX export — OPT-125m | Windows | FIXED — re-run pending |
| B | GPTQ export → vLLM load + generate | Linux (Ubuntu SSD) | NOT RUN |
| C | GGUF export → llama.cpp load + generate | Linux or Windows | NOT RUN |
| D | CLI smoke tests | Windows | NOT RUN |
| E | result.convert() Python API | Windows | NOT RUN |

## Next Steps

1. [ ] Get user approval → commit BF-016
2. [ ] Re-run test A2 with dynamo=True to confirm OPT-125m works end-to-end
3. [ ] Run tests B, C, D, E
4. [ ] Record all results, update relevant IMPL_LOGs
5. [ ] Raise PR dev → main for v2.0 release
6. [ ] Tag v2.0.0 on main
7. [ ] Publish to PyPI

## Open Questions / Decisions Pending

- T-038 (calibration-based conversion): deferred
- PyPI publish: not done yet, pyproject.toml configured

## Blockers

None.

## Quick Links

- Tasks index: docs/dev/tasks/TASKS.md
- Architecture: docs/dev/ARCHITECTURE.md
- CHANGELOG: CHANGELOG.md
