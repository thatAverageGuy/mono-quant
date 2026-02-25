# CONTEXT

## Current State

**Task:** None active — docs/dev/ bootstrap complete
**Phase:** v2.0 Phase 6 (GPTQ/AWQ Export) — not started, ready to plan
**Branch:** dev
**Last Commit:** b02a2b4 (fix: make mypy type checking non-blocking in CI/CD)
**Date:** 2026-02-24

## Previous Session Summary

No previous CONTEXT.md existed. This session was the initial docs/dev/ bootstrap.

- Traversed all `.planning/` files to extract project history and intent
- Set up full `docs/dev/` structure from scratch
- Created ARCHITECTURE.md, STATE_MACHINES.md, SPEC.md, CONTRIBUTING.md
- Created TASKS.md with complete index of T-001 to T-029
- Created DETAIL.md + IMPL_LOG.md for all completed tasks (T-001 to T-017, BF-001)
- Created DETAIL.md for all pending tasks (T-018 to T-029)
- Created 7 ADRs documenting key architectural decisions
- Switched to `dev` branch (already existed with CI/CD fixes above main)

## Current Task State

No active task. All tasks marked correctly in TASKS.md:
- T-001 to T-017 + BF-001: DONE
- T-018 to T-029: TODO

## Next Steps

1. [ ] Plan Phase 6 in detail (T-018: 4-bit packing is the logical start)
2. [ ] Verify ONNX export tests still pass on dev branch before starting Phase 6
3. [ ] Begin T-018: study AutoGPTQ packing format, implement GPTQ exporter
4. [ ] After T-018-T-021 complete: plan T-022 (GGUF binary writer)

## Open Questions / Decisions Pending

- T-021: Can CI/CD environment have vLLM installed? If not, manual test procedure needed.
- T-023: Need to verify exact group sizes for Q4_K_M vs Q4_K_S from llama.cpp source.
- T-028: Should cross-format conversion (GPTQ ↔ ONNX) be in scope for v2.0?
  Proposed answer: No — requires original model, out of scope.

## Blockers

None.

## Quick Links

- Architecture: docs/dev/ARCHITECTURE.md
- State machines: docs/dev/STATE_MACHINES.md
- Spec: docs/dev/SPEC.md
- Tasks: docs/dev/tasks/TASKS.md
- Next task: docs/dev/tasks/T-018/DETAIL.md
- ADRs: docs/dev/adr/
- Planning source: .planning/ (original planning files, preserved)
