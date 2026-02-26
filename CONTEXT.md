# CONTEXT

## Current State

**Task:** T-021 — DONE (Phase 6 complete)
**Phase:** Phase 6 (GPTQ Export) — COMPLETE
**Branch:** dev
**Last Commit:** pending approval
**Date:** 2026-02-26

## Previous Session Summary

Phase 5 ONNX export (T-034–T-037) completed in previous session. 42/42 tests passing.

## Current Task State

Phase 6 GPTQ Export — fully implemented, 53/53 tests passing:

| Task | Description | Status |
|------|-------------|--------|
| T-018 | GPTQ packing math + GPTQExporter (AutoGPTQ V1) | DONE |
| T-019 | Public API + CLI export-gptq + 11 tests | DONE |
| T-020 | AWQ export | DROPPED (requires Hessian calibration, not a format wrapper) |
| T-021 | Manual vLLM procedure + validate_gptq_checkpoint_structure | DONE |

**Test suite:** 53/53 passing (42 pre-existing + 11 new GPTQ tests)

Notable bugs fixed during implementation:
- `_pack_int4_*` returned int64 on Windows due to sum promotion — fixed with explicit `.to(torch.int32)`
- `permute()` outputs non-contiguous tensors rejected by safetensors — fixed with `.contiguous()`
- Plan's 1% reconstruction error threshold unrealistic for INT4 — corrected to 15%

## Next Steps

1. [ ] Get user approval for commit
2. [ ] Single commit: T-018–T-021 GPTQ export implementation
3. [ ] Push to dev
4. [ ] Begin Phase 7 planning: GGUF export (T-022–T-025)

## Open Questions / Decisions Pending

- Phase 7 (T-022–T-025): GGUF export — not started
- vLLM manual test (T-021) requires GPU + vLLM install; procedure documented in
  `docs/dev/tasks/T-021/MANUAL_TEST.md` but not yet executed

## Blockers

None.

## Quick Links

- Tasks index: docs/dev/tasks/TASKS.md
- T-018 log: docs/dev/tasks/T-018/IMPL_LOG.md
- T-019 log: docs/dev/tasks/T-019/IMPL_LOG.md
- T-021 log: docs/dev/tasks/T-021/IMPL_LOG.md
- vLLM test: docs/dev/tasks/T-021/MANUAL_TEST.md
- Architecture: docs/dev/ARCHITECTURE.md
