# CONTEXT

## Current State

**Task:** T-037 — DONE
**Phase:** Phase 5 (ONNX Export) — COMPLETE
**Branch:** dev
**Last Commit:** (pending — T-034 through T-037)
**Date:** 2026-02-25

## Previous Session Summary

All 17 audit fixes completed across 3 sessions (T-030, BF-002–BF-013, CL-001,
T-033, T-031, T-032). Phase 5 ONNX export (T-034–T-037) implemented in this session.

## Current Task State

Phase 5 ONNX Export — fully implemented and tested:

| Task | Description | Status |
|------|-------------|--------|
| T-034 | Export infrastructure, BaseExporter, validators, pyproject.toml | DONE |
| T-035 | QDQ node insertion: collect_quantization_params, insert_qdq_nodes | DONE |
| T-036 | ONNXExporter full pipeline: revert → export → QDQ → metadata | DONE |
| T-037 | CLI export command + 7 tests | DONE |

**Test suite:** 42/42 passing (35 pre-existing + 7 new ONNX tests)

Notable runtime fix: PyTorch 2.10 changed `torch.onnx.export` default to
`dynamo=True` (requires onnxscript). Fixed with `dynamo=False`.

## Next Steps

1. [ ] Push commit to dev
2. [ ] Begin Phase 6 planning: GPTQ/AWQ export (T-018–T-021)

## Open Questions / Decisions Pending

- Phase 6 (T-018–T-021): GPTQ/AWQ export — not started.
- INT4 QDQ in ONNX: deferred to opset >= 21 (future work).

## Blockers

None.

## Quick Links

- Tasks index: docs/dev/tasks/TASKS.md
- Active task: docs/dev/tasks/T-037/IMPL_LOG.md
- Architecture: docs/dev/ARCHITECTURE.md
- Previous session: docs/dev/session_context/session_2026-02-25.md
