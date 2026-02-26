# CONTEXT

## Current State
**Task:** Manual testing — IN PROGRESS (not started yet)
**Phase:** Phase 8 complete. Pending: manual test execution + results recording.
**Branch:** dev
**Last Commit:** 8a0152b (T-029 + docs sync)
**Date:** 2026-02-26

## Previous Session Summary
Phase 8 (T-026–T-029) fully implemented and pushed to dev in this session:
- `export/orchestrator.py` — unified export dispatch, format auto-detection
- `result.export()` and `result.convert()` on QuantizationResult
- Unified `monoquant export` CLI (replaced export/export-gptq/export-gguf)
- `monoquant convert` CLI command
- ExportWarning + validate_export_pre/post
- 37 new tests, 103 total passing
- Full docs sync (CHANGELOG, README, CLI docs, quickstart, ARCHITECTURE)

## Current Task State

**All code is done and pushed. What remains is manual testing only.**

Manual test guide lives at: `MANUAL_TESTS.md` (repo root, gitignored — do not commit).

| Test | What | Platform | Status |
|------|------|----------|--------|
| A | ONNX export — simple MLP model, full ONNX Runtime forward pass | Windows | NOT RUN |
| A | ONNX export — opt-125m (expect graceful tracing error) | Windows | NOT RUN |
| B | GPTQ export → vLLM load + generate | Linux (Ubuntu SSD) | NOT RUN |
| C | GGUF export → llama.cpp load + generate | Linux or Windows | NOT RUN |
| D | CLI smoke tests (list-formats, auto-detect, bad ext, convert, missing args) | Windows | NOT RUN |
| E | result.convert() Python API — warning emitted, models independent | Windows | NOT RUN |

## How to Resume

1. Read `MANUAL_TESTS.md` — it has complete step-by-step instructions for each test,
   including all installs from scratch, exact commands, expected output, and a
   pass/fail criteria table.
2. Suggested order: D and E first (no new installs), then A (onnx already installed),
   then boot Ubuntu SSD for B and C.
3. After running tests, paste the results table from the bottom of `MANUAL_TESTS.md`
   into the chat. The agent will update the relevant IMPL_LOGs and mark tests executed.

## Hardware context (for the agent resuming)
- Windows 11, RTX 4050 Laptop 6 GB VRAM, 16 GB RAM
- Ubuntu on external 1 TB SSD (available for vLLM / llama.cpp)
- vLLM requires Linux — use the Ubuntu SSD for Test B
- llama.cpp works on both platforms

## Next Steps After Manual Tests

1. [ ] Run manual tests (see MANUAL_TESTS.md)
2. [ ] Record results and update IMPL_LOGs for T-021 (GPTQ/vLLM) and T-025 (GGUF/llama.cpp)
3. [ ] Raise PR dev → main for v2.0 release
4. [ ] Tag v2.0.0 on main
5. [ ] Publish to PyPI (pyproject.toml is ready)
6. [ ] T-038 (calibration-based conversion) — whenever desired

## Open Questions / Decisions Pending

- T-038 (calibration-based conversion): deferred, stub at docs/dev/tasks/T-038/DETAIL.md
- PyPI publish: not done yet, pyproject.toml is configured

## Blockers

None.

## Quick Links

- **Manual test guide:** MANUAL_TESTS.md (gitignored, repo root)
- Tasks index: docs/dev/tasks/TASKS.md
- T-021 log (GPTQ manual test procedure): docs/dev/tasks/T-021/IMPL_LOG.md
- T-025 log (GGUF manual test procedure): docs/dev/tasks/T-025/IMPL_LOG.md
- T-026 log: docs/dev/tasks/T-026/IMPL_LOG.md
- T-027 log: docs/dev/tasks/T-027/IMPL_LOG.md
- T-028 log: docs/dev/tasks/T-028/IMPL_LOG.md
- T-029 log: docs/dev/tasks/T-029/IMPL_LOG.md
- Architecture: docs/dev/ARCHITECTURE.md
