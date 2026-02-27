# CONTEXT

## Current State
**Task:** T-041 — DONE, pending commit approval
**Branch:** dev
**Last Commit:** 64f1a63 (BF-017)
**Date:** 2026-02-27

## Previous Session Summary
Full fix run: BF-015 → BF-014 → CL-002 → T-040 → BF-016 → BF-017 all committed.
Manual test A2 (OPT-125m ONNX) PASSED. T-041 root cause investigated, implemented,
and tested in this session.

## T-041 — What Was Implemented

**Root cause** (confirmed empirically): dynamo lowers `F.linear(x, w, b)` as
`MatMul(x, w.T)` for attention projection weights. The transposed weight gets stored
as an anonymous `val_N` constant — original parameter name lost.

**Fix**: `_build_dynamo_name_map(model, proto)` in `qdq_inserter.py` — matches val_N
initializers to named parameters by value comparison (direct + transposed). Called
from `onnx.py` Step 6 when `dynamo=True`. Axis adjusted from 0→1 for transposed weights.

**Result**: fc1/fc2 ALREADY had QDQ nodes (Gemm path). Now attention projections
(k/q/v/out_proj) also get QDQ nodes on dynamo export.

## Current Task State

**T-041 implementation**: DONE (pending commit approval)
- Code: DONE — `qdq_inserter.py` + `onnx.py`
- Tests: DONE — 117/117 passing (2 new tests: unit + integration)
- IMPL_LOG.md: DONE
- TASKS.md: DONE (T-041 → DONE)
- CHANGELOG.md: DONE

## Next Steps

1. [ ] Get user approval and commit T-041
2. [ ] Optionally re-run manual test A2 (OPT-125m) to confirm attention QDQ in prod
3. [ ] Run manual tests D and E (CLI smoke + result.convert() — Windows, no new installs)
4. [ ] Boot Ubuntu SSD for manual tests B and C (GPTQ → vLLM, GGUF → llama.cpp)
5. [ ] Record all results
6. [ ] Raise PR dev → main for v2.0 release
7. [ ] Tag v2.0.0 on main
8. [ ] Publish to PyPI

## Pending (Future, post-v2.0)

- T-038: Calibration-based conversion (result.convert with calibration_data)

## Blockers

None.

## Quick Links

- T-041 detail: docs/dev/tasks/T-041/DETAIL.md
- T-041 impl log: docs/dev/tasks/T-041/IMPL_LOG.md
- Tasks index: docs/dev/tasks/TASKS.md
- CHANGELOG: CHANGELOG.md
