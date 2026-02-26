# CONTEXT

## Current State
**Task:** BF-017 — DONE (pending commit)
**Phase:** Manual test A2 PASSED. All Windows ONNX tests complete.
**Branch:** dev
**Last Commit:** 4202433 (BF-016)
**Date:** 2026-02-27

## Previous Session Summary
Sequential fix run complete (BF-014 through BF-016 committed). Manual test A2
(OPT-125m ONNX) revealed two more blocking issues fixed in BF-017:
1. DynamicCache pytree error (all HF causal LMs) — disabled use_cache before tracing
2. Windows CP1252 UnicodeEncodeError from torch.onnx emoji log — reconfigure stdout

Test A2 now passes: 74 layers quantized, 627MB ONNX file written successfully.

## Current Task State

**BF-017 — all done, pre-commit:**
- Code: DONE (onnx.py: use_cache guard + stdout reconfigure)
- Tests: DONE (2 new; 115 total passing, 9 skipped)
- IMPL_LOG: DONE
- TASKS.md: DONE (T-041 added to Pending Future)
- CHANGELOG.md: DONE
- CONTEXT.md: DONE (this file)
- Commit: PENDING USER APPROVAL

## ONNX Export Behavior (documented)

**File size**: ONNX files are FP32-sized regardless of quantization. The QDQ approach
stores FP32 initializers + QuantizeLinear/DequantizeLinear node pairs. "Smart" runtimes
(TensorRT, ORT INT8 EP) fuse the Q→Op→DQ pattern into native INT8 kernels. Plain CPU
runtimes run FP32. No file-size savings until INT8 initializers (opset 21, T-041 scope).

**QDQ gap (T-041)**: Dynamo-exported complex models (OPT, LLaMA, etc.) do not get QDQ
nodes inserted — dynamo uses different initializer naming than TorchScript for nested
modules. ONNX is valid/runnable FP32. Deferred to post-v2.0.

## Manual Test Status

| Test | What | Platform | Status |
|------|------|----------|--------|
| A1 | ONNX export — simple MLP | Windows | PASSED |
| A2 | ONNX export — OPT-125m (dynamo=True) | Windows | PASSED (627MB, FP32, no QDQ) |
| B | GPTQ export → vLLM load + generate | Linux (Ubuntu SSD) | NOT RUN |
| C | GGUF export → llama.cpp load + generate | Linux or Windows | NOT RUN |
| D | CLI smoke tests | Windows | NOT RUN |
| E | result.convert() Python API | Windows | NOT RUN |

## Next Steps

1. [ ] Get user approval → commit BF-017
2. [ ] Run tests D and E (Windows, no new installs needed)
3. [ ] Boot Ubuntu SSD for tests B and C
4. [ ] Record all results
5. [ ] Raise PR dev → main for v2.0 release
6. [ ] Tag v2.0.0 on main
7. [ ] Publish to PyPI

## Pending (Future, post-v2.0)

- T-041: Fix QDQ insertion for dynamo-exported ONNX graphs (naming mismatch)
- T-038: Calibration-based conversion (result.convert with calibration_data)

## Blockers

None.

## Quick Links

- T-041 detail: docs/dev/tasks/T-041/DETAIL.md
- Tasks index: docs/dev/tasks/TASKS.md
- CHANGELOG: CHANGELOG.md
