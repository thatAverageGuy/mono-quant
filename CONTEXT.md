# CONTEXT

## Current State
**Task:** T-042 (GGUF arch_maps expansion) — DONE, pending commit approval
**Branch:** dev
**Last Commit:** dfabf65 (T-041)
**Date:** 2026-02-27

## Previous Session Summary
T-041 committed (dfabf65). Manual tests D and E written and passed. Manual test
procedures for B (GPTQ→vLLM) and C (GGUF→llama.cpp) rewritten for Ubuntu.
T-042 implemented: arch_maps.py expanded from 5 to 17 supported GGUF architectures.

## T-042 — What Was Implemented

Rewrote `src/mono_quant/export/gguf/arch_maps.py`:
- Added 12 new architectures: OPT, Phi-2, Phi-3/4, ChatGLM/GLM-4, Falcon,
  Gemma, Gemma2, StarCoder, StarCoder2, BLOOM, MPT, Command-R
- Added `mixtral → "llama"` alias
- Fixed `detect_architecture` bug: `"RefinedWebModel"/"RefinedWeb"` keys were
  mixed-case but lookup uses `.lower()` — fixed to `"refinedwebmodel"/"refinedweb"`
- All 117 tests pass

## Manual Test Results

| Test | Status | Notes |
|------|--------|-------|
| A2 (OPT-125m ONNX) | PASS | Dynamo + QDQ + full validation |
| D (CLI smoke) | PASS | 12/12 checks pass |
| E (result.convert()) | PASS | 10/10 checks pass |
| B (GPTQ → vLLM) | PENDING | Requires Ubuntu SSD + CUDA GPU |
| C (GGUF → llama.cpp) | PENDING | Requires Ubuntu SSD |

### Findings from Tests D/E (non-blocking for v2.0)

1. **BF-018 candidate**: `mq validate` crashes on Windows with cp1252 charmap when printing `✓`. Workaround: `PYTHONIOENCODING=utf-8` in subprocess env.
2. **Known limitation**: FP16 converted models cannot run a forward pass (dequantize() → fp32 weight, bias stays fp16 → matmul dtype mismatch). FP16 is storage-only.
3. **INT4 dynamic uses qint8**: `quantize(model, bits=4, dynamic=True)` creates `QuantizedLinear`, not `QuantizedLinearInt4`. Expected.
4. **mq quantize input format**: Requires full `nn.Module` (torch.save(model, path)), not a state_dict.

## Next Steps

1. [ ] Get user approval and commit T-042
2. [ ] (Optional) Fix BF-018: mq validate Windows charmap crash on ✓
3. [ ] Boot Ubuntu SSD for manual tests B and C (GPTQ → vLLM, GGUF → llama.cpp)
4. [ ] Record B/C results
5. [ ] Raise PR dev → main for v2.0 release
6. [ ] Tag v2.0.0 on main
7. [ ] Publish to PyPI

## Pending (Future, post-v2.0)

- T-038: Calibration-based conversion (result.convert with calibration_data)

## Blockers

None.

## Quick Links

- T-042 detail: docs/dev/tasks/T-042/DETAIL.md
- T-042 impl log: docs/dev/tasks/T-042/IMPL_LOG.md
- Tasks index: docs/dev/tasks/TASKS.md
- CHANGELOG: CHANGELOG.md
