# Implementation Log: T-021

## Summary

Added `validate_gptq_checkpoint_structure()` to `validators.py` for pure-Python
structural validation of a GPTQ checkpoint directory (no vLLM required). Wrote
`MANUAL_TEST.md` with exact steps for a live vLLM compatibility test on RTX 4050.

## What Was Done

- Modified `src/mono_quant/export/common/validators.py` — added
  `validate_gptq_checkpoint_structure(path)` function
- Created `docs/dev/tasks/T-021/MANUAL_TEST.md` — step-by-step vLLM test procedure

## How It Was Done

`validate_gptq_checkpoint_structure` is pure Python using only `pathlib`, `json`,
and `safetensors.safe_open` (already a core dep). Uses `safe_open` header inspection
to check for `.qweight` keys without loading all tensor data.

MANUAL_TEST.md covers: env setup, model prep with `facebook/opt-125m`, export via CLI,
structure validation, vLLM load + generate, success criteria, known limitations
(no Hessian-based ordering, `desc_act: false`, `config.json` note for HF models),
and a troubleshooting table.

## Files Changed

| File | Change Type | What Changed |
|------|-------------|--------------|
| `src/mono_quant/export/common/validators.py` | Modified | Added `validate_gptq_checkpoint_structure` |
| `docs/dev/tasks/T-021/MANUAL_TEST.md` | Created | vLLM manual test procedure |

## Testing Results

- Full suite: 53/53 passing (no regressions from validators.py change)
- Manual vLLM test: documented in MANUAL_TEST.md (requires GPU + vLLM install)

## Final State

**Status**: DONE | **Date**: 2026-02-26
