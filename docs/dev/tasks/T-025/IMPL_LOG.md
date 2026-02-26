# Implementation Log: T-025

## Summary
Added `validate_gguf_checkpoint(path)` to `validators.py`, added `gguf` optional dep
to `pyproject.toml`, and documented the manual llama.cpp validation procedure below.

## What Was Done
- `validate_gguf_checkpoint(path)` in `src/mono_quant/export/common/validators.py`:
  - FileNotFoundError on missing file
  - ImportError with install hint when gguf-py not installed
  - ValueError on gguf-py parse failure, empty tensor list, or missing `general.architecture` key
- `pyproject.toml`: added `gguf = ["gguf>=0.1"]` optional dependency group

## Files Changed

| File | Change Type | What Changed |
|------|-------------|--------------|
| `src/mono_quant/export/common/validators.py` | Modified | Added validate_gguf_checkpoint() |
| `pyproject.toml` | Modified | Added [gguf] optional dep |
| `tests/test_gguf_export.py` | Created | 3 T-025 tests |

## Testing Results
- Unit: 3 tests — all pass
  - valid checkpoint: ✓ (skipped without gguf-py)
  - FileNotFoundError: ✓
  - ImportError (mocked): ✓

## Manual llama.cpp Validation Procedure

Prerequisites: llama.cpp release binary or local build; a small model exported with
`monoquant export-gguf`.

```bash
# 1. Export
monoquant export-gguf -m ./model.pt -o ./gguf_out/ \
  --config ./config.json --architecture llama

# 2. Structural validation (automated)
python -c "
from mono_quant.export.common.validators import validate_gguf_checkpoint
validate_gguf_checkpoint('./gguf_out/model.gguf')
print('Structural validation passed')
"

# 3. llama.cpp load test
./llama-cli -m ./gguf_out/model.gguf -p "The sky is" -n 10 --no-mmap
```

Expected: coherent text continuation, no "invalid quantization format" errors.

Failure diagnosis:
- "unsupported tensor type" → check GGML_TYPE_Q4_K = 12 in quant_types.py
- "invalid quantization format" → recheck 6-bit scale packing in _pack_6bit_scales
- Garbage output → verify d/dmin/ls/lm encoding matches dequantize_row_q4_K
- Segfault → check 32-byte alignment in writer.py

This procedure has NOT been executed (no llama.cpp binary available in this environment).

## Final State
**Status**: DONE | **Commit**: see T-022–T-025 commit | **Date**: 2026-02-26
