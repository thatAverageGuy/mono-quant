# Implementation Log: T-018

## Summary

Implemented GPTQ INT4 packing math (AutoGPTQ V1 format) and the `GPTQExporter`
class. This is the core quantization kernel that all GPTQ export builds on.

## What Was Done

- Created `src/mono_quant/export/common/gptq_packing.py` with four functions:
  `quantize_to_gptq_int4`, `_pack_int4_rows_to_int32`, `_pack_int4_cols_to_int32`,
  `unpack_gptq_weight`.
- Created `src/mono_quant/export/gptq.py` with `GPTQExporter(BaseExporter)`.

## How It Was Done

**Bit packing:** AutoGPTQ V1 packs 8 INT4 values (one per 4 bits) into one INT32,
LSB-first. `_pack_int4_rows_to_int32` handles qweight (pack 8 rows → 1 output row).
`_pack_int4_cols_to_int32` handles qzeros (pack 8 cols → 1 output col). Both are
fully vectorized via reshape + shift/sum — no Python loops.

**Dtype fix:** `torch.arange(..., dtype=torch.int32)` still produces int64 on Windows
after shift/sum. Explicit `.to(torch.int32)` at the return site fixes this.

**Contiguity:** `permute()` produces non-contiguous views which `safetensors` rejects.
All tensors in the checkpoint are made contiguous before `save_file()`.

**qzeros V1 encoding:** AutoGPTQ stores `zero - 1` before packing. Clamped to [0, 15]
after subtraction to prevent underflow.

## Files Changed

| File | Change Type | What Changed |
|------|-------------|--------------|
| `src/mono_quant/export/common/gptq_packing.py` | Created | Packing math |
| `src/mono_quant/export/gptq.py` | Created | GPTQExporter class |

## Why These Choices Were Made

- Vectorized packing: avoids Python loops over in_features (can be 4096+)
- `inplace=False` on `revert_to_standard_modules`: preserve original model
- V1 format (not V2): vLLM and AutoGPTQ default to V1; V2 requires additional metadata

## Testing Results

- Unit: covered by T-019 tests (shapes, dtypes, round-trip)
- Integration: covered by T-019 export tests

## Final State

**Status**: DONE | **Date**: 2026-02-26
