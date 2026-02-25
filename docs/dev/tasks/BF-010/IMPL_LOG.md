# Implementation Log: BF-010

## Summary
Fixed `quantize_embedding_module` silently ignoring the `dtype` parameter.
The `dtype` argument was accepted but never passed to `QuantizedEmbedding.from_embedding`,
which always called `quantize_weight_int8` regardless. Added `dtype` to the
call chain and implemented the FP16 path.

## What Was Done
- Added `dtype` parameter to `QuantizedEmbedding.from_embedding` class method
- In `from_embedding`: if `dtype == torch.float16`, store `module.weight.data.half()`
  directly; otherwise use existing `quantize_weight_int8` path
- Updated `from_embedding` to pass `dtype=dtype` when constructing `QuantizedEmbedding`
  (stores the dtype on the instance for `extra_repr`)
- Updated `forward()` and `weight` property to handle FP16 path:
  `buffer.is_quantized` dispatches to `dequantize_weight` for qint8,
  or `.float()` cast for FP16
- Updated `quantize_embedding_module` to pass `dtype=dtype` to `from_embedding`
- Added two tests: qint8 path gives quantized weight; fp16 path gives float16 weight

## How It Was Done
LEVEL 1 surgical edit across `modules/embedding.py`. The FP16 path is a simple
`.half()` cast — embeddings stored as FP16 reduce memory by 2× vs FP32 without
the overhead of INT8 quantization/dequantization. The `is_quantized` property
on tensors cleanly distinguishes qint8 from float16 at runtime.

## Files Changed

| File | Change Type | What Changed |
|------|-------------|--------------|
| src/mono_quant/modules/embedding.py | Modified | `from_embedding` gains `dtype` param; FP16 branch; `forward`/`weight` handle non-quantized path |
| src/mono_quant/modules/embedding.py | Modified | `quantize_embedding_module` passes `dtype` to `from_embedding` |
| tests/test_bugfixes.py | Modified | Two new dtype tests |

## Testing Results
- Unit: `test_embedding_quantize_int8_dtype` — PASS (weight.is_quantized == True)
- Unit: `test_embedding_quantize_fp16_dtype` — PASS (weight.dtype == float16)
- All 28 tests passing

## Final State
**Status**: DONE | **Date**: 2026-02-25
