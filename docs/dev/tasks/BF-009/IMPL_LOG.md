# Implementation Log: BF-009

## Summary
Fixed `dequantize_model` crashing on models with `qint8`/`quint8` buffers.
The original code used `buffer.data = buffer.data.to(torch.float32)` which
fails on quantized tensors. Also fixed the buffer replacement approach: plain
tensor `.data` assignment cannot change dtype, so `register_buffer` is required.

## What Was Done
- Replaced the buffer loop in `dequantize_model` with a `register_buffer`-based
  approach that correctly handles dtype-changing replacements
- Used `buffer.is_quantized` check (cleaner than dtype comparison) and
  `buffer.dequantize()` instead of `buffer.to(torch.float32)`
- Added test verifying no RuntimeError on qint8 buffer
- Added test verifying non-quantized buffers pass through unchanged

## How It Was Done
Two-part fix:
1. `buffer.is_quantized` → `buffer.dequantize()` (correct method for quantized tensors)
2. `register_buffer(name, new_tensor)` on the parent module instead of `buffer.data =`
   (required because plain tensor `.data` assignment enforces type compatibility)

The parent module is retrieved via `model.get_submodule(parent_name)` using the
dotted name from `named_buffers()`.

Before:
```python
for name, buffer in model.named_buffers():
    if buffer.dtype == torch.qint8 or buffer.dtype == torch.float16:
        buffer.data = buffer.data.to(torch.float32)   # CRASHES on qint8
```
After:
```python
for name, buffer in list(model.named_buffers()):
    if buffer is None:
        continue
    if buffer.is_quantized:
        new_buf = buffer.dequantize()
    elif buffer.dtype == torch.float16:
        new_buf = buffer.to(torch.float32)
    else:
        continue
    parent_name, _, attr_name = name.rpartition(".")
    parent = model if not parent_name else model.get_submodule(parent_name)
    parent.register_buffer(attr_name, new_buf)
```

## Files Changed

| File | Change Type | What Changed |
|------|-------------|--------------|
| src/mono_quant/core/quantizers.py | Modified | Buffer loop in `dequantize_model` rewritten |
| tests/test_bugfixes.py | Modified | Two `dequantize_model` buffer tests added |

## Testing Results
- Unit: `test_dequantize_model_qint8_buffer_no_crash` — no RuntimeError, dtype float32: PASS
- Unit: `test_dequantize_model_non_quantized_passthrough` — float32 buffers unchanged: PASS
- All 15 tests passing

## Final State
**Status**: DONE | **Date**: 2026-02-25
