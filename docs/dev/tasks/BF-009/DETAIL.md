# BF-009: Fix `dequantize_model` crash on qint8 buffers

## Status
TODO

## Audit Reference
C8 (Critical)

## Problem
In `core/quantizers.py`, `dequantize_model` converts quantized tensors back to
floating point using:

```python
buffer.data = buffer.data.to(torch.float32)
```

`torch.Tensor.to()` does **not** support casting from `torch.qint8` to
`torch.float32` directly. Calling `.to(torch.float32)` on a `qint8` tensor
raises:

```
RuntimeError: Could not run 'aten::empty_strided' with arguments from the 'QuantizedCPU' backend...
```

The correct method to convert a quantized tensor to float is `.dequantize()`,
which decodes stored integer values back to floating-point using the stored
scale and zero-point.

## Requirements
1. Replace `buffer.data.to(torch.float32)` with `buffer.data.dequantize()` for
   qint8/quint8 tensors.
2. Non-quantized buffers must still be handled (they don't have `.dequantize()`).
3. No crash on any model with quantized weight buffers.

## Decisions
- **Decision:** Check `buffer.data.is_quantized` before calling `.dequantize()`;
  fall back to `.to(torch.float32)` for non-quantized tensors.
  Reason: Defensive, handles mixed models cleanly.

## Success Criteria
- [ ] `dequantize_model` completes without RuntimeError on a quantized model
- [ ] Dequantized weights are floating-point values (not integers)
- [ ] Non-quantized buffers/parameters pass through unchanged
- [ ] Existing tests pass

## Dependencies
- None

## Implementation Guidance

In `src/mono_quant/core/quantizers.py`, in `dequantize_model`:

```python
# BEFORE
for name, buffer in model.named_buffers():
    if buffer.dtype in (torch.qint8, torch.quint8):
        buffer.data = buffer.data.to(torch.float32)   # CRASHES

# AFTER
for name, buffer in model.named_buffers():
    if buffer.data.is_quantized:
        buffer.data = buffer.data.dequantize()         # correct path
    elif buffer.dtype in (torch.qint8, torch.quint8):
        # fallback: shouldn't normally occur but guard it
        buffer.data = buffer.data.dequantize()
```

Also check `named_parameters()` loop for the same pattern if present.

## Testing Requirements
- Unit: Create a model with a `qint8` parameter; call `dequantize_model`;
  verify no RuntimeError and output dtype is float32.
- Unit: Verify model with no quantized tensors passes through unchanged.
- Coverage target: both quantized and non-quantized branches

## Open Questions
<!-- MUST be empty before implementation begins -->
