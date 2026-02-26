# Implementation Log: BF-016

## Summary
Fixed `validate_onnx_model` full-level validation to infer the correct numpy dtype
from the ONNX graph's input spec instead of hardcoding `np.float32`. Transformer
models export with `int64` inputs; the old hardcode caused onnxruntime to reject
the dummy input with a type mismatch error even on valid models.

---

## What Was Done

In `validate_onnx_model` (ValidationLevel.FULL branch), replaced:
```python
dummy = np.zeros(concrete_shape, dtype=np.float32)
```
with a dtype lookup from `inputs[0].type`:
```python
_ONNX_TO_NP = { "tensor(float)": np.float32, "tensor(int64)": np.int64, ... }
input_dtype = _ONNX_TO_NP.get(inputs[0].type, np.float32)
dummy = np.zeros(concrete_shape, dtype=input_dtype)
```

One new test: `test_validate_onnx_full_int64_model` — exports an Embedding+Linear
model via dynamo=True, then runs full validation; must not raise.

---

## Files Changed

| File | Change Type | What Changed |
|------|-------------|--------------|
| `src/mono_quant/export/common/validators.py` | Modified | dtype inferred from ONNX input spec; fallback to float32 |
| `tests/test_export_validation.py` | Modified | 1 new BF-016 test |
| `docs/dev/tasks/BF-016/DETAIL.md` | Created | Task planning document |
| `docs/dev/tasks/BF-016/IMPL_LOG.md` | Created | This file |

---

## Why These Choices Were Made

- **Lookup table over onnxruntime API**: The type string format (`"tensor(float)"` etc.)
  is stable and well-documented in the ONNX spec. A local dict is simpler and has no
  additional dependencies.
- **Fallback to float32**: Preserves existing behaviour for any ONNX type not in the
  table. Exotic types (bfloat16, complex) fall back gracefully.

---

## Testing Results

- Unit: 1/1 new test passing
- Regression: all prior 112 tests passing
- Total: **113 passed, 9 skipped, 0 failures**

---

## Final State
**Status**: DONE | **Date**: 2026-02-27
