# BF-016: Fix validate_onnx_model full-level hardcodes float32 dummy

## Status
TODO

## Change Level
**LEVEL 1 — SURGICAL**
One change site in one file. Map ONNX input type string to numpy dtype.

---

## Background

`validate_onnx_model` at `ValidationLevel.FULL` runs an onnxruntime inference pass
to verify the exported graph executes without error. It builds a dummy input as:

```python
dummy = np.zeros(concrete_shape, dtype=np.float32)
```

This hardcoded `float32` works for MLP/Conv models but fails for transformer models
whose ONNX graph expects `int64` (token ID) inputs — onnxruntime raises a type mismatch
error even though the model is valid.

The ONNX session's `inputs[0].type` carries the actual dtype as a string
(e.g., `"tensor(float)"`, `"tensor(int64)"`). Reading it and mapping to a numpy dtype
fixes the validation for all standard input types.

---

## Requirements

1. `validate_onnx_model` reads `inputs[0].type` from the onnxruntime session.
2. Maps the ONNX type string to a numpy dtype using a local lookup table.
3. Falls back to `np.float32` for any unrecognised type (safe default).
4. The dummy input passed to `session.run()` uses the inferred dtype.

---

## Decisions

- **Local lookup table** — no new dependencies; onnxruntime's type strings are stable.
- **Fallback to float32** — for exotic/unrecognised types, existing behaviour is preserved.
- **Single input only** — `validate_onnx_model` already only handles `inputs[0]`.
  Multi-input models are out of scope for this validator.

---

## Implementation Guidance

In `src/mono_quant/export/common/validators.py`, inside `validate_onnx_model`,
replace the hardcoded dtype with an inferred one:

```python
_ONNX_TO_NP = {
    "tensor(float)":   np.float32,
    "tensor(float16)": np.float16,
    "tensor(double)":  np.float64,
    "tensor(int64)":   np.int64,
    "tensor(int32)":   np.int32,
    "tensor(int16)":   np.int16,
    "tensor(int8)":    np.int8,
    "tensor(uint8)":   np.uint8,
    "tensor(bool)":    np.bool_,
}

session = ort.InferenceSession(str(path))
inputs = session.get_inputs()
input_shape = inputs[0].shape
concrete_shape = [d if isinstance(d, int) and d > 0 else 1 for d in input_shape]
input_dtype = _ONNX_TO_NP.get(inputs[0].type, np.float32)
dummy = np.zeros(concrete_shape, dtype=input_dtype)
session.run(None, {inputs[0].name: dummy})
```

---

## Testing Requirements

- Unit: 1 new test in `tests/test_export_validation.py`
  - `test_validate_onnx_full_int64_model` — exports an Embedding model with dynamo=True,
    then validates with level="full"; must not raise
- Regression: existing `test_export_onnx_validate_load` must still pass

---

## Open Questions

*None.*

---

## Dependencies

- T-040 (done — dynamo=True path is needed to export transformer models that have int64 inputs)
