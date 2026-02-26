# BF-014: Three ONNX / dynamic-quantization bugs found in manual test A2

## Status
TODO

## Change Level
**LEVEL 2 — LOCAL**
Three surgical fixes across three files. No new abstractions, no public interface changes.

---

## Background

Found during manual test A2 (OPT-125m ONNX export, 2026-02-26).
Root-caused by diagnostic script `mq_manual_test/diagnose.py`.

All three bugs are independent. They compound to produce the observed
output:
1. Misleading "Layers quantized: 0" (reporting gap)
2. False "Model has no quantized parameters" warning (validator logic error)
3. Raw `RuntimeError` crash during ONNX tracing (unhandled exception)

---

## Bug 1 — Misleading "Layers quantized: 0" (dynamic path)

**File:** `src/mono_quant/api/quantize.py:205`

**Root cause:**
```python
info = QuantizationInfo(
    selected_layers=[],  # Dynamic doesn't track selected layers
    ...
)
```
The dynamic quantization path hardcodes `selected_layers=[]`. In reality,
`_quantize_int8_model` creates `QuantizedLinear` / `QuantizedConv2d` /
`QuantizedEmbedding` modules throughout the model tree. For OPT-125m, 73
layers are quantized but the info object reports 0.

**Fix:**
After `dynamic_quantize()` returns `q_model`, introspect it for
quantized module types and collect their names into `selected_layers`.
Do not modify `_quantize_int8_model` internals — gather from the
returned model in `quantize.py` only.

```python
from mono_quant.modules.linear import QuantizedLinear, QuantizedConv2d, QuantizedLinearInt4
from mono_quant.modules.embedding import QuantizedEmbedding

QUANTIZED_TYPES = (QuantizedLinear, QuantizedConv2d, QuantizedLinearInt4, QuantizedEmbedding)

selected_layers = [
    name for name, m in q_model.named_modules()
    if isinstance(m, QUANTIZED_TYPES)
]
```

---

## Bug 2 — False "Model has no quantized parameters" warning

**File:** `src/mono_quant/export/common/validators.py:52-63`

**Root cause:**
```python
has_quantized = any(
    p.dtype in {torch.qint8, torch.quint8, torch.float16}
    for p in model.parameters()
)
```
`QuantizedLinear` stores the quantized weight as `_quantized_weight`
(a plain tensor attribute, **not** `nn.Parameter`). The only parameter
is `bias` (float32). So `model.parameters()` never yields a quantized
dtype → `has_quantized` is always `False` → warning always fires.

**Fix:**
Check for quantized module instances directly, not parameter dtypes.

```python
from mono_quant.modules.linear import QuantizedLinear, QuantizedConv2d, QuantizedLinearInt4
from mono_quant.modules.embedding import QuantizedEmbedding

QUANTIZED_TYPES = (QuantizedLinear, QuantizedConv2d, QuantizedLinearInt4, QuantizedEmbedding)

has_quantized = any(isinstance(m, QUANTIZED_TYPES) for m in model.modules())
```

Note: this check runs on the model **before** `revert_to_standard_modules`
is called (pre-validation), so the `QuantizedLinear` modules are still
present at check time.

---

## Bug 3 — Raw RuntimeError crash in ONNX tracing for LLMs

**File:** `src/mono_quant/export/onnx.py:153-163` (`_infer_dummy_input`)

**Root cause:**
```python
def _infer_dummy_input(self, model: nn.Module) -> torch.Tensor:
    for module in model.modules():
        if isinstance(module, nn.Linear):
            return torch.zeros(1, module.in_features)  # always float32
        ...
```
For OPT-125m, this finds the first `nn.Linear` (e.g., `in_features=768`)
and returns `zeros(1, 768)` as a float32 tensor. OPT's `forward()` expects
`input_ids` — a LongTensor of token indices. The float tensor reaches
`F.embedding()` → `RuntimeError: Expected ... Long, Int; but got FloatTensor`.

The CONTEXT.md stated this test should produce a "graceful tracing error".
The fix is to catch the tracing error in `ONNXExporter.export` and re-raise
as a clear, user-facing message.

**Fix (two parts):**

Part A — `_infer_dummy_input`: detect if the model's forward likely
requires integer input by checking for `nn.Embedding` layers. If the model
has an `nn.Embedding` before any `nn.Linear` in the module tree, return
a LongTensor dummy instead:

```python
def _infer_dummy_input(self, model: nn.Module) -> torch.Tensor:
    for module in model.modules():
        if isinstance(module, nn.Embedding):
            # LLM-style: input is token IDs (LongTensor)
            return torch.zeros(1, 16, dtype=torch.long)
        if isinstance(module, nn.Linear):
            return torch.zeros(1, module.in_features)
        if isinstance(module, nn.Conv2d):
            return torch.zeros(1, module.in_channels, 32, 32)
    raise RuntimeError(...)
```

Part B — Catch tracing errors in `export()` and surface them cleanly.
Wrap the `torch.onnx.export(...)` call to catch `RuntimeError` and
re-raise with a message that tells the user to provide `dummy_input`
explicitly:

```python
try:
    torch.onnx.export(...)
except RuntimeError as e:
    raise RuntimeError(
        f"ONNX tracing failed: {e}\n\n"
        "This model likely requires a non-standard dummy input "
        "(e.g., LongTensor for token IDs). "
        "Pass dummy_input explicitly: result.export('out.onnx', dummy_input=...)"
    ) from e
```

---

## Requirements

1. `result.info.selected_layers` must be non-empty after `quantize(model, dynamic=True)` on any model that has quantizable layers.
2. The "Model has no quantized parameters" warning must NOT fire when the model contains `QuantizedLinear` / `QuantizedConv2d` / `QuantizedEmbedding` modules.
3. For models that require non-float dummy input, `_infer_dummy_input` must return a LongTensor when an `nn.Embedding` layer is detected first.
4. When ONNX tracing fails at runtime, the error message must be user-actionable (tell them to pass `dummy_input` explicitly) — not a raw PyTorch internal traceback.

---

## Decisions

- **Bug 1 fix location:** `quantize.py` only (post-hoc introspection of `q_model`). Avoids changing `_quantize_int8_model` return type, which would require updating all callers.
- **Bug 2 fix:** Instance check > dtype check. More robust — works regardless of how the quantized weight is stored.
- **Bug 3 Part A:** Embedding-first heuristic (`seq_len=16`) is a reasonable default for the graceful path; if the model still fails to trace, Part B catches it cleanly.
- **Bug 3 Part B:** Catch at the `torch.onnx.export` call site only, not at the top level. Keeps error context precise.

---

## Success Criteria

- [ ] `result.info.selected_layers` is non-empty after `quantize(opt_model, bits=8, dynamic=True)`
- [ ] Running `test_a_onnx.py` (OPT-125m) no longer emits "Model has no quantized parameters" warning
- [ ] Running `test_a_onnx.py` no longer crashes with a raw `RuntimeError` traceback; instead prints a clear user-facing error about `dummy_input`
- [ ] Running `test_a_onnx_simple.py` (simple MLP) still passes — no regression
- [ ] All 103 existing tests still pass
- [ ] New tests added for each fix

---

## Dependencies

None. All three fixes are self-contained.

---

## Implementation Guidance

### Step 1 — Fix Bug 1 (`quantize.py`)

File: `src/mono_quant/api/quantize.py`

After the call to `dynamic_quantize()` succeeds, add introspection
before building `QuantizationInfo`:

```python
# After: q_model, skipped = dynamic_quantize(...)

from mono_quant.modules.linear import (
    QuantizedLinear, QuantizedConv2d, QuantizedLinearInt4
)
from mono_quant.modules.embedding import QuantizedEmbedding

_QUANTIZED_TYPES = (QuantizedLinear, QuantizedConv2d, QuantizedLinearInt4, QuantizedEmbedding)
selected_layers = [
    name for name, m in q_model.named_modules() if isinstance(m, _QUANTIZED_TYPES)
]

info = QuantizationInfo(
    selected_layers=selected_layers,   # was: []
    skipped_layers=skipped,
    ...
)
```

### Step 2 — Fix Bug 2 (`validators.py`)

File: `src/mono_quant/export/common/validators.py`

Replace the `has_quantized` check (lines 52-57) with an instance check:

```python
from mono_quant.modules.linear import QuantizedLinear, QuantizedConv2d, QuantizedLinearInt4
from mono_quant.modules.embedding import QuantizedEmbedding

_QUANTIZED_TYPES = (QuantizedLinear, QuantizedConv2d, QuantizedLinearInt4, QuantizedEmbedding)
has_quantized = any(isinstance(m, _QUANTIZED_TYPES) for m in model.modules())
```

### Step 3 — Fix Bug 3 (`onnx.py`)

File: `src/mono_quant/export/onnx.py`

**Part A** — Update `_infer_dummy_input` to check for `nn.Embedding` first:

```python
def _infer_dummy_input(self, model: nn.Module) -> torch.Tensor:
    for module in model.modules():
        if isinstance(module, nn.Embedding):
            return torch.zeros(1, 16, dtype=torch.long)
        if isinstance(module, nn.Linear):
            return torch.zeros(1, module.in_features)
        if isinstance(module, nn.Conv2d):
            return torch.zeros(1, module.in_channels, 32, 32)
    raise RuntimeError(
        "Cannot infer dummy input: no Linear, Conv2d, or Embedding layer found. "
        "Please provide dummy_input explicitly."
    )
```

**Part B** — Wrap `torch.onnx.export` call (around line 104) to catch tracing failures:

```python
try:
    torch.onnx.export(...)
except RuntimeError as e:
    raise RuntimeError(
        f"ONNX tracing failed: {e}\n\n"
        "Hint: this model likely requires a non-standard input format "
        "(e.g., integer token IDs for transformer models). "
        "Pass dummy_input explicitly: result.export('out.onnx', dummy_input=torch.zeros(1, 16, dtype=torch.long))"
    ) from e
```

### Step 4 — Tests

Add to `tests/unit/test_onnx_export.py` (or create `tests/unit/test_bf014.py` if
the existing file is large enough to warrant separation):

1. `test_dynamic_quantize_selected_layers_populated` — verifies `selected_layers` is non-empty after dynamic INT8 on a nested model
2. `test_validate_export_pre_no_false_positive` — QuantizedLinear model does NOT trigger "no quantized parameters" warning
3. `test_infer_dummy_input_embedding_model_returns_long` — `_infer_dummy_input` returns LongTensor when model has Embedding
4. `test_onnx_export_tracing_error_is_graceful` — confirm RuntimeError is caught and re-raised with user-friendly message (using a model whose forward requires long input but dummy_input not provided)

### Step 5 — Run full test suite

```
pytest tests/ -v
```

Must pass: 103 existing + 4 new = 107 tests minimum.

---

## Testing Requirements

- Unit: 4 new tests (see Step 4)
- Integration: `test_a_onnx_simple.py` must still pass (regression check)
- Coverage target: all three fix sites + all four new test cases

---

## Open Questions

*None — all root causes confirmed by diagnostic script.*
