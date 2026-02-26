# Implementation Log: BF-014

## Summary
Fixed three bugs found during manual test A2 (OPT-125m ONNX export).
All three were independent root causes that compounded to produce misleading
output and a raw crash. Quantization was actually working correctly throughout.

---

## What Was Done

1. **Bug 1** — `quantize()` now populates `selected_layers` for dynamic INT8
   quantization by introspecting the returned model for quantized module types.

2. **Bug 2** — Pre-export validator now detects quantized models via module
   instance checks instead of parameter dtype checks. `QuantizedLinear` stores
   its quantized weight as a plain tensor attribute, not `nn.Parameter`, so the
   old dtype check always returned False.

3. **Bug 3** — `_infer_dummy_input` now checks for `nn.Embedding` before
   `nn.Linear`, returning a LongTensor for LLM-style models. ONNX tracing
   errors are now caught and re-raised with a user-actionable message.

---

## How It Was Done

### Bug 1 (`api/quantize.py`)
Post-hoc introspection of the returned `q_model` after `dynamic_quantize()`:
```python
selected_layers = [
    name for name, m in q_model.named_modules()
    if isinstance(m, _QUANTIZED_TYPES)
]
```
No changes to `_quantize_int8_model` internals or its return type.

### Bug 2 (`export/common/validators.py`)
Combined check — instance check for INT8/INT4 (wrapper modules) plus dtype
check for FP16 (parameter cast in-place, no wrapper):
```python
has_quantized = (
    any(isinstance(m, _QUANTIZED_TYPES) for m in model.modules())
    or any(p.dtype == torch.float16 for p in model.parameters())
)
```
The FP16 dtype check was retained to avoid breaking the existing
`test_validate_pre_fp16_model_no_warning` test, which is correct behaviour
(FP16-cast models are legitimately quantized).

### Bug 3 (`export/onnx.py`)
Part A — `_infer_dummy_input` checks `nn.Embedding` before `nn.Linear`:
```python
if isinstance(module, nn.Embedding):
    return torch.zeros(1, 16, dtype=torch.long)
```
Part B — `torch.onnx.export` wrapped in `try/except RuntimeError`:
```python
except RuntimeError as e:
    raise RuntimeError(
        f"ONNX tracing failed: {e}\n\n"
        "Hint: ... Pass dummy_input explicitly: ..."
    ) from e
```

---

## Files Changed

| File | Change Type | What Changed |
|------|-------------|--------------|
| `src/mono_quant/api/quantize.py` | Modified | Post-hoc selected_layers population for dynamic path |
| `src/mono_quant/export/common/validators.py` | Modified | Instance-based has_quantized check |
| `src/mono_quant/export/onnx.py` | Modified | Embedding-first dummy input + tracing error catch |
| `tests/test_api.py` | Modified | 1 new test: BF-014/Bug1 selected_layers populated |
| `tests/test_export_validation.py` | Modified | 1 new test: BF-014/Bug2 no false positive on QuantizedLinear |
| `tests/test_onnx_export.py` | Modified | 2 new tests: BF-014/Bug3a (LongTensor), Bug3b (graceful error) |
| `docs/dev/tasks/BF-014/DETAIL.md` | Created | Task planning document |
| `docs/dev/tasks/BF-014/IMPL_LOG.md` | Created | This file |

---

## Why These Choices Were Made

- **Bug 1: post-hoc introspection** — avoids changing `_quantize_int8_model`
  return type, which would ripple through all callers. The returned model is
  authoritative; checking it directly is simpler and more correct.

- **Bug 2: combined check** — FP16 models use in-place parameter casting (no
  wrapper modules), so the dtype check must be retained for that path. The
  instance check handles INT8/INT4. Both are needed.

- **Bug 3: Embedding-first heuristic** — `seq_len=16` is arbitrary but
  sufficient for the tracing pass. The user can always override with
  `dummy_input` for unusual architectures. The error message tells them how.

---

## Testing Results

- Unit: 4/4 new tests passing
- All prior: 103/103 passing (no regressions)
- Total: **107 passed, 9 skipped, 0 failures**
- Coverage: all three fix sites exercised by dedicated tests

---

## Issues Encountered

- **FP16 regression risk**: Initial validator fix broke `test_validate_pre_fp16_model_no_warning`
  because FP16 models don't use `QuantizedLinear` wrappers. Resolved by
  combining the instance check with a retained `float16` parameter dtype check.

- **Bug 3 catch too narrow**: Initial `except RuntimeError` didn't cover `TypeError`.
  When re-testing against real OPT-125m, `revert_to_standard_modules` converts
  `QuantizedEmbedding` back to plain `nn.Embedding`. OPT's `OPTLearnedPositionalEmbedding`
  then calls `embed_positions(attention_mask, ..., position_ids=position_ids)` — which
  hits the reverted `nn.Embedding.forward()` with an unexpected kwarg, raising `TypeError`,
  not `RuntimeError`. Resolved by widening catch to `(RuntimeError, TypeError)` and
  adding a second test case covering the `TypeError` path.

---

## Impact

- `result.info.selected_layers` is now accurate for dynamic quantization
- Pre-export validator no longer fires false warnings on correctly quantized models
- ONNX export of LLM-style models no longer crashes raw; fails gracefully with hint

---

## Final State
**Status**: DONE | **Date**: 2026-02-26
**Note**: Gap found during re-test (TypeError not caught) — fixed before commit.
