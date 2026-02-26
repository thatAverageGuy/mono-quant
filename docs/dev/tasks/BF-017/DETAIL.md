# BF-017: Disable KV cache before ONNX export — DynamicCache blocks tracing

## Status
TODO

## Change Level
**LEVEL 1 — SURGICAL**
~5 lines in one file. No interface changes.

---

## Background

HuggingFace causal LMs (OPT, LLaMA, GPT-2, Mistral, Falcon, etc.) return a
`DynamicCache` object from `forward()` by default. PyTorch's tracing mechanisms
(both TorchScript and `torch.export`) require all outputs to be registered pytree
types. `DynamicCache` is not registered, so tracing fails with:

```
RuntimeError: Found <class 'transformers.cache_utils.DynamicCache'> in output,
which is not a known type. If this type holds tensors, you need to register a
pytree for it.
```

Standard fix: set `model.config.use_cache = False` before export. This makes the
model omit the KV cache from its output (`past_key_values=None`). Restore the
original value after tracing.

## Requirements

1. In `ONNXExporter.export()`, before `torch.onnx.export`, check if `fp32_model`
   has a `config` attribute with a `use_cache` field.
2. If so, save the original value and set `fp32_model.config.use_cache = False`.
3. Restore the original value in a `finally` block (covers both success and failure).
4. This must apply to BOTH the TorchScript and dynamo branches — `DynamicCache`
   blocks both paths equally.

## Decisions

- **Scoped to HF models only**: `hasattr(model, 'config') and hasattr(model.config, 'use_cache')`
  — plain `nn.Module` models have no `config`, so this is a no-op for them.
- **Restore in finally**: guarantees the model config is always restored, even if
  export fails, so the user's model is not left in a modified state.
- **Both paths**: `use_cache=False` is needed for both TorchScript and dynamo.
  The cache guard goes before the if/else branch.

## Implementation Guidance

In `ONNXExporter.export()`, just before Step 5 (torch.onnx.export block):

```python
# Temporarily disable KV cache for HuggingFace models.
# DynamicCache is not a registered pytree type and blocks both
# TorchScript and dynamo tracing.
_use_cache_orig = None
if hasattr(fp32_model, "config") and hasattr(fp32_model.config, "use_cache"):
    _use_cache_orig = fp32_model.config.use_cache
    fp32_model.config.use_cache = False

try:
    # ... existing TorchScript/dynamo branch ...
finally:
    if _use_cache_orig is not None:
        fp32_model.config.use_cache = _use_cache_orig
```

The existing `try/finally` block (which cleans up `tmp_path`) already wraps steps
5–9. The cache guard should be set BEFORE entering that block, and restored inside
the outer finally — OR the cache guard can wrap only the torch.onnx.export call
itself. Either is acceptable; wrapping the entire export block is simpler.

## Testing Requirements

- Unit: 1 new test in `tests/test_onnx_export.py`
  - `test_export_onnx_hf_model_use_cache_restored` — model with `config.use_cache=True`
    has it restored to `True` after export (even if export fails)
- Regression: all existing tests must pass

## Open Questions

*None.*
