# BF-015: Fix nn.Embedding subclass quantization — use exact-type match

## Status
TODO

## Change Level
**LEVEL 1 — SURGICAL**
Three one-line edits in a single file. No new files. No interface changes.

---

## Background

`_quantize_int8_model` and `_quantize_sequential_module` use
`isinstance(module, nn.Embedding)` to detect embedding layers for quantization.
`isinstance` matches the class AND all subclasses.

Transformer models universally subclass `nn.Embedding` for positional encodings:
- HuggingFace OPT: `OPTLearnedPositionalEmbedding(nn.Embedding)` — custom `forward(attention_mask, past_seen_tokens, position_ids=...)`
- HuggingFace Llama: `LlamaRotaryEmbedding` (sometimes `nn.Embedding`-based)
- GPT-2: `nn.Embedding` subclass for position embeddings with custom forward
- Falcon, Mistral, similar pattern

When a subclass like `OPTLearnedPositionalEmbedding` is quantized:
1. Its weight is wrapped in `QuantizedEmbedding`
2. `revert_to_standard_modules` later reverts it back to a PLAIN `nn.Embedding`
3. The reverted module loses the custom `forward()` signature
4. ONNX tracing then calls `nn.Embedding.forward(attention_mask, past_seen_tokens, position_ids=...)`
   which raises `TypeError: Embedding.forward() got unexpected keyword argument 'position_ids'`

BF-014 caught the resulting `TypeError` and converted it to a graceful error.
BF-015 fixes the root cause: positional encoding subclasses should never be quantized.

---

## Requirements

1. `isinstance(module, nn.Embedding)` at all three quantization sites changed to
   `type(module) is nn.Embedding`.
2. Only exact `nn.Embedding` instances (token embedding tables) are quantized.
3. `nn.Embedding` subclasses (positional encodings, rotary embeddings, etc.)
   fall through to the `else` branch and are skipped.
4. No changes to `_infer_dummy_input` — that function uses `isinstance` intentionally
   to detect embedding-model input type (LongTensor needed for ANY embedding-bearing model).
5. No changes to `revert_to_standard_modules` — it only runs on `QuantizedEmbedding`
   instances, which can only be created from exact `nn.Embedding` instances after this fix.

---

## Decisions

- **`type(module) is nn.Embedding` not `isinstance`**: Exact-type match is the correct
  guard here. Subclasses are architectural components (positional encodings, rotary
  embeddings) that should NOT be quantized. Their weight tables are either learned
  fixed positions or mathematical constructs — quantizing them corrupts model behaviour
  and destroys the custom `forward()` after revert.
- **All three sites changed**: Consistency — all quantization paths must use the same
  check. Changing only one site would create inconsistent behavior depending on model
  structure (flat vs nested vs Sequential).
- **`_infer_dummy_input` left unchanged**: That function uses `isinstance` to determine
  the correct dummy tensor dtype (LongTensor for any model containing an embedding-like
  layer). Detecting subclasses there is CORRECT behaviour.

---

## Implementation Guidance

### Step 1 — `quantizers.py` site 1 (named_children loop, line ~937)

```python
# Before:
elif isinstance(module, nn.Embedding):  # NEW
# After:
elif type(module) is nn.Embedding:
```

Remove the `# NEW` comment while at it (it's stale).

### Step 2 — `quantizers.py` site 2 (named_modules loop, line ~981)

```python
# Before:
elif isinstance(module, nn.Embedding):
# After:
elif type(module) is nn.Embedding:
```

### Step 3 — `quantizers.py` site 3 (`_quantize_sequential_module`, line ~1030)

```python
# Before:
elif isinstance(module, nn.Embedding):  # NEW
# After:
elif type(module) is nn.Embedding:
```

Remove the `# NEW` comment here too.

---

## Testing Requirements

- Unit: 2 new tests in `tests/test_quantizers.py` (or nearest appropriate file)
  1. `test_quantize_int8_exact_embedding_is_quantized` — plain `nn.Embedding` IS quantized
  2. `test_quantize_int8_embedding_subclass_is_skipped` — `nn.Embedding` subclass is NOT quantized
- Regression: all existing tests must pass (especially embedding-related tests)
- Coverage: both branches of the new exact-type check exercised

---

## Open Questions

*None.*

---

## Dependencies

- BF-014 (done — provides graceful error handling that catches the downstream TypeError
  this fix prevents at the root)
