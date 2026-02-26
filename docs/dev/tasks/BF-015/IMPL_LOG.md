# Implementation Log: BF-015

## Summary
Fixed `_quantize_int8_model` and `_quantize_sequential_module` to use exact-type
matching (`type(module) is nn.Embedding`) instead of `isinstance(module, nn.Embedding)`
when selecting embedding layers for quantization. Prevents custom `nn.Embedding`
subclasses (positional encoding modules common across transformer families) from being
quantized and subsequently having their custom `forward()` signatures destroyed.

---

## What Was Done

Replaced three `isinstance(module, nn.Embedding)` guards with `type(module) is nn.Embedding`
in `src/mono_quant/core/quantizers.py`:
1. Named-children loop in `_quantize_int8_model` (~line 937)
2. Named-modules loop in `_quantize_int8_model` (~line 981)
3. `_quantize_sequential_module` (~line 1030)

Added two tests to `tests/test_bugfixes.py`:
1. `test_quantize_int8_exact_embedding_is_quantized` — plain `nn.Embedding` IS quantized
2. `test_quantize_int8_embedding_subclass_is_skipped` — subclass is skipped

---

## How It Was Done

Single mechanical substitution: `isinstance(module, nn.Embedding)` → `type(module) is nn.Embedding`.

The `# NEW` stale comments at two of the three sites were also cleaned up.

No changes to:
- `_infer_dummy_input` in `onnx.py` — that function uses `isinstance` intentionally
  to detect whether ANY embedding-bearing layer exists (for LongTensor dummy input inference).
  Detecting subclasses there is correct.
- `revert_to_standard_modules` — only runs on `QuantizedEmbedding` instances, which
  after this fix can only originate from exact `nn.Embedding` types. No behaviour change.
- Any INT4 path — INT4 quantization does not operate on `nn.Embedding`.

---

## Files Changed

| File | Change Type | What Changed |
|------|-------------|--------------|
| `src/mono_quant/core/quantizers.py` | Modified | 3 isinstance → type-is replacements; 2 stale comments removed |
| `tests/test_bugfixes.py` | Modified | 2 new BF-015 tests |
| `docs/dev/tasks/BF-015/DETAIL.md` | Created | Task planning document |
| `docs/dev/tasks/BF-015/IMPL_LOG.md` | Created | This file |

---

## Why These Choices Were Made

- **`type(module) is nn.Embedding`**: Subclasses of `nn.Embedding` are architectural
  components (positional encodings, rotary embeddings) whose `forward()` signatures
  are NOT compatible with plain `nn.Embedding.forward(input)`. Quantizing them and
  reverting them to `nn.Embedding` silently destroys their custom behavior, causing
  `TypeError` at tracing time. Exact-type match is the principled fix.

- **Generalised, not OPT-specific**: The fix applies to ANY `nn.Embedding` subclass
  across all transformer families — OPT (`OPTLearnedPositionalEmbedding`), GPT-2
  (positional embedding subclasses), Llama/Mistral/Falcon variants. The test models a
  generic `LearnedPositionalEmbedding` pattern, not OPT specifically.

- **Token embeddings still quantized**: Plain `nn.Embedding` (token ID → vector lookup
  tables) remains quantizable. Only positional/rotary subclasses are excluded.

---

## Testing Results

- Unit: 2/2 new tests passing
- All prior: 107/107 passing (no regressions)
- Total: **109 passed, 9 skipped, 0 failures**
- Coverage: both branches of the new exact-type check exercised

---

## Impact

- `_quantize_int8_model` no longer quantizes `nn.Embedding` subclasses that carry
  custom `forward()` signatures (positional encodings across all transformer families)
- BF-014's graceful `TypeError` catch remains in place as a safety net for any other
  complex forward signatures not covered by this fix
- Quantization of plain token embedding tables (`nn.Embedding`) is unaffected

---

## Final State
**Status**: DONE | **Date**: 2026-02-27
