# Implementation Log: BF-008

## Summary
Fixed `_quantize_int8_model`'s nested layer detection in `core/quantizers.py`.
The second `named_modules()` loop had a dead-code guard — `not isinstance(module,
type(model_copy.get_submodule(name)))` — that is always `False` because
`get_submodule(name)` returns the same object as `module`. Removed the guard.

## What Was Done
- Removed the broken `and not isinstance(module, type(model_copy.get_submodule(name)))`
  condition from all three branches (nn.Linear, nn.Conv2d, nn.Embedding) in the
  `named_modules()` loop of `_quantize_int8_model`
- Also removed the redundant local `from mono_quant.modules.embedding import
  quantize_embedding_module` inside the Embedding elif (already imported at line 891)
- Added test: 3-level nested model (`_OuterModule` → `_InnerModule` → `nn.Linear`)
  via `dynamic_quantize`; verified inner nn.Linear is now a QuantizedLinear

## How It Was Done
Single LEVEL 1 edit. Removed the broken condition, leaving the isinstance checks
against `nn.Linear`, `nn.Conv2d`, `nn.Embedding` as the only guards. This is
safe because:
1. Top-level modules are handled by the first `named_children()` loop
2. The second loop skips top-level with `if "." not in name: continue`
3. After the first loop, top-level modules become QuantizedLinear/etc., which
   do not match `isinstance(..., nn.Linear)` since they inherit from nn.Module
   directly — so no double-quantization risk

## Files Changed

| File | Change Type | What Changed |
|------|-------------|--------------|
| src/mono_quant/core/quantizers.py | Modified | Removed broken isinstance guard from 3 branches in named_modules() loop |
| tests/test_bugfixes.py | Modified | Added `test_nested_non_sequential_linear_is_quantized` |

## Testing Results
- Unit: `test_nested_non_sequential_linear_is_quantized` — PASS
- All 28 tests passing

## Final State
**Status**: DONE | **Date**: 2026-02-25
