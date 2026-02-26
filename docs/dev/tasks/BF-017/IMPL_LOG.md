# Implementation Log: BF-017

## Summary
Fixed two issues that blocked ONNX export of HuggingFace transformer models:
1. `DynamicCache` pytree error — disabled KV cache before export, restored after.
2. `UnicodeEncodeError` in torch.onnx success callback on Windows — temporarily
   reconfigured stdout/stderr to `errors='replace'` around the dynamo call.

Both fixes are scoped to the dynamo path / HF models only and are no-ops for
plain nn.Module models without a config.

---

## What Was Done

### Fix 1 — KV cache (`ONNXExporter.export()`)

Before Step 5, check for `fp32_model.config.use_cache`. If present, set to `False`
and restore in the outer `finally` block (which already handles `tmp_path` cleanup).
This prevents HuggingFace models from returning `DynamicCache` during tracing.

### Fix 2 — Windows emoji encoding (`ONNXExporter.export()`, dynamo branch)

`torch.onnx.export(..., dynamo=True)` logs a ✅ emoji on success via `print()`.
On Windows, the default CP1252 console encoding raises `UnicodeEncodeError` inside
the success callback, BEFORE the ONNX file is written to disk. The exception is not
a `RuntimeError` or `TypeError`, so our existing catch didn't handle it.

Fix: temporarily reconfigure `sys.stdout` and `sys.stderr` to `errors='replace'`
around the dynamo call, restore to original error mode in a nested `finally`.

---

## Files Changed

| File | Change Type | What Changed |
|------|-------------|--------------|
| `src/mono_quant/export/onnx.py` | Modified | `import sys` added; KV cache disable/restore; stdout/stderr reconfigure for dynamo |
| `tests/test_onnx_export.py` | Modified | 2 new BF-017 tests (use_cache restored on success and failure) |
| `docs/dev/tasks/BF-017/DETAIL.md` | Created | Task planning document |
| `docs/dev/tasks/BF-017/IMPL_LOG.md` | Created | This file |

---

## Why These Choices Were Made

- **`use_cache=False` on config**: Standard HuggingFace export preprocessing step.
  Applies only to models with `config.use_cache` — plain nn.Module is unaffected.
  Restored in `finally` so user's model is never left in a modified state.

- **stdout `reconfigure` not global redirect**: `reconfigure(errors='replace')` is
  the narrowest change that fixes the encoding issue. It restores the original error
  mode in a `finally` and only runs in the dynamo branch. Available Python 3.7+.
  Unlike redirecting stdout entirely, it keeps logging/progress visible.

---

## Testing Results

- Unit: 2/2 new tests passing (use_cache restored on success, use_cache restored on failure)
- Manual test A2 (OPT-125m): **PASSED** — 74 layers quantized, 627MB ONNX written
- All prior: 113/113 passing (no regressions)
- Total: **115 passed, 9 skipped, 0 failures**

---

## Observed Gap (not blocking)

The dynamo-exported OPT-125m ONNX does not contain QDQ nodes — the QDQ inserter
emits warnings "Could not find ONNX initializer for 'model.decoder.layers.N.*.weight'".
Dynamo uses different initializer naming for complex nested models than TorchScript.
The ONNX is valid and runnable (weights dequantized to FP32). QDQ insertion on
dynamo graphs for complex models is a future improvement (not scoped for v2.0).

---

## Final State
**Status**: DONE | **Date**: 2026-02-27
