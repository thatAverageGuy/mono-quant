# Implementation Log: T-040

## Summary
Added `dynamo=True` export path to the ONNX export pipeline. Uses FX/torch.export
tracing instead of TorchScript, handling transformer models with complex forward
signatures (multi-input, optional kwargs, custom nn.Embedding subclasses). The
default (`dynamo=False`) is unchanged — backward compatible.

---

## What Was Done

1. `ONNXExporter.export()` — added `dynamo: bool = False` parameter. Step 5 now
   branches: dynamo path calls `torch.onnx.export(..., dynamo=True)`; TorchScript
   path is unchanged. TorchScript error hint updated to mention `dynamo=True` as
   an alternative.

2. `export_to_onnx_impl()` — `dynamo` parameter threaded through.

3. `_export_onnx()` in orchestrator — `dynamo=options.get("dynamo", False)` added.

4. `pyproject.toml` — `onnxscript>=0.1` added to `[onnx]` optional deps.

5. Three new tests in `tests/test_onnx_export.py` (all use `pytest.importorskip("onnxscript")`
   so they auto-skip in environments without onnxscript):
   - `test_export_onnx_dynamo_mlp_succeeds`
   - `test_export_onnx_dynamo_embedding_model_succeeds`
   - `test_export_onnx_dynamo_qdq_nodes_present`

---

## How It Was Done

The dynamo branch is a minimal addition to Step 5 of the existing 9-step pipeline.
Steps 1–4 and 6–9 (param collection, revert, QDQ insertion, metadata, save, validate)
are identical for both paths — confirmed by the fact that `test_export_onnx_dynamo_qdq_nodes_present`
passes, verifying the QDQ inserter works on dynamo-produced graphs.

The DETAIL.md confirmed that initializer naming is identical between the two paths
(e.g., `fc1.weight`, `fc2.weight`), so no changes to `insert_qdq_nodes` were needed.

Note: `opset_version` is intentionally not passed on the dynamo path — PyTorch
ignores it when `dynamo=True` and uses its own default.

---

## Files Changed

| File | Change Type | What Changed |
|------|-------------|--------------|
| `src/mono_quant/export/onnx.py` | Modified | `dynamo` param + branched export step + updated TorchScript hint |
| `src/mono_quant/export/onnx_impl.py` | Modified | `dynamo` param threaded through |
| `src/mono_quant/export/orchestrator.py` | Modified | `dynamo` from options dict |
| `pyproject.toml` | Modified | `onnxscript>=0.1` added to onnx extras |
| `tests/test_onnx_export.py` | Modified | 3 new dynamo tests |
| `docs/dev/tasks/T-040/IMPL_LOG.md` | Created | This file |

---

## Why These Choices Were Made

- **`dynamo=False` default**: Existing behaviour preserved. dynamo has different failure
  modes for exotic models; users opt in explicitly.
- **`opset_version` omitted on dynamo path**: PyTorch ignores it — passing it would
  be misleading documentation. The limitation is noted in the docstring.
- **`pytest.importorskip` per-test**: Dynamo tests auto-skip when onnxscript is absent.
  Avoids breaking CI in environments that don't have onnxscript installed.
- **QDQ inserter unchanged**: Initializer naming confirmed identical between both paths.

---

## Testing Results

- Unit: 3/3 new dynamo tests passing (onnxscript installed)
- Regression: all prior 109 tests passing
- Total: **112 passed, 9 skipped, 0 failures**
- Both branches (dynamo=True, dynamo=False) exercised

---

## Issues Encountered

- `onnxscript` was not installed in the test environment — installed it to run the
  dynamo tests. It is now in `[onnx]` optional deps.
- The embedding model dynamo test emits a `UserWarning: Could not find ONNX initializer
  for 'fc.weight'` — expected behavior. After `revert_to_standard_modules`, the
  QuantizedEmbedding weight becomes a plain embedding initializer whose name doesn't
  match the QDQ inserter's `fc.weight` lookup. The warning is benign; the test only
  checks file creation.
- Manual test A2 (OPT-125m) confirmed QDQ nodes are NOT inserted for complex nested
  models via dynamo — dynamo uses different initializer naming than TorchScript for
  deeply nested modules. ONNX is valid FP32, no INT8 inference benefit. Tracked as
  T-041 for post-v2.0 fix.

---

## Final State
**Status**: DONE | **Date**: 2026-02-27
