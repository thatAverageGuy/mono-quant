# Implementation Log: T-036

## Summary
Implemented `ONNXExporter` and its thin `onnx_impl` wrapper, completing the
full 7-step ONNX export pipeline: collect qparams → revert → torch.onnx.export
→ insert QDQ → attach metadata → save → optional validate.

## What Was Done
- Created `src/mono_quant/export/onnx.py` with `ONNXExporter(BaseExporter)`.
- Created `src/mono_quant/export/onnx_impl.py` with `export_to_onnx_impl()` wrapper.

## How It Was Done

### ONNXExporter.export()
Orchestrates the full pipeline:
1. `validate_compatibility(model)` — asserts isinstance nn.Module.
2. `collect_quantization_params(model, opset)` — extract INT8/INT4 qparams.
3. Warn if INT4 layers detected and opset < 21.
4. Filter to INT8-only qparams for QDQ insertion.
5. `revert_to_standard_modules(model, inplace=False)` — FP32 copy.
6. `_infer_dummy_input(fp32_model)` — auto-detect input shape from first Linear/Conv2d.
7. `torch.onnx.export(..., dynamo=False, opset_version=opset)` → temporary .onnx file.
8. `onnx.load(tmp)` + `insert_qdq_nodes(proto, qdq_qparams)`.
9. `build_metadata(model)` → JSON doc_string.
10. `onnx.save(proto, path)`.
11. `validate_onnx_model(path, level)`.

Temporary file is always cleaned up in a finally block.

### Runtime fix discovered
PyTorch 2.10 changed `torch.onnx.export` to default `dynamo=True`, which requires
`onnxscript`. Fixed by passing `dynamo=False` to use the legacy TorchScript exporter.
Input args wrapped in a tuple `(dummy_input,)` as required by the unified API signature.

### _infer_dummy_input
Walks model modules looking for the first `nn.Linear` or `nn.Conv2d` and creates a
`torch.zeros` tensor of the right shape. Raises `RuntimeError` with a helpful message
if no suitable layer is found.

### build_metadata
Returns a dict with: mono_quant_version, export_format="onnx_qdq", opset,
has_int4_layers, library="mono-quant". Stored as JSON in `model_proto.doc_string`.

## Files Changed

| File | Change Type | What Changed |
|------|-------------|--------------|
| `src/mono_quant/export/onnx.py` | Created | `ONNXExporter` class (full pipeline) |
| `src/mono_quant/export/onnx_impl.py` | Created | `export_to_onnx_impl()` thin wrapper |

## Testing Results
- All 7 ONNX export tests pass (see T-037 IMPL_LOG for test details).
- 42/42 total tests passing.

## Final State
**Status**: DONE | **Date**: 2026-02-25
