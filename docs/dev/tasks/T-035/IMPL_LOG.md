# Implementation Log: T-035

## Summary
Implemented QDQ node insertion utilities: `collect_quantization_params()` extracts
per-channel INT8 quantization parameters from quantized modules, and `insert_qdq_nodes()`
rewires the ONNX graph with QuantizeLinear/DequantizeLinear node pairs.

## What Was Done
- Created `src/mono_quant/export/common/qdq_inserter.py` with:
  - `LayerQParams` dataclass (int8_weight, scales, zero_points, axis, is_int4)
  - `collect_quantization_params(model, opset)` — walks named_modules, extracts params
    from QuantizedLinear and QuantizedConv2d; marks QuantizedLinearInt4 as is_int4=True
  - `insert_qdq_nodes(model_proto, qparams)` — inserts QDQ pairs into the ONNX protobuf

## How It Was Done

### collect_quantization_params
Iterates `model.named_modules()`, identifies QuantizedLinear/QuantizedConv2d instances,
and extracts the per-channel quantized weight attributes:
- `._quantized_weight.int_repr()` → int8 weight tensor
- `._quantized_weight.q_per_channel_scales()` → float32 scales
- `._quantized_weight.q_per_channel_zero_points()` → int32 zero points
- `._quantized_weight.q_per_channel_axis()` → quantization axis (0)

QuantizedLinearInt4 is collected with `is_int4=True` so the exporter can warn; its
per-group scales/zero_points are not suitable for opset-14 QDQ insertion.

### insert_qdq_nodes
For each non-INT4 entry:
1. Locates `{name}.weight` in `graph.initializer` (with fallback suffix search + warning).
2. Creates `__qdq_{name}_scale` (float32) and `__qdq_{name}_zero_point` (int8) initializers.
3. Creates QuantizeLinear node (FP32→INT8, axis=0 for per-channel).
4. Creates DequantizeLinear node (INT8→FP32, axis=0 for per-channel).
5. Rewires all graph nodes that consume `{name}.weight` to consume `{name}_dequantized`.
6. Inserts QDQ pair just before the consuming Gemm/Conv node (preserves topological order).

## Files Changed

| File | Change Type | What Changed |
|------|-------------|--------------|
| `src/mono_quant/export/common/qdq_inserter.py` | Created | `LayerQParams`, `collect_quantization_params()`, `insert_qdq_nodes()` |

## Testing Results
- Tested indirectly through T-036/T-037 tests.
- `test_export_onnx_qdq_nodes_present`: verifies QuantizeLinear and DequantizeLinear
  are present in the exported graph.
- 42/42 tests passing after all four tasks.

## Final State
**Status**: DONE | **Date**: 2026-02-25
