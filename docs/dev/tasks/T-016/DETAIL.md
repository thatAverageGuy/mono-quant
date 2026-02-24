# T-016: Opset Version Support and Quantization Parameter Preservation

## Status
DONE

## Phase
05-03 — Phase 5: ONNX Export

## Requirements
- ONNX-02: Support opset >= 13
- ONNX-03: Preserve scale/zero-point in ONNX graph
- ONNX-04: Support INT8
- ONNX-05: Support INT4 (with fallback)

## Decisions
- Default opset: 14 (balance of compatibility and features)
- No pre-validation of opset — let torch.onnx.export() fail naturally
- INT4 quantization detected via QuantizedLinearInt4 module type
- INT4 fallback: if opset < 21, convert to INT8 with documented warning
- Quantization metadata embedded in ONNX graph (bits, scheme, layer count, version)

## ONNX INT4 Limitation
ONNX opset 21 is required for native INT4 (INT4Tensor type).
For opset < 21, INT4 models export as INT8 with warning.
This is documented behavior, not a bug.

## Success Criteria
- [x] opset parameter accepted and passed to torch.onnx.export
- [x] INT4 detection via _detect_int4_quantization()
- [x] INT4 → INT8 fallback with warning for opset < 21
- [x] _add_quantization_metadata() embeds metadata in ONNX graph
- [x] Exported ONNX graph preserves scale/zero_point from source modules

## Files
- `src/mono_quant/export/onnx.py` — ONNXExporter, export_to_onnx_impl() (530 lines)
- `src/mono_quant/export/onnx_impl.py` — thin wrapper
