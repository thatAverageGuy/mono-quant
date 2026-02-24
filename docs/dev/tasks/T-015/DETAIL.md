# T-015: ONNX QDQ Node Insertion Utilities

## Status
DONE

## Phase
05-02 — Phase 5: ONNX Export

## Requirements
- ONNX-01: Export with QDQ nodes
- ONNX-03: Preserve scale/zero-point in ONNX graph

## Decisions
- QDQ format (not direct PyTorch quantized:: export — those are ONNX-incompatible)
- DequantizeLinear node per quantized layer
- Scale stored as FLOAT initializer
- Zero-point: INT8 for symmetric, UINT8 for asymmetric with min >= 0
- Per-channel axis=0 for weight quantization

## Key Pitfall Addressed
PyTorch quantized:: operators have no ONNX opset equivalent.
NEVER export QuantizedLinear modules directly to ONNX.
Solution: revert_to_standard_modules() first, then insert QDQ nodes explicitly.

## Success Criteria
- [x] insert_qdq_for_linear() creates correct DequantizeLinear nodes
- [x] insert_qdq_for_conv2d() creates correct DequantizeLinear nodes
- [x] Scale/zero-point preserved exactly from QuantizedLinear attributes
- [x] INT8/UINT8 dtype matching correct per ONNX spec

## Files
- `src/mono_quant/export/common/qdq_inserter.py` — QDQ insertion utilities (558 lines)
