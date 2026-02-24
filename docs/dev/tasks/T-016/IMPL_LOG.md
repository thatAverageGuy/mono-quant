# Implementation Log: T-016

## Summary
ONNX exporter with opset version support, INT4 handling with documented fallback,
and quantization metadata embedding in the ONNX graph.

## Status
DONE — 2026-02-04 | Milestone: v2.0/Phase 5

## Files Changed
| File | Change | What |
|------|--------|------|
| src/mono_quant/export/onnx.py | Created | ONNXExporter, export_to_onnx_impl() (530 lines) |
| src/mono_quant/export/onnx_impl.py | Created | thin wrapper module |
