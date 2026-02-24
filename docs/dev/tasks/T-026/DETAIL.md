# T-026: Python Export API on QuantizationResult

## Status
TODO

## Phase
08-01 — Phase 8: Unified Export API and Format Conversion

## Requirements
- API-01: Python API: QuantizationResult.export(format, path, **kwargs)

## Context
Currently, only ONNX export is available via export_to_onnx() called directly.
After Phases 6-7, we have GPTQ, AWQ, and GGUF exporters. This task unifies them
under a single result.export() interface.

## Proposed API

```python
result = quantize(model, bits=4, calibration_data=data)

# Export to any format
result.export("model.onnx", format="onnx", opset=14)
result.export("model_gptq/", format="gptq")
result.export("model_awq/", format="awq")
result.export("model.gguf", format="gguf", architecture="llama")

# Auto-detect format from extension
result.export("model.onnx")   # → onnx
result.export("model.gguf")   # → gguf
result.export("model_gptq/")  # → gptq (directory)
```

## Export Orchestrator

```python
# export/orchestrator.py
def export_model(model, path, format, info, **options):
    exporter = _get_exporter(format)
    exporter.validate_compatibility(model, info)
    exporter.export(model, path, info, **options)

FORMAT_MAP = {
    'onnx': ONNXExporter,
    'gptq': GPTQExporter,
    'awq': AWQExporter,
    'gguf': GGUFExporter,
}
```

## Success Criteria
- [ ] result.export(path, format) works for all 4 formats
- [ ] Auto-detection of format from extension
- [ ] Helpful error when format is unsupported
- [ ] Docstring with all format options and examples

## Dependencies
- T-017 (ONNX export), T-019 (GPTQ), T-020 (AWQ), T-025 (GGUF)

## Implementation Guidance

1. Create `src/mono_quant/export/orchestrator.py` with `export_model()` dispatch
2. Add `export()` method to `api/result.py` — calls orchestrator
3. Add format auto-detection from path extension
4. Update `export/__init__.py` to export orchestrator function
5. Add `monoquant export <format>` as unified CLI (see T-027)
