# T-028: Export Validation and Runtime Compatibility Checks

## Status
TODO

## Phase
08-03 — Phase 8: Unified Export API and Format Conversion

## Requirements
- API-03: Export validation with runtime compatibility checks

## Context
Consolidate all validation logic (ONNX validate, GPTQ vLLM validate, GGUF gguf-py validate)
under a unified interface. Add pre-export compatibility checks.

## Unified Validation Interface

```python
# pre-export check
is_valid, warnings = validate_export_pre(model, info, format)

# post-export check
results = validate_export_post(path, format, runtime=None)
# results.load_ok: bool
# results.inference_ok: bool
# results.warnings: list[str]
```

## Pre-Export Compatibility Checks

| Check | ONNX | GPTQ | AWQ | GGUF |
|-------|------|------|-----|------|
| Model has quantized layers | warn | error | error | error |
| INT4 with opset < 21 | warn→INT8 | ok | ok | re-quant |
| INT8 model for INT4 format | warn | error | error | re-quant |
| Unknown layer types | warn | warn | warn | warn |

## Success Criteria
- [ ] validate_export_pre() checks all format-specific constraints
- [ ] validate_export_post() wraps format-specific post-validation
- [ ] Unified warning/error reporting across all formats
- [ ] --validate flag works consistently across all export formats

## Dependencies
- T-026 (unified Python API)
- T-017 (ONNX validation)
- T-021 (GPTQ/AWQ validation)
- T-025 (GGUF validation)

## Implementation Guidance

1. Add `validate_export_pre()` to `export/common/validators.py`
2. Add `validate_export_post()` dispatcher based on format
3. Standardize ValidationResult structure across all formats
4. Integrate into BaseExporter.export() workflow
