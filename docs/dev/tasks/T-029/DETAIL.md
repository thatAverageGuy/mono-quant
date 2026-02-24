# T-029: Format Conversion Between Quantization Types

## Status
TODO

## Phase
08-04 — Phase 8: Unified Export API and Format Conversion

## Requirements
- CONV-01: Convert between quantization formats without re-quantizing
- CONV-02: Re-quantization fallback when direct conversion is impossible

## Context
Users want to convert INT8 ↔ INT4 ↔ FP16 without running calibration again.
Direct conversion is possible when the source has enough precision to represent
the target (e.g., INT8 → INT4 by repacking). Re-quantization from FP32 weights
is the fallback when direct conversion loses too much precision.

## Conversion Matrix

| From | To | Strategy |
|------|----|----------|
| INT8 | INT4 | Re-quantize (INT8 → FP32 → INT4) |
| INT4 | INT8 | Dequantize + re-quantize (loss expected) |
| FP16 | INT8 | Quantize from FP16 weights |
| INT8 | FP16 | Dequantize to FP16 |
| INT8 | FP32 | Already supported: dequantize_model() |
| GPTQ | ONNX | Not direct — re-quantize from original |
| ONNX | GPTQ | Not direct — re-quantize from original |

## API

```python
# Convert quantization format
result = quantize(model, bits=8, calibration_data=data)

# Direct conversion where possible
result_int4 = result.convert(bits=4)           # INT8 → INT4
result_fp16 = result.convert(bits=16)          # INT8 → FP16

# Cannot convert export formats without original model
result.convert(format="gguf")  # Error: use result.export() instead
```

## CLI

```bash
monoquant convert model_int8.pt model_int4.pt --bits 4
monoquant convert model_int8.pt model_fp16.pt --bits 16
```

## Success Criteria
- [ ] INT8 → INT4 conversion works (with SQNR impact warning)
- [ ] INT8 → FP16 conversion works (dequantize + FP16 cast)
- [ ] INT4 → INT8 conversion works (with documented precision loss)
- [ ] Re-quantization fallback documented and triggered appropriately
- [ ] SQNR reported for each conversion

## Dependencies
- T-026 (unified API established)

## Open Questions
- [ ] Should format-to-format conversion (GPTQ ↔ ONNX) be in scope for v2.0?
  Decision: CONV-01/02 scope is quantization bit conversion only. Cross-format
  conversion requires original model — out of scope for v2.0.

## Implementation Guidance

1. Add `convert(bits, ...)` method to `QuantizationResult`
2. Implement conversion paths via dequantize → re-quantize
3. Add SQNR comparison before/after conversion
4. Add `monoquant convert` CLI command
5. Document precision loss expectations for each conversion path
