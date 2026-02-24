# T-017: CLI Export Command, Error Handling, Validation Testing

## Status
DONE

## Phase
05-04 — Phase 5: ONNX Export

## Requirements
- ONNX-06: Validate exported ONNX with ONNX Runtime

## Decisions
- Four-part error messages: location + root cause + suggested fixes + documentation links
- ONNXValidationError.from_onnx_export_error() analyzes torch.onnx.export() errors
- CLI model reconstruction: infer Sequential/ModuleDict from state_dict for basic export
- export_to_onnx() exported from main mono_quant package
- CLI options: --model, --output, --format, --opset, --mode, --validate

## Success Criteria
- [x] monoquant export --model x.pt --output y.onnx works
- [x] --validate load / --validate full validation levels
- [x] ONNXValidationError provides actionable 4-part messages
- [x] 23 tests covering INT8 export, INT4 export, validation, error paths

## Files
- `src/mono_quant/cli/commands.py` — export_cmd (158 lines added)
- `tests/test_onnx_export.py` — 23 tests, 483 lines
