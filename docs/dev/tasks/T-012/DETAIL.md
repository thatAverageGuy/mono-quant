# T-012: Python API — Unified quantize(), QuantizationResult

## Status
DONE

## Phase
04-01 — Phase 4: User Interfaces

## Requirements
- UI-01: Python API quantize(model, bits, dynamic)
- UI-03: Specify quantization parameters via API

## Decisions
- Single quantize() function dispatches to dynamic_quantize or static_quantize
- QuantizationResult wraps (model, info) with .save(), .validate(), .export() methods
- Accepts nn.Module, state_dict, or file path
- bits + calibration_data determine path; dynamic=True forces dynamic path

## Success Criteria
- [x] quantize(model, bits=8, dynamic=True) works
- [x] quantize(model, bits=8, calibration_data=data) works
- [x] QuantizationResult.save() and .validate() work
- [x] Exported from mono_quant.__init__

## Files
- `src/mono_quant/api/quantize.py` — quantize() unified function
- `src/mono_quant/api/result.py` — QuantizationResult
- `src/mono_quant/api/exceptions.py` — exception hierarchy
- `src/mono_quant/api/__init__.py` — exports
