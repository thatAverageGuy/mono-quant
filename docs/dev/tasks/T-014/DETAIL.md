# T-014: Export Infrastructure — BaseExporter, Lazy Imports, Validation Framework

## Status
DONE

## Phase
05-01 — Phase 5: ONNX Export

## Requirements
- ONNX-01 to ONNX-06 (infrastructure enabling all)

## Decisions
- BaseExporter abstract class as common interface for all future exporters
- Lazy imports: onnx/onnxruntime imported only when export_to_onnx() is called
- Optional [onnx] extras group in pyproject.toml
- Validation framework: none / load / full levels

## Success Criteria
- [x] BaseExporter abstract class with validate_compatibility, build_metadata, export
- [x] export_to_onnx() uses lazy imports with helpful ImportError message
- [x] onnx + onnxruntime as optional deps only
- [x] Validation framework skeleton ready for ONNX + future formats

## Files
- `src/mono_quant/export/__init__.py` — public export_to_onnx(), lazy imports
- `src/mono_quant/export/base.py` — BaseExporter abstract class
- `src/mono_quant/export/common/validators.py` — validation framework
- `pyproject.toml` — [onnx] optional dep group
