# Implementation Log: T-037

## Summary
Added the `monoquant export` CLI command and the `tests/test_onnx_export.py`
test suite (7 tests). Updated a stale `test_bugfixes.py` test that was guarding
the old NotImplementedError stub.

## What Was Done
- Added `export_cmd` to `src/mono_quant/cli/commands.py`.
- Registered `export_cmd` in `src/mono_quant/cli/main.py`.
- Created `tests/test_onnx_export.py` with 7 tests.
- Updated `tests/test_bugfixes.py`: replaced `test_export_to_onnx_raises_not_implemented`
  (now obsolete) with `test_export_to_onnx_is_callable`.

## How It Was Done

### CLI export_cmd
Accepts `--model`, `--output` (both required), `--opset` (default 14),
`--validate` (choice: none/load/full, default none). Loads the model with
`torch.load(path, weights_only=False)` (full model, not state_dict), then
calls `export_to_onnx(model, output_path, opset=opset, validate=validate)`.
ImportError (onnx not installed) surfaces as a `ClickException` with the
helpful pip install message.

### test_onnx_export.py
Module-level `pytest.importorskip("onnx")` and `pytest.importorskip("onnxruntime")`
skip the entire file when optional dependencies are not installed.

| Test | What it verifies |
|------|-----------------|
| `test_export_to_onnx_int8_linear` | No exception on INT8 model export |
| `test_export_onnx_file_exists` | Output .onnx file created |
| `test_export_onnx_validate_load` | validate="load" passes onnx.checker |
| `test_export_onnx_qdq_nodes_present` | QuantizeLinear+DequantizeLinear in graph |
| `test_export_onnx_opset_default_14` | Default opset == 14 in exported model |
| `test_export_onnx_int4_warning` | UserWarning emitted for INT4 + opset<21 |
| `test_export_onnx_raises_without_onnx_installed` | ImportError with pip hint when onnx mocked away |

### INT4 test fix
Original INT4 test helper used `nn.Linear(128, 64)` with `group_size=128`.
`out_features=64 < group_size=128` triggers RuntimeError (correct behavior per BF-007).
Fixed by using `nn.Linear(128, 128)` so `out_features >= group_size`.

### Stale test update
`test_export_to_onnx_raises_not_implemented` asserted `NotImplementedError` was
raised — behavior from the old stub. Since the function is now implemented, updated
to `test_export_to_onnx_is_callable` which verifies the function exists and is callable,
and checks the ImportError path only when onnx is absent.

## Files Changed

| File | Change Type | What Changed |
|------|-------------|--------------|
| `src/mono_quant/cli/commands.py` | Modified | Added `export_cmd` (63 lines) |
| `src/mono_quant/cli/main.py` | Modified | Import + register `export_cmd` |
| `tests/test_onnx_export.py` | Created | 7 ONNX export tests |
| `tests/test_bugfixes.py` | Modified | Updated stale T-030 stub guard test |

## Testing Results
- Unit/integration: 7/7 new ONNX tests passing
- Regression: 35/35 pre-existing tests still passing
- Total: 42/42 passing
- Coverage: all 7 planned test cases implemented

## Final State
**Status**: DONE | **Date**: 2026-02-25
