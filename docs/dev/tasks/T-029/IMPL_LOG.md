# Implementation Log: T-029

## Summary

Added `result.convert(bits)` to `QuantizationResult` and `monoquant convert INPUT OUTPUT --bits N`
CLI command. Both use dynamic re-quantization: `dequantize_model(self.model)` then
`quantize(fp32, bits, dynamic=True)`. A UserWarning with the resulting SQNR is emitted.

## What Was Done

### api/result.py

- Added `convert(self, bits, **kwargs) -> QuantizationResult`:
  - Calls `dequantize_model(self.model)` to get FP32
  - Calls `quantize(fp32, bits=bits, dynamic=True)` (high-level API, returns QuantizationResult)
  - Emits `UserWarning` with SQNR and a hint to re-quantize from the original FP32

### cli/commands.py

- Added `convert_cmd(input_path, output_path, --bits)`:
  - Loads model with `torch.load`
  - Dequantizes then re-quantizes via same path as Python API
  - Saves result with `torch.save`
  - Reports SQNR to stdout

### cli/main.py

- Registered `convert_cmd`.

### docs/dev/tasks/T-038/DETAIL.md

- Created stub for calibration-based conversion (deferred, depends on T-029).

## Files Changed

| File | Change Type | What Changed |
|------|-------------|--------------|
| `src/mono_quant/api/result.py` | Modified | Added `convert()` method |
| `src/mono_quant/cli/commands.py` | Modified | Added `convert_cmd` |
| `src/mono_quant/cli/main.py` | Modified | Registered `convert_cmd` |
| `docs/dev/tasks/T-038/DETAIL.md` | Created | Deferred task stub |
| `tests/test_convert.py` | Created | 6 tests |

## Testing Results

- Unit: 6/6 passing
- Key: `test_convert_emits_sqnr_warning` verifies the UserWarning is emitted
- `test_cli_convert_dispatches` verifies the CLI saves a real model file

## Final State

**Status**: DONE | **Date**: 2026-02-26
