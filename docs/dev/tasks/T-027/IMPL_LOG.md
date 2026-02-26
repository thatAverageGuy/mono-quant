# Implementation Log: T-027

## Summary

Replaced three separate CLI export commands (`export`, `export-gptq`, `export-gguf`) with one
unified `export` command. Also added the `convert` command (T-029 co-implementation).
Python API is completely unchanged — zero impact on library users.

## What Was Done

- Removed `export_cmd` (ONNX-only), `export_gptq_cmd`, `export_gguf_cmd` from `commands.py`.
- Added new unified `export_cmd` with:
  - `--model / -m`, `--output / -o`, `--format / -f`, `--list-formats`
  - ONNX options: `--opset`, `--validate`
  - GPTQ options: `--group-size`, `--sym`
  - GGUF options: `--architecture`, `--config`, `--model-param`
- Updated `main.py` — removed old imports/registrations, added unified `export_cmd` + `convert_cmd`.
- Updated `commands.__all__` — removed `export_gptq_cmd`, `export_gguf_cmd`; added `convert_cmd`.
- Updated pre-existing `test_gguf_export.py::test_gguf_export_cli_runs` to use new command.

## Files Changed

| File | Change Type | What Changed |
|------|-------------|--------------|
| `src/mono_quant/cli/commands.py` | Modified | Replaced 3 cmds → 1 unified + convert |
| `src/mono_quant/cli/main.py` | Modified | Updated imports and registrations |
| `tests/test_cli_export.py` | Created | 8 tests for unified export + error cases |
| `tests/test_gguf_export.py` | Modified | Updated CLI test for new command |

## Testing Results

- Unit: 8/8 new CLI tests passing; pre-existing GGUF CLI test updated and passing
- Full suite: 103 passed, 9 skipped, 0 failed

## Final State

**Status**: DONE | **Date**: 2026-02-26
