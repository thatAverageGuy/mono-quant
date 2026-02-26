# Implementation Log: T-024

## Summary
Implemented `GGUFExporter`, architecture tensor name maps, HF config → GGUF KV mapping,
public API `export_to_gguf()`, `monoquant export-gguf` CLI command, and the full test suite
for Phase 7 (T-022–T-025).

## What Was Done
- `arch_maps.py`: tensor name regex patterns for llama/mistral, qwen2, deepseek_v2, gpt2,
  generic; HF config.json → GGUF KV key mapping per architecture; `detect_architecture()`,
  `map_tensor_name()`, `build_kv_entries()`, `reset_generic_counter()`
- `exporter.py` (`GGUFExporter`): full 7-step export pipeline — resolve_config → detect_arch
  → revert_to_FP32 → build_kv → quantize + map names → write
- `gguf_impl.py`: thin wrapper (mirrors gptq_impl.py pattern)
- `export/__init__.py`: added `export_to_gguf()`
- `mono_quant/__init__.py`: added `export_to_gguf` wrapper + `__all__` entry
- `cli/commands.py`: added `export_gguf_cmd` with `--model`, `--output`,
  `--quantization-type`, `--architecture`, `--config`, `--model-param` options
- `cli/main.py`: registered `export_gguf_cmd`

## How It Was Done
- `config_path` + `model_params` both accepted; dict values override config.json on collision
- Architecture auto-detected from `config["model_type"]` via `_HF_TYPE_TO_ARCH` dict
- Unknown architecture or unmatched tensor names → generic sequential naming + `warnings.warn`
- Layers with `in_features % 256 != 0` → fallback to FP32 with warning (no crash)
- Tensor dimensions stored reversed (innermost-first) in GGUF tensor info via writer

## Files Changed

| File | Change Type | What Changed |
|------|-------------|--------------|
| `src/mono_quant/export/gguf/arch_maps.py` | Created | Tensor + config mappings for 5 architectures |
| `src/mono_quant/export/gguf/exporter.py` | Created | GGUFExporter class — full pipeline |
| `src/mono_quant/export/gguf_impl.py` | Created | Thin wrapper |
| `src/mono_quant/export/gguf.py` | Modified | Replaced with redirect comment (shadowed by package) |
| `src/mono_quant/export/__init__.py` | Modified | Added export_to_gguf() |
| `src/mono_quant/__init__.py` | Modified | Added export_to_gguf wrapper + __all__ |
| `src/mono_quant/cli/commands.py` | Modified | Added export_gguf_cmd |
| `src/mono_quant/cli/main.py` | Modified | Registered export_gguf_cmd |
| `tests/test_gguf_export.py` | Created | 22 total tests (T-022–T-025) |

## Issues Encountered

**Name collision between `gguf.py` and `gguf/` package**: Python silently preferred the
package, making `from mono_quant.export.gguf import GGUFExporter` fail since `GGUFExporter`
was in `gguf.py`. Fixed by moving `GGUFExporter` into `gguf/exporter.py` and exporting
from `gguf/__init__.py`. The `gguf.py` file was left in place with a redirect comment
(it is now dead code, shadowed by the package).

## Testing Results
- Unit: 13 pass, 9 skip (gguf-py not installed in CI)
- gguf-py skips: test_gguf_writer_kv_string, test_gguf_writer_kv_uint32,
  test_gguf_writer_kv_float32, test_gguf_writer_tensor_offsets,
  test_gguf_export_ggufpy_reads_file, test_gguf_export_tensor_count,
  test_gguf_export_config_path, test_gguf_export_model_params_override,
  test_validate_gguf_checkpoint_valid
- Full suite: 66 passed, 9 skipped, 0 failed

## Final State
**Status**: DONE | **Commit**: see T-022–T-025 commit | **Date**: 2026-02-26
