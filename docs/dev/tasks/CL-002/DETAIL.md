# CL-002: Version bump 1.1.0 → 2.0.0

## Status
TODO

## Change Level
**LEVEL 1 — SURGICAL**
Two string replacements in two files.

---

## Background

The library has been at version `1.1.0` since v1.1. The work completed through
Phase 8 (ONNX, GPTQ, GGUF export, unified export API, result.convert(), and the
audit bug-fix wave) constitutes a major version increment:
- Breaking: `export` / `export-gptq` / `export-gguf` CLI commands replaced by unified `export`
- Breaking: INT4 skip-list was being injected into INT8 static_quantize (now fixed)
- Major: three new export formats (ONNX, GPTQ, GGUF)
- Major: unified Python export API (result.export(), result.convert())

## Requirements

1. `pyproject.toml`: `version = "1.1.0"` → `version = "2.0.0"`
2. `src/mono_quant/__init__.py`: `__version__ = "1.1.0"` → `__version__ = "2.0.0"`

## Open Questions

*None.*
