# T-001: Project Setup, Config System, Model-Agnostic Input Handling

## Status
DONE

## Phase
01-01 — Phase 1: Core Quantization Foundation

## Requirements
- AGN-01: Accept any PyTorch nn.Module
- AGN-02: Accept PyTorch state_dict
- AGN-03: Work with models from any source
- AGN-04: Require only torch as core dependency

## Decisions
- src/ layout with `mono_quant` package (setuptools)
- `QuantizationConfig` dataclass for all quantization parameters
- `_prepare_model()` in `io/handlers.py` handles all input forms
- pyproject.toml with torch as only required dep

## Success Criteria
- [x] Package installs with only torch
- [x] Accepts nn.Module, state_dict, file path
- [x] QuantizationConfig dataclass defined

## Files
- `pyproject.toml` — package configuration
- `src/mono_quant/__init__.py` — package entry point
- `src/mono_quant/config/quant_config.py` — QuantizationConfig
- `src/mono_quant/io/handlers.py` — _prepare_model()
