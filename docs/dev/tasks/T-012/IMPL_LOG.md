# Implementation Log: T-012

## Summary
Unified quantize() Python API and QuantizationResult container. Single entry
point for all quantization modes.

## Status
DONE — 2026-02-03 | Milestone: v1.0

## Files Changed
| File | Change | What |
|------|--------|------|
| src/mono_quant/api/quantize.py | Created | quantize() dispatch function |
| src/mono_quant/api/result.py | Created | QuantizationResult class |
| src/mono_quant/api/exceptions.py | Created | Exception hierarchy |
| src/mono_quant/api/__init__.py | Created | API exports |
