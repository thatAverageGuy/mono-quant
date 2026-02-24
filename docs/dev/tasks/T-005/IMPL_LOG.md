# Implementation Log: T-005

## Summary
Calibration infrastructure: MinMaxObserver and runner that hooks into a model's
forward pass to collect per-layer quantization statistics.

## Status
DONE — 2026-02-03 | Milestone: v1.0

## Files Changed
| File | Change | What |
|------|--------|------|
| src/mono_quant/core/observers.py | Created | MinMaxObserver |
| src/mono_quant/calibration/runner.py | Created | Calibration forward-pass runner |
| src/mono_quant/calibration/data.py | Created | Tensor list / DataLoader normalization |
