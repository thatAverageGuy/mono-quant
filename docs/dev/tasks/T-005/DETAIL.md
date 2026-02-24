# T-005: Calibration Infrastructure

## Status
DONE

## Phase
02-01 — Phase 2: Static Quantization & I/O

## Requirements
- CAL-01: MinMaxObserver for scale/zero-point computation
- CAL-05: Accept calibration data as tensor list or DataLoader

## Decisions
- MinMaxObserver as base; tracks running min/max per layer
- calibration/runner.py hooks into model forward pass
- calibration/data.py normalizes tensor list vs DataLoader inputs

## Success Criteria
- [x] MinMaxObserver collects stats correctly
- [x] Calibration runner attaches/detaches hooks cleanly
- [x] Both tensor list and DataLoader accepted as input

## Files
- `src/mono_quant/core/observers.py` — MinMaxObserver
- `src/mono_quant/calibration/runner.py` — calibration execution
- `src/mono_quant/calibration/data.py` — data normalization
