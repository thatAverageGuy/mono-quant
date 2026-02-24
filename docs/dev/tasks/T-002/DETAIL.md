# T-002: Core Quantization Math

## Status
DONE

## Phase
01-02 — Phase 1: Core Quantization Foundation

## Requirements
- QCORE-07: Symmetric and asymmetric quantization schemes

## Decisions
- Symmetric scheme: scale = max(|min|, |max|) / qmax; zero_point = 0
- Asymmetric scheme: scale = (max - min) / (qmax - qmin); zero_point computed from min
- Per-tensor and per-channel mappers separate

## Success Criteria
- [x] Symmetric scheme: zero_point always 0
- [x] Asymmetric scheme: zero_point computed correctly
- [x] Per-channel mappers work for Linear weight shapes

## Files
- `src/mono_quant/core/schemes.py` — quantization scheme implementations
- `src/mono_quant/core/mappers.py` — scale/zero-point calculation
