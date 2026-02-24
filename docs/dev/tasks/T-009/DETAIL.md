# T-009: INT4 Quantization with Group-Wise Scaling

## Status
DONE

## Phase
03-01 — Phase 3: Advanced Calibration & INT4

## Requirements
- QCORE-02: INT4 quantization with group-wise scaling

## Decisions
- QuantizedLinearInt4 stores weights packed as INT8 (2 × INT4 per byte)
- group_size=128 default (industry standard for GPTQ compatibility)
- Per-group scale/zero_point (not per-channel) for INT4

## Success Criteria
- [x] QuantizedLinearInt4 stores packed INT4 weights
- [x] group_size configurable, default 128
- [x] ~2x compression vs INT8 verified

## Files
- `src/mono_quant/modules/linear.py` — QuantizedLinearInt4
- `src/mono_quant/core/quantizers.py` — INT4 quantization path
