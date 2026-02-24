# T-011: Layer Skipping and Accuracy Warnings

## Status
DONE

## Phase
03-03 — Phase 3: Advanced Calibration & INT4

## Requirements
- CAL-04: Layer skipping to protect sensitive modules
- VAL-04: Warn about potential accuracy issues

## Decisions
- DEFAULT_INT4_SKIP tuple: LayerNorm, BatchNorm, Embedding, MultiheadAttention
- 512-parameter threshold: layers with fewer params auto-skipped from INT4
- SQNR thresholds: >30 dB good, 20-30 warning, <10 critical
- check_accuracy_warnings() called automatically in static_quantize

## Success Criteria
- [x] DEFAULT_INT4_SKIP protects all listed layer types
- [x] 512-param threshold enforced
- [x] SQNR warnings at correct thresholds
- [x] Warnings visible to user in API output

## Files
- `src/mono_quant/core/quantizers.py` — DEFAULT_INT4_SKIP, layer skip logic
- `src/mono_quant/io/validation.py` — check_accuracy_warnings(), SQNR thresholds
