# Implementation Log: BF-012

## Summary
Fixed the hardcoded absolute threshold `> 100` in `_check_weight_ranges` that
caused false positives on real models. Replaced with a relative outlier check
based on standard deviations from the mean.

## What Was Done
- Replaced `if torch.any(torch.abs(dequantized) > 100): return False` with a
  relative check: flag if any value is more than 10 standard deviations from
  the mean, which detects quantization-induced corruption without false-positives
  on legitimately large weights
- Added test: a QuantizedLinear with weights initialized to 150.0 (> old threshold
  of 100) passes `_check_weight_ranges` without triggering a false positive

## How It Was Done
LEVEL 1 edit. The original absolute threshold (100) fires unconditionally for
LLM embeddings, final projection layers, and many real-world models where weight
magnitudes commonly exceed 100. A 10-sigma outlier check is meaningful: it
catches quantization explosion (scale miscalculation producing absurd values)
while tolerating large but uniformly-distributed weights.

The original weights are not available inside `_check_weight_ranges` (it only
receives the quantized model), so a self-relative check on the dequantized
tensor is the correct approach per the DETAIL.md guidance.

Before:
```python
if torch.any(torch.abs(dequantized) > 100):
    return False
```

After:
```python
std = dequantized.std().item()
if std > 0:
    mean = dequantized.mean().item()
    max_dev = (dequantized - mean).abs().max().item()
    if max_dev > 10 * std:
        return False
```

## Files Changed

| File | Change Type | What Changed |
|------|-------------|--------------|
| src/mono_quant/io/validation.py | Modified | Replaced hardcoded > 100 with 10-sigma relative check |
| tests/test_bugfixes.py | Modified | Added `test_check_weight_ranges_no_false_positive_on_large_weights` |

## Testing Results
- Unit: `test_check_weight_ranges_no_false_positive_on_large_weights` — PASS
- All 28 tests passing

## Final State
**Status**: DONE | **Date**: 2026-02-25
