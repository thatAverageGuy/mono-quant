# BF-003: Fix INT4 symmetric quantization formula — weights are numerically inverted

## Status
TODO

## Audit Reference
C2 (Critical)

## Problem
In `core/quantizers.py:1195–1200`, the INT4 symmetric quantization formula applies
an incorrect `-8` shift:

```python
int4_group = torch.clamp(
    (group_weights / group_scale).round().to(torch.int32) - 8,
    qmin, qmax   # [-8, 7]
)
```

With `scale = max_abs / 7` (from `calculate_scale_zp_groupwise` where `qmax=7`):
- weight = +max_abs → `round(7) - 8 = -1` → after dequant: `-1 * scale = -max_abs/7` ❌
- weight = 0        → `0 - 8 = -8`      → after dequant: `-8 * scale ≠ 0` ❌
- weight = -max_abs → `-7 - 8 = -15` → clamped to -8

The weights are effectively inverted and shifted. Any model using INT4 symmetric
quantization produces silently wrong inference results.

The `-8` is unnecessary. The pack/unpack functions (`_pack_int4_to_int8`,
`_unpack_int8_to_int4`) already handle signed [-8,7] values correctly via two's
complement bitwise operations (`& 0x0F` and the `>= 8 → subtract 16` rule).

## Requirements
1. Remove the spurious `-8` from the symmetric quantization path.
2. Verify dequantization in `QuantizedLinearInt4._dequantize_weight` is consistent.
3. Round-trip test: quantize then dequantize should reconstruct weights within expected
   INT4 error bounds.
4. No change to the asymmetric path.

## Decisions
- **Decision:** Remove `- 8` from symmetric path only.
  Reason: The formula with `-8` maps [qmin=-8, qmax=7] to [-15, -1] after shift,
  which is wrong. Without it, `round(w/scale)` maps to roughly [-7, 7], clamped to [-8, 7].
  Alternatives: Rewrite as unsigned [0,15] with offset — unnecessary complexity given
  the pack/unpack already handles signed INT4.

- **Decision:** Add explicit test verifying that quantize → dequantize → mean absolute
  error is within expected bounds for INT4 (typically < 10% of max_abs).

## Success Criteria
- [ ] `w = torch.randn(256, 128); packed, s, zp = quantize_weight_int4(w, symmetric=True)`
  then `QuantizedLinearInt4.from_float(linear).weight` ≈ `linear.weight` within INT4 tolerance
- [ ] `QuantizedLinearInt4.forward()` produces correct output direction (not inverted)
- [ ] Test: forward pass output has positive correlation with fp32 output (cosine sim > 0.9)
- [ ] INT4 asymmetric path is unchanged and still passes its own round-trip check

## Dependencies
- None (self-contained to `core/quantizers.py` and `modules/linear.py`)

## Implementation Guidance

### Step 1: Fix `quantize_weight_int4` in `core/quantizers.py`
Find the symmetric branch (~line 1194):
```python
# BEFORE
int4_group = torch.clamp(
    (group_weights / group_scale).round().to(torch.int32) - 8,
    qmin, qmax
)

# AFTER
int4_group = torch.clamp(
    (group_weights / group_scale).round().to(torch.int32),
    qmin, qmax
)
```

### Step 2: Verify `_dequantize_weight` in `modules/linear.py`
The dequantization formula is `(int4 - zp) * scale`. For symmetric, `zp=0`, so
`int4 * scale`. With the fix:
- max_abs stored as `round(max_abs / scale) = 7` (or clamped to 7)
- After dequant: `7 * (max_abs/7) = max_abs` ✓
No changes needed to dequantization.

### Step 3: Add round-trip tests in `tests/`
```python
def test_int4_symmetric_round_trip():
    linear = nn.Linear(128, 256)
    q = QuantizedLinearInt4.from_float(linear, symmetric=True)
    dq_weight = q.weight
    # Dequantized should correlate positively with original
    cos_sim = F.cosine_similarity(dq_weight.flatten(), linear.weight.data.flatten(), dim=0)
    assert cos_sim > 0.9, f"Cosine similarity too low: {cos_sim}"
    # Max error should be within INT4 tolerance
    max_abs = linear.weight.data.abs().max().item()
    max_err = (dq_weight - linear.weight.data).abs().max().item()
    assert max_err < 0.2 * max_abs, f"Max error {max_err} > 20% of max_abs {max_abs}"
```

## Testing Requirements
- Unit: Round-trip test as above
- Unit: Verify quantized values are in [-8, 7] after fix
- Unit: Forward pass output direction is preserved (cosine similarity with fp32 output)
- Coverage target: symmetric and asymmetric branches both exercised

## Open Questions
<!-- MUST be empty before implementation begins -->
