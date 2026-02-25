# BF-012: Fix hardcoded weight range threshold in `_check_weight_ranges`

## Status
TODO

## Audit Reference
H4 (High)

## Problem
In `io/validation.py`, `_check_weight_ranges` uses a hardcoded absolute threshold:

```python
if torch.any(torch.abs(dequantized) > 100):
    warnings.append(f"Layer {name}: extreme weight values detected")
```

A threshold of `100` causes false positives on real language models and vision
models where weight magnitudes commonly exceed 100 (e.g. vocabulary embeddings
in LLMs, final projection layers). This fires the warning unconditionally for
normal production models.

Additionally, the threshold is absolute rather than relative — it doesn't
adapt to the model's actual weight distribution. A threshold that flags a 5B
LLM's weights as "extreme" is meaningless.

## Requirements
1. Replace the hardcoded absolute threshold with a relative approach.
2. The check should flag values that are extreme **relative to the layer's
   own distribution** — not a global constant.
3. Configurable threshold preferred (or a documented, justified constant that
   makes sense across model families).

## Decisions
- **Decision:** Replace with a relative threshold: flag when any dequantized
  weight exceeds `N * std(original_weight)` for a configurable `N` (default 5).
  Alternatively, flag if max dequantized value > 3× original max — this catches
  quantization explosion rather than just "large weights".
  Preferred: compare dequantized vs original max to detect quantization-induced
  range expansion. `abs(dequant).max() > 1.5 * abs(original).max()` is a
  meaningful check.
  Reason: The check should detect quantization artefacts, not just large weights.

## Success Criteria
- [ ] No false-positive warnings on a normally-distributed weight tensor with
      values > 100
- [ ] Warning still fires when dequantized weights have exploded (e.g. due to
      scale miscalculation)
- [ ] Threshold is either configurable or clearly documented with rationale

## Dependencies
- None

## Implementation Guidance

In `src/mono_quant/io/validation.py`, `_check_weight_ranges`:

```python
# BEFORE
if torch.any(torch.abs(dequantized) > 100):
    warnings.append(...)

# AFTER — relative check
original_max = torch.abs(original_weight).max().item()
dequant_max = torch.abs(dequantized).max().item()
EXPANSION_THRESHOLD = 1.5   # flag if dequant range > 150% of original
if dequant_max > EXPANSION_THRESHOLD * original_max and original_max > 0:
    warnings.append(
        f"Layer {name}: dequantized range ({dequant_max:.2f}) "
        f"significantly exceeds original ({original_max:.2f})"
    )
```

Adjust based on what `_check_weight_ranges` actually receives as inputs.
If original weights are not available at the call site, use a z-score approach
on the dequantized weights instead.

## Testing Requirements
- Unit: Weight tensor with max value 500 (normal for LLM) → no warning
- Unit: Weight tensor where dequantized max is 3× original max → warning fires
- Coverage target: both warning and no-warning branches

## Open Questions
<!-- MUST be empty before implementation begins -->
