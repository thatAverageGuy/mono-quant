# BF-006: Fix INT4 skip list silently injected into INT8 static quantization

## Status
TODO

## Audit Reference
H2 (High)

## Problem
In `core/quantizers.py`, the `_quantize_int8_model` function (called by
`static_quantize`) contains this branch:

```python
if group_size > 0:          # group_size is a param of quantize_linear_module
    skip_layers |= DEFAULT_INT4_SKIP
```

`DEFAULT_INT4_SKIP` is a set of layer types and name patterns that should be
excluded specifically from INT4 quantization (embeddings, layer norms, lm_head,
etc.). However, `group_size` is also non-zero for INT8 group-wise quantization
paths. This means every INT8 `static_quantize` call with `group_size > 0`
silently inherits the INT4 skip list, causing those layers to be skipped without
any notification to the user.

## Requirements
1. `DEFAULT_INT4_SKIP` must only be applied when the quantization dtype is INT4.
2. INT8 quantization must not apply INT4 skip rules.
3. If a user explicitly provides a `skip_layers` set it must still be respected.

## Decisions
- **Decision:** Gate the injection on `dtype` (or `bits`), not `group_size`.
  Reason: `group_size > 0` is true for both INT4 and grouped INT8 — `dtype` or
  `bits` is the correct discriminator.

## Success Criteria
- [ ] `DEFAULT_INT4_SKIP` is only merged into `skip_layers` when `bits == 4`
      or `dtype` is INT4
- [ ] INT8 static quantize with `group_size > 0` no longer skips embeddings
- [ ] INT4 quantize still skips the default skip types
- [ ] All existing tests pass

## Dependencies
- None

## Implementation Guidance

In `src/mono_quant/core/quantizers.py`, find the branch in `_quantize_int8_model`
(or the caller path from `static_quantize`):

```python
# BEFORE (approximate)
if group_size > 0:
    skip_layers |= DEFAULT_INT4_SKIP

# AFTER — gate on actual INT4 dtype
if bits == 4:          # or: if dtype == torch.qint8 and bits == 4
    skip_layers |= DEFAULT_INT4_SKIP
```

Confirm the surrounding context to find the exact variable name for bits/dtype.

## Testing Requirements
- Unit: Call `static_quantize` with INT8 dynamic, `group_size=32`; verify
  embedding layers are NOT skipped.
- Unit: Call with INT4; verify embedding layers ARE skipped.
- Coverage target: the `DEFAULT_INT4_SKIP` injection branch

## Open Questions
<!-- MUST be empty before implementation begins -->
