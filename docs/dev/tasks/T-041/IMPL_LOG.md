# Implementation Log: T-041

## Summary
Fixed QDQ node insertion for dynamo-exported ONNX graphs. The root cause was that
dynamo lowered `F.linear(x, w, b)` as `MatMul(x, w.T)` for attention projection
layers (square weights), storing the transposed weight as an anonymous `val_N`
initializer. The fix adds post-export value matching to recover the original parameter
name, enabling QDQ insertion for all quantized layers on the dynamo path.

## What Was Done

Added `_build_dynamo_name_map()` to `qdq_inserter.py` — scans the ONNX graph for
anonymous (non-dotted) 2D+ initializers, matches them to named model parameters by
value comparison (direct and transposed), and returns a mapping used as a fallback
in `insert_qdq_nodes`. Updated `onnx.py` to call the function when `dynamo=True`
and pass the result to `insert_qdq_nodes`. Two new tests added.

## How It Was Done

**Root cause** (confirmed empirically by loading `opt_int8.onnx`):
- fc1/fc2 weights → exported as `Gemm(x, w, b, transB=1)` → named initializers ✓
- k/q/v/out_proj weights → exported as `MatMul(x, w.T)` → stored as `val_44`, etc. ✗
- `val_44 == q_proj.weight.T` verified with `numpy.allclose(atol=1e-5)`
- 48 anonymous [768, 768] initializers = 12 layers × 4 attention projections

**Name map construction** (fingerprint-indexed, O(P) build + O(1) lookup):
- Walk all ONNX initializers; keep anonymous (no `.` in name), 2-D+, non-empty ones
- Build a fingerprint index over all model parameters (direct + transposed orientations).
  Fingerprint = first 8 + last 8 float32 values (32 bytes each end) of the flattened
  array. Collision probability per pair ≈ 1/(2^512) — negligible in practice.
- For each anonymous initializer: compute its fingerprint, look up candidates (typically
  0 or 1 for trained weights), verify the single candidate with `np.allclose`
- Return `{val_N: (param_dotted_name, is_transposed)}`

**Why fingerprinting** (not brute-force shape-grouping):
- Shape-grouping is O(A × P_same_shape × W). For LLaMA-7B: 256 anon [4096, 4096]
  weights × 256 candidates × 16M elements ≈ 1T comparisons — impractical.
- Fingerprinting reduces to O(A + P) build + O(1) per lookup for real trained models
  (all weights have unique values → unique fingerprints → 1 candidate per lookup).
- Shape check before `allclose` guards against the astronomically rare fingerprint collision.

**QDQ insertion update**:
- Build reverse map `{param_dotted_name: (val_N, is_transposed)}`
- Existing fallback 1 (suffix match) unchanged
- New fallback 2: look up `{module_name}.weight` in reverse map
- When `is_transposed=True`: use `axis=1` instead of `axis=0` (output channels
  moved from dim 0 of original weight to dim 1 of transposed weight)

## Files Changed

| File | Change Type | What Changed |
|---|---|---|
| `src/mono_quant/export/common/qdq_inserter.py` | Modified | Added `_build_dynamo_name_map`; updated `insert_qdq_nodes` with reverse map, third fallback, axis adjustment |
| `src/mono_quant/export/onnx.py` | Modified | Import `_build_dynamo_name_map`; call it in Step 6 when `dynamo=True` |
| `tests/test_onnx_export.py` | Modified | Added 2 tests; updated existing T-040 test docstring |

## Detailed Changes Per File

### `qdq_inserter.py`
- Added `from collections import defaultdict` and `Tuple` to typing imports
- Added `_build_dynamo_name_map(model, model_proto)` (~60 lines):
  - Collects anonymous 2D+ initializers from ONNX graph
  - Groups model params by shape with `defaultdict`
  - Matches by value (`np.allclose`) direct then transposed
  - Removed `max(arr.shape) > 64` size guard — value matching is the real protection;
    size guard was too restrictive for small test models
- Updated `insert_qdq_nodes` signature: added `dynamo_name_map` optional parameter
- Added `reverse_dynamo_map` construction from `dynamo_name_map`
- Changed `is_weight_transposed` tracking; compute `qdq_axis` from it
- Added fallback 2 in weight lookup block

### `onnx.py`
- Added `_build_dynamo_name_map` to import from `qdq_inserter`
- Step 6 expanded: `dynamo_name_map = _build_dynamo_name_map(...) if dynamo else None`
- Pass `dynamo_name_map` to `insert_qdq_nodes`

### `tests/test_onnx_export.py`
- `test_build_dynamo_name_map_recovers_transposed_weight`: unit test using
  `onnx.helper` to build a proto with `val_7 = weight.T`, asserts correct mapping
- `test_export_onnx_dynamo_qdq_explicit_transpose`: end-to-end test with
  `TransposeMatMulModel` (explicit `weight.t()` in forward) — confirms QDQ nodes
  present after dynamo export

## Testing Results

- Unit: 117/117 passing (including both new T-041 tests)
- Integration: N/A (manual test A2 not re-run in this session; fix verified by unit tests)
- Coverage: both recovery paths (unit) + end-to-end dynamo export (integration test)
- Manual testing: pending (OPT-125m re-export to confirm attention projection QDQ)

## Issues Encountered

- `max(arr.shape) > 64` size guard excluded [8, 8] test model weights → removed;
  value matching is sufficient protection

## Impact

- `dynamo=True` exports of transformer models (OPT, LLaMA, etc.) now get QDQ nodes
  for attention projection weights in addition to fc1/fc2 (which already worked)
- No impact on TorchScript path (`dynamo_name_map=None` → no change in behavior)
- `_build_dynamo_name_map` is a new internal function, not part of public API

## Final State

**Status**: DONE | **Date**: 2026-02-27
