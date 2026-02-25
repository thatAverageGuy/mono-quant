# Implementation Log: T-031

## Summary
Implemented activation-based calibration in `static_quantize`. Observer stats
were previously computed but never used — calibration was silent dead code.
The fix collects input activation ranges after calibration and applies them as
fake-quantized input noise in `QuantizedLinear.forward()`.

## What Was Done

### `calibration/runner.py` — `collect_observer_stats()`
Added new function that iterates the `observers` dict (built inside
`static_quantize`) and calls `calculate_qparams()` on each observer that has
seen data (`min_val is not None`). Returns `{layer_name: (scale, zp)}`.

### `calibration/__init__.py`
Exported `collect_observer_stats` in `__all__`.

### `modules/linear.py` — `QuantizedLinear`
- Added `self.input_scale: Optional[float] = None`
- Added `self.input_zero_point: int = 0`
- `forward()`: when `input_scale` is set, applies
  `torch.fake_quantize_per_tensor_affine` to the input before the matrix
  multiply — simulating INT8 precision loss on activations.

### `modules/linear.py` — `quantize_linear_module()`
Added `input_scale: Optional[torch.Tensor] = None` and
`input_zero_point: Optional[torch.Tensor] = None` parameters. When provided,
sets them as Python floats/ints on the returned `QuantizedLinear`.

### `core/quantizers.py` — `static_quantize()`
Three targeted changes:
1. **Hook fix**: changed `obs.forward(output)` → `obs.forward(input[0])`.
   Observing `input[0]` captures the actual distribution entering each linear
   layer (the correct range for input quantization). The previous approach
   observed each layer's OUTPUT, which is semantically wrong for setting
   per-layer input scales.
2. **Import**: added `collect_observer_stats` to the local import.
3. **Stats collection**: after `hook.remove()` loop, calls
   `collect_observer_stats(observers)` and passes `act[0]`/`act[1]` to
   `quantize_linear_module` for each layer where calibration data was seen.

## Files Changed

| File | Change Type | What Changed |
|------|-------------|--------------|
| src/mono_quant/calibration/runner.py | Modified | Added `collect_observer_stats()` |
| src/mono_quant/calibration/__init__.py | Modified | Exported `collect_observer_stats` |
| src/mono_quant/modules/linear.py | Modified | `input_scale`/`input_zero_point` attrs + `forward()` fake-quant + `quantize_linear_module` params |
| src/mono_quant/core/quantizers.py | Modified | Hook fixed to `input[0]`; import + call `collect_observer_stats`; pass activation qparams |
| tests/test_bugfixes.py | Modified | 3 new T-031 tests |
| docs/dev/tasks/T-031/IMPL_LOG.md | Created | This file |

## Why These Choices Were Made

- **`input[0]` vs `output`**: The input distribution to a layer is what determines
  how that layer's input should be quantized. Output distribution of the previous
  layer and input distribution of the current layer are related but differ for
  activations with nonlinear transforms (e.g., ReLU). Using `input[0]` is strictly
  correct.
- **Plain float/int vs tensor**: `torch.fake_quantize_per_tensor_affine` requires
  Python float for scale and Python int for zero_point. Storing as plain Python
  scalars avoids `.item()` overhead in the forward path.
- **No serialization**: `input_scale`/`input_zero_point` are not registered as
  buffers. Serialization of activation qparams is deferred to future work.

## Testing Results
- Unit: `test_static_quantize_sets_input_scale_from_calibration` — PASS
- Unit: `test_static_quantize_forward_with_activation_qparams` — PASS
- Unit: `test_static_quantize_no_calibration_data_no_input_scale` — PASS
- All 35 tests passing

## Final State
**Status**: DONE | **Date**: 2026-02-25
