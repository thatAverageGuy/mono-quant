# T-031: Implement activation-based calibration in `static_quantize`

## Status
TODO

## Audit Reference
C4 (Critical — silent dead code)

## Problem
`static_quantize` in `core/quantizers.py` has calibration infrastructure
(observer attachment, `run_calibration`, observer detachment) but the
computed observer statistics are **never used**. After running calibration:

```python
run_calibration(model, calibration_data)
```

The code proceeds to call `quantize_linear_module` using raw weight tensors
directly, completely ignoring the scale/zero-point values computed by the
observers. The calibration step is pure overhead — it runs but has no effect
on the quantized output.

True static quantization requires:
1. Run calibration data through observers to determine input activation ranges.
2. Use those ranges to set the quantization parameters for activations.
3. Apply these parameters during `quantize_linear_module` (not just weights).

## Requirements
1. After `run_calibration`, retrieve observer statistics from each observed
   module.
2. Pass activation scale/zero-point to `quantize_linear_module` (or apply them
   as quantization parameters on the output activations).
3. The observable result: static quantized models should have lower SQNR error
   than dynamic when calibration data is representative.
4. Preserve the existing dynamic quantization path (no regression).

## Decisions
- **Decision:** Implement a `collect_observer_stats()` function that walks
  the model after calibration and extracts `.calculate_qparams()` from each
  attached observer.
  Reason: Observers already implement `calculate_qparams()` — we just need to
  call it and use the results.

- **Decision:** Store activation qparams alongside weight qparams in
  `QuantizedLinear` as `input_scale` / `input_zero_point` attributes.
  Reason: This is the standard PyTorch QAT approach; keeps qparams co-located
  with the module.

## Success Criteria
- [ ] `static_quantize` with calibration data produces different (lower error)
      results than without calibration data
- [ ] Observer statistics are retrieved and applied after `run_calibration`
- [ ] `QuantizedLinear.forward` uses `input_scale`/`input_zero_point` for
      input activation quantization
- [ ] Dynamic quantize path is unchanged

## State Machine

```
  static_quantize()
        │
        ▼
  attach_observers(model)
        │
        ▼
  run_calibration(model, data) ──► observers accumulate stats
        │
        ▼
  collect_observer_stats(model) ──► {layer: (scale, zp)}
        │
        ▼
  quantize_linear_module(layer,
    act_scale=scale,
    act_zp=zp)
        │
        ▼
  detach_observers(model)
        │
        ▼
  return QuantizationResult
```

## Dependencies
- BF-009 (dequantize fix) — may expose issues once calibration actually works
- T-032 (HistogramObserver fix) — calibration stats will be more accurate after

## Implementation Guidance

### Step 1: Add `collect_observer_stats` in `calibration/runner.py`
```python
def collect_observer_stats(model):
    """Return {module_name: (scale, zero_point)} for all observed modules."""
    stats = {}
    for name, module in model.named_modules():
        if hasattr(module, '_activation_observer'):
            scale, zp = module._activation_observer.calculate_qparams()
            stats[name] = (scale, zp)
    return stats
```

### Step 2: Modify `static_quantize` in `core/quantizers.py`
```python
run_calibration(model_copy, calibration_data)
activation_qparams = collect_observer_stats(model_copy)  # ADD THIS
detach_observers(model_copy)

# Pass stats to quantize_linear_module
for name, module in model_copy.named_modules():
    if isinstance(module, nn.Linear):
        qparams = activation_qparams.get(name)
        _replace_with_quantized(model_copy, name, module, config, qparams)
```

### Step 3: Update `QuantizedLinear` to store activation qparams
Add `input_scale` and `input_zero_point` attributes, used in `forward()` to
quantize input activations before the matrix multiply.

## Testing Requirements
- Unit: `static_quantize` with calibration data; verify SQNR > dynamic baseline
- Unit: `static_quantize` without calibration data; verify graceful fallback
  (uses MinMax heuristic or weight-only qparams)
- Integration: Full pipeline with real-ish tensor data; verify no crash
- Coverage target: calibration stat collection path, activation qparam application

## Open Questions
<!-- MUST be empty before implementation begins -->
