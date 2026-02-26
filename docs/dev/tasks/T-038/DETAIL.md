# T-038: Calibration-Based Conversion (DEFERRED)

## Status
TODO

## Phase
08-05 — Phase 8 extension (deferred from T-029)

## Summary

Allow `result.convert(bits, calibration_data=...)` to use static re-quantization
instead of dynamic when calibration data is provided.

Dynamic-only re-quantization (T-029) degrades accuracy compared to quantizing from
the original FP32 model. When calibration data is available, static re-quantization
produces significantly better SQNR.

## Requirements

- When `calibration_data` is passed to `result.convert()`, use `quantize(model, bits, calibration_data=...)` instead of the dynamic path.
- The warning message should distinguish dynamic vs static conversion and report SQNR accordingly.
- CLI `monoquant convert` should accept `--calibration PATH` (optional); if provided, use static path.

## Dependencies

- T-029 (result.convert() — dynamic path)

## Open Questions

None at planning time. Implementation is straightforward — it is an else-branch inside
`result.convert()` that calls `quantize(fp32, bits=bits, calibration_data=calibration_data)`.

## Success Criteria

- [ ] `result.convert(bits=4, calibration_data=[...])` uses static quantization
- [ ] Warning message distinguishes "static" vs "dynamic" path
- [ ] `monoquant convert --bits 4 --calibration calib.pt` works
- [ ] Tests: at least `test_convert_with_calibration_uses_static_path`,
      `test_cli_convert_with_calibration`
