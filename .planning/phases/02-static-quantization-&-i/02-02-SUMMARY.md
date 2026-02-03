---
phase: 02-static-quantization-&-i
plan: 02
subsystem: layer-selection
tags: [static-quantization, layer-selection, calibration, pytorch, observers]

# Dependency graph
requires:
  - phase: 01-core-quantization-foundation
    provides: |
      - Quantization primitives (quantize_weight_int8, dequantize_weight)
      - Model I/O handlers (_prepare_model for copying)
      - Module quantization patterns (QuantizedLinear)
  - phase: 02-calibration-infrastructure
    provides: |
      - MinMaxObserver for tracking activation ranges
      - run_calibration() for forward pass execution
      - Calibration data normalization (List[torch.Tensor], DataLoader)
provides:
  - Layer selection functions (_select_layers_by_type, _select_layers_by_name, _merge_selection_results)
  - static_quantize() function with calibration and layer selection
  - dequantize_model() for converting quantized models back to FP32
  - QuantizationInfo dataclass for metadata tracking
affects: [02-model-serialization, 02-validation]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - Layer type/name selection with include/exclude patterns
    - Calibration before quantization for static quantization
    - Model-level dequantization reusing weight-level utilities
    - QuantizationInfo dataclass for metadata reporting

key-files:
  created: []
  modified:
    - src/mono_quant/core/observers.py - Layer selection functions
    - src/mono_quant/core/quantizers.py - static_quantize, dequantize_model, QuantizationInfo
    - src/mono_quant/core/__init__.py - Public API exports
    - src/mono_quant/core/mappers.py - zero_point clamping bug fix

key-decisions:
  - "Layer selection uses isinstance() for type matching (flexible for inheritance)"
  - "Name-based selection uses exact string matching from named_modules()"
  - "Calibration runs before quantization to determine activation ranges"
  - "FP16 static quantization bypasses calibration (simple dtype casting)"
  - "dequantize_model uses inplace=False default for consistency with static_quantize"
  - "zero_point clamped to valid range [-128, 127] to prevent runtime errors"

patterns-established:
  - "Pattern 1: Layer selection returns (selected, skipped) tuples for user reporting"
  - "Pattern 2: Calibration attaches observers via forward hooks, removes after calibration"
  - "Pattern 3: QuantizationInfo dataclass provides metadata about what was quantized"
  - "Pattern 4: Model-level functions reuse lower-level utilities (dequantize_weight)"

# Metrics
duration: 11 min
completed: 2026-02-03
---

# Phase 2 Plan 2: Layer Selection API & Static Quantization Summary

**Layer selection with type/name filtering, static quantization with calibration, and model-level dequantization for FP32 conversion**

## Performance

- **Duration:** 11 min
- **Started:** 2026-02-03T19:24:23Z
- **Completed:** 2026-02-03T19:35:17Z
- **Tasks:** 4
- **Files modified:** 4

## Accomplishments

- Layer selection functions supporting type-based (include/exclude) and name-based filtering
- static_quantize() function that runs calibration before quantizing selected layers
- QuantizationInfo dataclass for reporting selected/skipped layers and calibration metadata
- dequantize_model() for converting quantized models back to FP32
- Fixed zero_point out-of-bounds bug in asymmetric quantization

## Task Commits

Each task was committed atomically:

1. **Task 1: Create layer selection functions** - `3d32e0e` (feat)
2. **Task 2: Create static_quantize function** - `abd10d4` (feat)
3. **Task 3: Export static_quantize from core module** - `f32c820` (feat)
4. **Task 4: Create dequantize_model function** - `19b844d` (feat)

**Bug fix:** `8f5d429` (fix) - Clamp zero_point to valid range

## Files Created/Modified

- `src/mono_quant/core/observers.py` - Added _select_layers_by_type, _select_layers_by_name, _merge_selection_results
- `src/mono_quant/core/quantizers.py` - Added static_quantize, dequantize_model, QuantizationInfo, LayerTypes, CalibrationData, _split_layer_name
- `src/mono_quant/core/__init__.py` - Exported static_quantize, dequantize_model, QuantizationInfo, layer selection functions
- `src/mono_quant/core/mappers.py` - Fixed zero_point clamping in calculate_scale_zp_per_tensor and calculate_scale_zp_per_channel

## Decisions Made

- Layer selection uses isinstance() for type matching, allowing inheritance-based selection
- Name-based selection uses exact string matching from named_modules() hierarchy
- Calibration runs before quantization, attaching observers via forward hooks
- FP16 static quantization bypasses calibration (simple dtype casting)
- dequantize_model uses inplace=False default for consistency with static_quantize API
- zero_point clamped to [-128, 127] range to prevent PyTorch runtime errors

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed zero_point out-of-bounds error in asymmetric quantization**

- **Found during:** Task 4 verification
- **Issue:** zero_point calculation could produce values outside [-128, 127] range, causing `RuntimeError: quantize_tensor_per_channel_affine zero_point X is below lower bound`
- **Fix:** Added zero_point clamping to valid range [qmin, qmax] in both calculate_scale_zp_per_tensor and calculate_scale_zp_per_channel
- **Files modified:** src/mono_quant/core/mappers.py
- **Verification:** All 10 verification checks pass with various weight ranges
- **Committed in:** `8f5d429` (separate commit after task completion)

---

**Total deviations:** 1 auto-fixed (1 bug)
**Impact on plan:** Bug fix necessary for correctness. Quantization would fail for certain weight ranges without this fix. No scope creep.

## Issues Encountered

- MinMaxObserver methods were incorrectly placed at module level instead of class level after Task 1 edit - fixed by restructuring the class definition properly

## Authentication Gates

None encountered during this plan.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- Layer selection API complete with type/name filtering and include/exclude patterns
- Static quantization with calibration is functional
- Model-level dequantization enables recovery from quantization for testing/debugging
- Ready for next plan: Model Serialization (02-03) - save/load quantized models
- Consider adding: Observer attachment utilities for automatic hook registration

---
*Phase: 02-static-quantization-&-i*
*Completed: 2026-02-03*
