---
phase: 03-advanced-calibration-&-int4
plan: 02
subsystem: calibration
tags: [observers, ema, kl-divergence, histogram, outlier-robust]

# Dependency graph
requires:
  - phase: 02-static-quantization-&-io
    provides: MinMaxObserver baseline, calibration runner infrastructure
provides:
  - MovingAverageMinMaxObserver with EMA smoothing for outlier handling
  - HistogramObserver with KL divergence minimization for skewed distributions
  - Observer factory pattern for easy instantiation by string name
affects: [03-03-quantization-aware-training, 03-04-int4-quantization]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - Observer factory pattern with string-based instantiation
    - EMA smoothing for calibration robustness
    - KL divergence minimization for threshold selection

key-files:
  created: []
  modified:
    - src/mono_quant/core/observers.py
    - src/mono_quant/calibration/runner.py
    - src/mono_quant/core/__init__.py

key-decisions:
  - averaging_constant default 0.01 matches PyTorch standard
  - KL divergence searches 50-100% of histogram range for threshold
  - Auto-selection marked as experimental due to unreliable heuristics
  - All observers share identical interface (forward, calculate_qparams, reset)

patterns-established:
  - Factory pattern: create_observer() accepts string names with flexible matching
  - Observer substitutability: All observers implement same interface
  - Experimental feature labeling: Auto-selection clearly marked as unreliable

# Metrics
duration: 7min
completed: 2026-02-03
---

# Phase 3 Plan 2: Advanced Calibration Observers Summary

**MovingAverageMinMaxObserver with EMA smoothing and HistogramObserver with KL divergence minimization for robust calibration**

## Performance

- **Duration:** 7 min
- **Started:** 2026-02-03T15:15:30Z
- **Completed:** 2026-02-03T15:22:03Z
- **Tasks:** 4
- **Files modified:** 3

## Accomplishments

- **MovingAverageMinMaxObserver**: EMA-based observer that smooths transient spikes using configurable averaging_constant (default 0.01 per PyTorch standard)
- **HistogramObserver**: Distribution-aware observer using KL divergence minimization for optimal threshold selection (similar to TensorRT)
- **Observer factory**: create_observer() function enables instantiation by string name with flexible matching ("minmax", "movingaverage", "histogram", "auto")
- **Module exports**: Advanced observers exported from mono_quant.core for direct user access

## Task Commits

Each task was committed atomically:

1. **Task 1: MovingAverageMinMaxObserver** - `f60f3db` (feat)
2. **Task 2: HistogramObserver** - `7a3a6d2` (feat)
3. **Task 3: Observer factory** - `9ca287c` (feat)
4. **Task 4: Export from core** - `b6bf273` (feat)

## Files Created/Modified

- `src/mono_quant/core/observers.py` - Added MovingAverageMinMaxObserver and HistogramObserver classes
- `src/mono_quant/calibration/runner.py` - Added create_observer() factory and _auto_select_observer()
- `src/mono_quant/core/__init__.py` - Exported advanced observers, updated module docstring

## Decisions Made

- **averaging_constant validation**: Enforced 0 < c <= 1 range with clear error message, preventing silent misuse
- **KL divergence search range**: Threshold search limited to 50-100% of histogram range to avoid overly narrow quantization ranges
- **Auto-selection heuristics**: Simple kwargs-based approach (averaging_constant -> MovingAverage, bins -> Histogram, default -> MinMax) marked as experimental
- **Histogram bins default**: 2048 bins provides good resolution for most distributions without excessive computation

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

- **Syntax error in module docstring**: Fixed unterminated triple-quoted string by properly formatting the docstring in core/__init__.py

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- Advanced observers ready for use in quantization pipelines
- Observer factory enables easy selection in calibration code
- HistogramObserver KL divergence approach may need refinement based on real-world testing
- Auto-selection heuristics could be improved with distribution analysis in future work

---
*Phase: 03-advanced-calibration-&-int4*
*Plan: 02*
*Completed: 2026-02-03*
