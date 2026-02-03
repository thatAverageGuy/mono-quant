---
phase: 02-static-quantization-&-i
plan: 01
subsystem: calibration
tags: [observers, calibration, dataloader, progress-reporting, pytorch]

# Dependency graph
requires:
  - phase: 01-core-quantization-foundation
    provides: |
      - Quantization primitives (quantize_weight_int8, schemes)
      - Model I/O handlers (_prepare_model for copying)
      - Module quantization patterns (QuantizedLinear)
provides:
  - MinMaxObserver for tracking activation ranges during calibration
  - Calibration data normalization supporting List[torch.Tensor] and DataLoader
  - run_calibration() function for forward pass execution with progress reporting
affects: [02-integration, 02-validation]

# Tech tracking
tech-stack:
  added: [tqdm (optional dependency for progress bars)]
  patterns:
    - Custom MinMaxObserver to avoid deprecated torch.ao.quantization APIs
    - DataLoader (input, target) batching pattern support
    - Auto-detecting progress bars based on sample count threshold

key-files:
  created:
    - src/mono_quant/core/observers.py - MinMaxObserver class
    - src/mono_quant/calibration/data.py - Data normalization utilities
    - src/mono_quant/calibration/runner.py - Calibration forward pass runner
    - src/mono_quant/calibration/__init__.py - Package exports
  modified:
    - src/mono_quant/core/__init__.py - Exported MinMaxObserver

key-decisions:
  - "Custom MinMaxObserver implementation avoids deprecated torch.ao.quantization.MinMaxObserver (removal in PyTorch 2.10+)"
  - "150 sample default aligns with research recommendation (100-200 baseline per RESEARCH.md)"
  - "Progress bar threshold at 50 samples for auto-detection (Claude's discretion per CONTEXT.md)"
  - "DataLoader (input, target) pattern handled by extracting batch[0]"

patterns-established:
  - "Pattern 1: MinMaxObserver.forward() tracks running min/max across multiple calls"
  - "Pattern 2: _normalize_calibration_data() abstracts DataLoader vs List[torch.Tensor] input"
  - "Pattern 3: Optional tqdm with graceful fallback to silent iteration"

# Metrics
duration: 7 min
completed: 2026-02-03
---

# Phase 2 Plan 1: Calibration Infrastructure Summary

**MinMaxObserver for tracking activation ranges with 150-sample default calibration and DataLoader support**

## Performance

- **Duration:** 7 min
- **Started:** 2026-02-03T13:48:49Z
- **Completed:** 2026-02-03T13:55:00Z
- **Tasks:** 3
- **Files modified:** 5 (4 created, 1 modified)

## Accomplishments

- Created MinMaxObserver class for tracking min/max activation values during calibration
- Implemented calibration data normalization supporting both List[torch.Tensor] and DataLoader inputs
- Built run_calibration() function with optional progress reporting (auto-detects for >50 samples)
- Used 150 sample default aligning with research recommendations (100-200 baseline)
- Custom implementation avoids deprecated torch.ao.quantization APIs (removal in PyTorch 2.10+)

## Task Commits

Each task was committed atomically:

1. **Task 1: Create MinMaxObserver** - `93c1ea1` (feat)
2. **Task 2: Create calibration data normalization** - `5ce6523` (feat)
3. **Task 3: Create calibration runner with progress reporting** - `8bea986` (feat)

## Files Created/Modified

- `src/mono_quant/core/observers.py` - MinMaxObserver class with forward(), calculate_qparams(), reset() methods
- `src/mono_quant/calibration/data.py` - _normalize_calibration_data(), _limit_samples(), CalibrationData type alias
- `src/mono_quant/calibration/runner.py` - run_calibration() with auto-detecting progress bar, _auto_detect_progress_threshold()
- `src/mono_quant/calibration/__init__.py` - Package exports (_normalize_calibration_data, run_calibration)
- `src/mono_quant/core/__init__.py` - Added MinMaxObserver to exports

## Decisions Made

- Custom MinMaxObserver implementation instead of torch.ao.quantization.MinMaxObserver to avoid deprecated API (removal in PyTorch 2.10+)
- 150 sample default for run_calibration() aligns with RESEARCH.md recommendation (100-200 baseline for static quantization)
- Progress bar auto-detection threshold at 50 samples (Claude's discretion per CONTEXT.md)
- DataLoader (input, target) batching pattern handled by extracting batch[0] for supervised learning use case
- Optional tqdm dependency with graceful ImportError fallback to silent iteration

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None - all tasks completed as expected.

## Authentication Gates

None encountered during this plan.

## User Setup Required

None - no external service configuration required. Optional tqdm dependency for progress bars is automatically handled with ImportError fallback.

## Next Phase Readiness

- Calibration infrastructure complete with MinMaxObserver, run_calibration(), and data normalization
- Ready for next plan: Layer type selection with include/exclude patterns
- Calibration sample count (150 default) may need tuning based on model-specific accuracy requirements
- Consider adding: Observer attachment utilities for automatic hook registration on model layers

---
*Phase: 02-static-quantization-&-i*
*Completed: 2026-02-03*
