# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-02-03)

**Core value:** Simple, reliable model quantization with minimal dependencies - just torch, no bloat
**Current focus:** Phase 3: Advanced Calibration & INT4

## Current Position

Phase: 3 of 4 (Advanced Calibration & INT4)
Plan: 2 of 4 (Advanced Observers, INT4, QAT, Calibration Tools)
Status: In progress
Last activity: 2026-02-03 — Completed Plan 03-02 (Advanced Observers)

Progress: [██████████░] 56%

## Performance Metrics

**Velocity:**
- Total plans completed: 9
- Average duration: 7.4 min
- Total execution time: 1.1 hours

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 1 | 4 | 4 | 8.5 min |
| 2 | 4 | 4 | 7.5 min |
| 3 | 1 | 4 | 7.0 min |

**Recent Trend:**
- Last 5 plans: 02-04 (7 min), 03-01 (7 min), 03-02 (7 min)
- Trend: On track

*Updated after each plan completion*

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table.
Recent decisions affecting current work:

**Phase 1:**
- Model-agnostic design - accept any PyTorch nn.Module or state_dict
- Weights-only approach - simpler API, focused responsibility
- Minimal dependencies - only torch required
- Torch-only dependency enforced via pyproject.toml (AGN-04)
- Model copying via deepcopy to preserve original
- Configuration priority pattern: kwargs > config > defaults
- Per-channel reduction over all dimensions EXCEPT axis
- Scale clamped BEFORE zero_point calculation to handle zero-range edge case
- qint4 removed from dtype range - PyTorch 2.10 doesn't support it yet
- FP16 quantization uses simple dtype casting (not full quantization pipeline)
- Per-channel axis=0 for standard PyTorch weight layouts
- Bias preserved but not quantized
- Conv2d quantization returns standard nn.Conv2d with dequantized weights
- Local imports used in helper functions to avoid circular dependencies

**Phase 2:**
- Custom MinMaxObserver implementation to avoid deprecated torch.ao.quantization APIs (removal in PyTorch 2.10+)
- Calibration sample count: 150 (aligns with research 100-200 baseline)
- Progress bar threshold at 50 samples for auto-detection
- DataLoader (input, target) pattern handled by extracting batch[0]
- Layer selection supports both include (layer_types) and exclude (skip_types)
- Layer selection supports both type-based and name-based selection
- Safetensors metadata values must be strings (JSON-serialize complex types)
- Safetensors metadata stores all 4 categories (quant params, model arch, versions, metrics)
- Global + per-layer metadata structure in Safetensors
- Validation runs by default after quantization
- Validation includes all 4 checks (SQNR, size, load test, weight range)
- on_failure parameter configurable (error/warn/ignore)
- Zero-point clamped to valid range [-128, 127] to prevent PyTorch runtime errors

**Phase 3:**
- MovingAverageMinMaxObserver uses EMA smoothing with averaging_constant default 0.01 (PyTorch standard)
- averaging_constant validated to (0, 1] range to prevent misuse
- HistogramObserver uses KL divergence minimization for threshold selection (TensorRT-style)
- KL divergence search range: 50-100% of histogram to avoid overly narrow ranges
- Histogram bins default: 2048 for good resolution
- Observer factory pattern: create_observer() accepts flexible string names
- Auto-selection marked as experimental due to unreliable heuristics
- All observers share identical interface: forward(), calculate_qparams(), reset()

### Pending Todos

None yet.

### Blockers/Concerns

None yet.

## Session Continuity

Last session: 2026-02-03
Stopped at: Completed 03-02-PLAN.md (Advanced Observers)
Resume file: None
