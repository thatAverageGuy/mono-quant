# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-02-03)

**Core value:** Simple, reliable model quantization with minimal dependencies - just torch, no bloat
**Current focus:** Phase 4: User Interfaces

## Current Position

Phase: 4 of 4 (User Interfaces)
Plan: 1 of 2 (Python API)
Status: Ready to start
Last activity: 2026-02-03 — Completed Phase 3 (all 3 plans, verification passed)

Progress: [██████████████] 75%

## Performance Metrics

**Velocity:**
- Total plans completed: 11
- Average duration: 7.3 min
- Total execution time: 1.3 hours

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 1 | 4 | 4 | 8.5 min |
| 2 | 4 | 4 | 7.5 min |
| 3 | 3 | 3 | 7.2 min |

**Recent Trend:**
- Last 5 plans: 02-02 (11 min), 02-03 (5 min), 02-04 (7 min), 03-01 (9 min), 03-03 (6 min)
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
- Group size 128 as default for INT4 (industry standard from AWQ, GPTQ, HuggingFace)
- Symmetric quantization default for INT4 (simpler, faster, good accuracy)
- Fallback to per-channel INT8 for layers smaller than group_size (safe approach)
- Packed int8 storage for INT4 (PyTorch doesn't have native qint4 support)
- Bit packing stores 2 INT4 values per INT8 byte for 2x compression over INT8
- QuantizedLinearInt4 dequantizes during forward pass using per-group parameters
- DEFAULT_INT4_SKIP list includes Embedding, EmbeddingBag, LayerNorm, BatchNorm1d, BatchNorm2d
- Default parameter threshold of 512 for skipping small layers
- Unified layer skipping API with modules_to_not_convert (HuggingFace compatible)
- SQNR thresholds: <10 dB (critical), <20 dB (warning), >30 dB (good)
- Accuracy warnings checked automatically after quantization
- Warnings tracked in QuantizationInfo for user inspection

### Pending Todos

None yet.

### Blockers/Concerns

None yet.

## Session Continuity

Last session: 2026-02-03
Stopped at: Completed 03-03-PLAN.md (Layer Skipping and Accuracy Warnings)
Resume file: None
