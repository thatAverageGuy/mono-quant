# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-02-03)

**Core value:** Simple, reliable model quantization with minimal dependencies - just torch, no bloat
**Current focus:** Phase 2: Static Quantization & I/O

## Current Position

Phase: 2 of 4 (Static Quantization & I/O)
Plan: 3 of 4 (Model Serialization)
Status: In progress
Last activity: 2026-02-03 — Completed 02-03-PLAN.md

Progress: [██████░░░░] 50%

## Performance Metrics

**Velocity:**
- Total plans completed: 7
- Average duration: 7.6 min
- Total execution time: 0.9 hours

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 1 | 4 | 4 | 8.5 min |
| 2 | 3 | 4 | 6.3 min |

**Recent Trend:**
- Last 5 plans: 01-04 (4 min), 02-01 (7 min), 02-02 (7 min), 02-03 (5 min)
- Trend: On track

*Updated after each plan completion*

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table.
Recent decisions affecting current work:

- [Phase 1]: Model-agnostic design - accept any PyTorch nn.Module or state_dict
- [Phase 1]: Weights-only approach - simpler API, focused responsibility
- [Phase 1]: Minimal dependencies - only torch required
- [01-01]: Torch-only dependency enforced via pyproject.toml (AGN-04)
- [01-01]: Model copying via deepcopy to preserve original (CONTEXT.md requirement)
- [01-01]: Configuration priority pattern: kwargs > config > defaults
- [01-02]: Per-channel reduction over all dimensions EXCEPT axis (standard for nn.Linear/Conv2d)
- [01-02]: Scale clamped BEFORE zero_point calculation to handle zero-range edge case
- [01-02]: qint4 removed from dtype range - PyTorch 2.10 doesn't support it yet
- [01-03]: FP16 quantization uses simple dtype casting (not full quantization pipeline)
- [01-03]: Per-channel axis=0 for standard PyTorch weight layouts (out_features/out_channels)
- [01-03]: Bias preserved but not quantized (standard PyTorch quantization practice)
- [01-03]: Conv2d quantization returns standard nn.Conv2d with dequantized weights
- [01-04]: Local imports used in helper functions to avoid circular dependencies
- [01-04]: FP16 quantization uses simple parameter casting (no layer type filtering)
- [01-04]: INT8 quantization uses layer-specific approach (Linear, Conv2d only)
- [02-01]: Custom MinMaxObserver to avoid deprecated torch.ao.quantization APIs (removal in PyTorch 2.10+)
- [02-01]: 150 sample default for calibration aligns with research (100-200 baseline)
- [02-01]: Progress bar auto-detection at 50 samples threshold
- [02-01]: DataLoader (input, target) pattern handled by extracting batch[0]
- [02-02]: Layer type selection with include/skip parameters for flexible quantization
- [02-02]: Support both module type selection and exact layer name selection
- [02-02]: Default layer_types=(nn.Linear, nn.Conv2d) for static quantization
- [02-03]: Safetensors format preferred over PyTorch pickle for secure serialization
- [02-03]: All metadata values must be strings per safetensors constraint
- [02-03]: QuantizationInfo dataclass captures quantization parameters for metadata
- [02-03]: Format auto-detection from file extension (.safetensors vs .pt/.pth)

### Pending Todos

None yet.

### Blockers/Concerns

None yet.

## Session Continuity

Last session: 2026-02-03
Stopped at: Completed 02-03-PLAN.md (model serialization with Safetensors and PyTorch formats)
Resume file: None
