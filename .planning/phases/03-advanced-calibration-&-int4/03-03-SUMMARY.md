# Phase 3 Plan 3: Layer Skipping and Accuracy Warnings Summary

**One-liner:** INT4 layer skipping mechanism with DEFAULT_INT4_SKIP list and accuracy warning system for detecting potential quantization quality issues.

---

## Frontmatter

```yaml
phase: 03-advanced-calibration-&-int4
plan: 03
subsystem: layer-skipping-accuracy-warnings
tags: [INT4, layer-skipping, accuracy-warnings, SQNR, CI/CD, validation]
type: execute
wave: 2
autonomous: true
```

## Dependency Graph

| Aspect | Description |
|--------|-------------|
| **requires** | Phase 2 (static quantization, validation), Phase 3 Plan 01 (INT4 quantization) |
| **provides** | Layer skipping API for INT4, accuracy warning system, unified skip list |
| **affects** | Phase 3 Plan 04 (advanced calibration techniques) |

## Tech Tracking

| Aspect | Details |
|--------|---------|
| **tech-stack.added** | None (uses existing torch, dataclasses) |
| **tech-stack.patterns** | Unified API pattern (HuggingFace-compatible), threshold-based filtering |

## File Tracking

| File | Status | Description |
|------|--------|-------------|
| `src/mono_quant/core/observers.py` | Modified | Added DEFAULT_INT4_SKIP constant, _get_layers_to_skip unified function |
| `src/mono_quant/core/quantizers.py` | Modified | Extended static_quantize with layer skipping params, added warnings field to QuantizationInfo, integrated check_accuracy_warnings |
| `src/mono_quant/io/validation.py` | Modified | Extended ValidationResult with warnings field, added check_accuracy_warnings function |
| `src/mono_quant/__init__.py` | Modified | Exported check_accuracy_warnings from package root |

## Key Artifacts Created

1. **DEFAULT_INT4_SKIP constant** (`src/mono_quant/core/observers.py`)
   - Embedding, EmbeddingBag, LayerNorm, BatchNorm1d, BatchNorm2d
   - Parameter threshold: 512 (skip small layers)
   - Default skip names: ["lm_head"]

2. **_get_layers_to_skip function** (`src/mono_quant/core/observers.py`)
   - Unified API combining type-based, name-based, and threshold filtering
   - HuggingFace-compatible `modules_to_not_convert` parameter
   - Returns unified skip set for flexible layer exclusion

3. **check_accuracy_warnings function** (`src/mono_quant/io/validation.py`)
   - SQNR thresholds: <10 dB (critical), <20 dB (warning), >30 dB (good)
   - All-layers-quantized warning for aggressive INT4
   - Low calibration sample count warning (<50 samples)
   - Configurable `on_failure` parameter (error/warn/ignore) for CI/CD

4. **Extended static_quantize**
   - New parameters: `modules_to_not_convert`, `skip_layer_types`, `skip_layer_names`, `skip_param_threshold`, `group_size`, `accuracy_warning`, `sqnr_warning_threshold`, `sqnr_error_threshold`
   - Default INT4 skip list automatically applied when `group_size > 0`
   - Accuracy warnings checked after validation

5. **QuantizationInfo.warnings field**
   - List of warning messages about potential accuracy issues
   - Populated by `check_accuracy_warnings` function

## Decisions Made

1. **DEFAULT_INT4_SKIP contents**: Selected based on research recommendations (Embedding, LayerNorm, BatchNorm) with 512-parameter threshold to skip tiny layers where overhead > benefit.

2. **Unified API design**: `modules_to_not_convert` as primary parameter for HuggingFace compatibility, with additional `skip_layer_types`, `skip_layer_names`, `skip_param_threshold` for flexibility.

3. **SQNR thresholds**: Aligned with industry practice - <10 dB critical, <20 dB warning, >30 dB good.

4. **on_failure parameter**: Default "warn" for interactive use, "error" for CI/CD, "ignore" for automated pipelines.

## Deviations from Plan

None - plan executed exactly as written.

## Authentication Gates

No authentication gates encountered during execution.

## Success Criteria Status

| Criterion | Status |
|-----------|--------|
| Layer skipping mechanism protects sensitive components from INT4 quantization | [x] Complete |
| Default INT4 skip list excludes embeddings, normalization layers, and small layers (<512 params) | [x] Complete |
| Unified API (modules_to_not_convert) compatible with HuggingFace patterns | [x] Complete |
| Accuracy warnings alert users to low SQNR (<20 dB warn, <10 dB error) | [x] Complete |
| Warnings for aggressive quantization (all layers quantized) and low calibration count | [x] Complete |
| Configurable warning behavior (error/warn/ignore) for CI/CD compatibility | [x] Complete |
| Warnings tracked in QuantizationInfo for user inspection | [x] Complete |

## Next Phase Readiness

- No blockers identified
- Layer skipping infrastructure in place for INT4 quantization
- Accuracy warning system provides user feedback on quantization quality
- Ready for Phase 3 Plan 04 (advanced calibration techniques)

## Metrics

| Metric | Value |
|--------|-------|
| **Duration** | 5.8 minutes |
| **Completed** | 2026-02-03 |
| **Tasks Completed** | 6/6 |
| **Commits** | 6 |
