---
phase: 03-advanced-calibration-&-int4
verified: 2025-02-03T21:00:00Z
status: passed
score: 21/21 must-haves verified
---

# Phase 3: Advanced Calibration & INT4 Verification Report

**Phase Goal:** Users can apply INT4 quantization with advanced observers and layer skipping
**Verified:** 2025-02-03T21:00:00Z
**Status:** PASSED
**Re-verification:** No — Initial verification

## Goal Achievement

### Observable Truths

| #   | Truth | Status | Evidence |
| --- | ----- | ------ | -------- |
| 1 | User can quantize a model to INT4 with group-wise scaling | VERIFIED | quantize_weight_int4() in quantizers.py with group_size=128 default |
| 2 | User can choose MovingAverageMinMaxObserver for robust calibration | VERIFIED | MovingAverageMinMaxObserver in observers.py with EMA smoothing |
| 3 | User can choose Histogram observer for outlier-aware calibration | VERIFIED | HistogramObserver in observers.py with KL divergence minimization |
| 4 | User can skip sensitive layers during quantization | VERIFIED | DEFAULT_INT4_SKIP, _get_layers_to_skip, and static_quantize integration |
| 5 | System warns about potential accuracy issues | VERIFIED | check_accuracy_warnings() with SQNR thresholds and warning categories |
| 6 | Group-wise quantization handles layers smaller than group_size gracefully | VERIFIED | Fallback to per-channel INT8 when dim_size < group_size |
| 7 | INT4 quantization uses symmetric scheme by default | VERIFIED | symmetric=True default in quantize_weight_int4() |
| 8 | QuantizedLinearInt4 module stores packed INT4 weights with per-group scales | VERIFIED | QuantizedLinearInt4 class with _packed_weight, _scales, _zero_points buffers |
| 9 | Serialization format stores INT4 metadata (group_size, bits) | VERIFIED | _build_metadata() includes group_size and bits fields |
| 10 | MovingAverageMinMaxObserver uses configurable averaging_constant | VERIFIED | averaging_constant parameter (default 0.01) |
| 11 | HistogramObserver uses KL divergence for threshold selection | VERIFIED | _find_optimal_threshold() and _compute_kl_divergence() methods |
| 12 | Observers follow same interface as MinMaxObserver | VERIFIED | All have forward(), calculate_qparams(), reset() methods |
| 13 | Observers avoid deprecated torch.ao.quantization APIs | VERIFIED | Custom implementations, no torch.ao imports |
| 14 | DEFAULT_INT4_SKIP skip list contains embeddings, normalization, small layers | VERIFIED | DEFAULT_INT4_SKIP dict with skip_types, skip_param_threshold, skip_names |
| 15 | _get_layers_to_skip supports type, name, and unified modules_to_not_convert | VERIFIED | Function accepts all three parameter types |
| 16 | Accuracy warnings configurable (error/warn/ignore) for CI/CD | VERIFIED | on_failure parameter in check_accuracy_warnings() |
| 17 | Layer skipping API supports type-based, name-based, and unified parameters | VERIFIED | static_quantize accepts skip_layer_types, skip_layer_names, modules_to_not_convert |
| 18 | QuantizationInfo includes warnings list | VERIFIED | warnings: List[str] field in QuantizationInfo dataclass |
| 19 | Observer factory creates observer by name | VERIFIED | create_observer() function in calibration/runner.py |
| 20 | Auto-selection marked as experimental | VERIFIED | _auto_select_observer() with EXPERIMENTAL docstring warning |
| 21 | check_accuracy_warnings checks SQNR, all-layers-quantized, low calibration | VERIFIED | Checks for SQNR thresholds, skipped_layers empty, calibration_samples_used < 50 |

**Score:** 21/21 truths verified (100%)

### Required Artifacts

| Artifact | Expected | Status | Details |
| -------- | -------- | ------ | ------- |
| src/mono_quant/core/mappers.py | Group-wise scale/zero-point calculation | VERIFIED | calculate_scale_zp_groupwise() (150 lines), _pack_int4_to_int8(), _unpack_int8_to_int4() |
| src/mono_quant/core/quantizers.py | INT4 weight quantization function | VERIFIED | quantize_weight_int4() (136 lines) with group-wise scaling |
| src/mono_quant/modules/linear.py | QuantizedLinearInt4 module | VERIFIED | QuantizedLinearInt4 class (194 lines) with from_float() |
| src/mono_quant/core/observers.py | MovingAverageMinMaxObserver | VERIFIED | Class (152 lines) with EMA formula |
| src/mono_quant/core/observers.py | HistogramObserver | VERIFIED | Class (259 lines) with KL divergence |
| src/mono_quant/core/observers.py | DEFAULT_INT4_SKIP constant | VERIFIED | Dict with skip_types, skip_param_threshold, skip_names |
| src/mono_quant/core/observers.py | _get_layers_to_skip function | VERIFIED | Function (92 lines) with unified API |
| src/mono_quant/calibration/runner.py | create_observer factory | VERIFIED | Function (68 lines) with type matching |
| src/mono_quant/calibration/runner.py | _auto_select_observer | VERIFIED | Function (43 lines) marked experimental |
| src/mono_quant/io/validation.py | check_accuracy_warnings | VERIFIED | Function (106 lines) with SQNR thresholds |
| src/mono_quant/io/formats.py | INT4 metadata in _build_metadata | VERIFIED | group_size and bits fields |
| src/mono_quant/core/quantizers.py | Extended static_quantize with layer skipping | VERIFIED | modules_to_not_convert, skip_layer_types, skip_layer_names, skip_param_threshold |
| src/mono_quant/core/quantizers.py | QuantizationInfo with warnings field | VERIFIED | warnings: List[str] field at line 74 |
| src/mono_quant/__init__.py | Public API exports | VERIFIED | check_accuracy_warnings exported |

### Key Link Verification

All key links verified as wired. Core functions import and call dependencies correctly:
- quantize_weight_int4 -> calculate_scale_zp_groupwise, _pack_int4_to_int8
- QuantizedLinearInt4.from_float -> quantize_weight_int4
- create_observer -> All observer classes (factory pattern)
- static_quantize -> _get_layers_to_skip, check_accuracy_warnings

### Requirements Coverage

All requirements satisfied:
- CAL-02: MovingAverageMinMaxObserver support
- CAL-03: HistogramObserver support
- CAL-04: Layer skipping mechanism
- QCORE-02: INT4 quantization support
- VAL-04: Accuracy warning system

### Anti-Patterns Found

None. All implementations are substantive with real logic, comprehensive docstrings, and proper error handling.

### Human Verification Required

None required for implementation completeness. All artifacts exist and are properly wired.

### Gaps Summary

**No gaps found.** All 21 must-haves verified (100%).

**Implementation Highlights:**
- INT4 quantization with group-wise scaling (group_size=128)
- Packed INT4 storage (2 values per int8 byte)
- Fallback for layers smaller than group_size
- Two advanced observers: MovingAverageMinMaxObserver (EMA) and HistogramObserver (KL divergence)
- Observer factory with string-based instantiation
- Default INT4 skip list protecting sensitive layers
- Unified layer skipping API (type/name/threshold)
- Accuracy warning system with configurable thresholds
- Full integration with static_quantize
- INT4 metadata support in serialization

**Phase 3 Status:** PASSED - Goal achieved, all success criteria met.

---

_Verified: 2025-02-03T21:00:00Z_
_Verifier: Claude (gsd-verifier)_
