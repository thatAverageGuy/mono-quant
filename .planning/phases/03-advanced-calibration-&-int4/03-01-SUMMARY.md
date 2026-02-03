---
phase: 03-advanced-calibration-&-int4
plan: 01
subsystem: quantization
tags: [int4, group-wise, quantization, awq, gptq, bit-packing]

# Dependency graph
requires:
  - phase: 02-static-quantization-&-i-o
    provides: per-channel quantization, calibration, serialization formats
provides:
  - INT4 weight quantization with group-wise scaling (group_size=128)
  - QuantizedLinearInt4 module for INT4 linear layers
  - Packed int8 storage format for 2x compression over INT8
  - Group-wise scale/zero-point calculation for INT4 accuracy
affects: [03-02, 03-03]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - Group-wise quantization with configurable group_size
    - Packed INT4 storage (2 values per int8 byte)
    - Fallback to per-channel for layers smaller than group_size
    - Symmetric quantization default for INT4 (zero_point=0)

key-files:
  created: []
  modified:
    - src/mono_quant/core/mappers.py: calculate_scale_zp_groupwise, _pack_int4_to_int8, _unpack_int8_to_int4
    - src/mono_quant/core/quantizers.py: quantize_weight_int4
    - src/mono_quant/modules/linear.py: QuantizedLinearInt4, quantize_linear_module_int4
    - src/mono_quant/io/formats.py: INT4 metadata (group_size, bits)
    - src/mono_quant/core/__init__.py: INT4 function exports

key-decisions:
  - "Group size 128 as default (industry standard from AWQ, GPTQ, HuggingFace)"
  - "Symmetric quantization default for INT4 (simpler, faster, good accuracy)"
  - "Fallback to per-channel INT8 for layers smaller than group_size (safe approach)"
  - "Packed int8 storage for INT4 (PyTorch doesn't have native qint4 support)"

patterns-established:
  - "Pattern 1: Group-wise quantization divides channels into groups of fixed size"
  - "Pattern 2: Bit packing stores 2 INT4 values per INT8 byte for efficiency"
  - "Pattern 3: Symmetric INT4 uses zero_point=0 for all groups"
  - "Pattern 4: Dequantization happens during forward pass in QuantizedLinearInt4"

# Metrics
duration: 9min
completed: 2026-02-03
---

# Phase 3 Plan 1: INT4 Quantization with Group-wise Scaling Summary

**INT4 quantization with 2x compression over INT8 using group-wise scaling (group_size=128) and packed int8 storage format**

## Performance

- **Duration:** 9 min
- **Started:** 2026-02-03T20:45:33Z
- **Completed:** 2026-02-03T20:54:04Z
- **Tasks:** 5
- **Files modified:** 5

## Accomplishments

- INT4 weight quantization with group-wise scaling (industry standard group_size=128)
- QuantizedLinearInt4 module with packed weight storage and per-group dequantization
- Group-wise scale/zero-point calculation with fallback for small layers
- Bit packing functions (_pack_int4_to_int8, _unpack_int8_to_int4) for 2x compression
- Extended serialization format to capture INT4 metadata (group_size, bits)

## Task Commits

Each task was committed atomically:

1. **Task 1: Create group-wise scale/zero-point calculation** - `a4652da` (feat)
2. **Task 2: Create INT4 weight quantization function** - `5aa6b34` (feat)
3. **Task 3: Create QuantizedLinearInt4 module** - `4d5ed27` (feat)
4. **Task 4: Extend serialization for INT4 metadata** - `c242436` (feat)
5. **Task 5: Export INT4 functions from core module** - `07b972f` (feat)

## Files Created/Modified

- `src/mono_quant/core/mappers.py` - Added calculate_scale_zp_groupwise, _pack_int4_to_int8, _unpack_int8_to_int4 functions
- `src/mono_quant/core/quantizers.py` - Added quantize_weight_int4 function with group-wise scaling
- `src/mono_quant/modules/linear.py` - Added QuantizedLinearInt4 class and quantize_linear_module_int4 helper
- `src/mono_quant/io/formats.py` - Extended QuantizationInfo with group_size and bits fields
- `src/mono_quant/core/__init__.py` - Exported new INT4 functions from core module
- `src/mono_quant/modules/__init__.py` - Exported QuantizedLinearInt4 and quantize_linear_module_int4

## Decisions Made

- Group size 128 as default - aligns with AWQ, GPTQ, HuggingFace industry standard
- Symmetric quantization default for INT4 - simpler, faster, maintains good accuracy
- Fallback to per-channel INT8 for layers smaller than group_size - safe approach per RESEARCH.md
- Packed int8 storage for INT4 - PyTorch 2.10 doesn't have native qint4 support

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None - all tasks completed without issues.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- INT4 quantization foundation complete with group-wise scaling
- QuantizedLinearInt4 module ready for use in model quantization
- Serialization format supports INT4 metadata
- Ready for Plan 03-02: Advanced calibration techniques for better accuracy

---
*Phase: 03-advanced-calibration-&-int4*
*Completed: 2026-02-03*
