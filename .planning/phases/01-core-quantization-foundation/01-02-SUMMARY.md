---
phase: 01-core-quantization-foundation
plan: 02
subsystem: quantization
tags: torch, quantization, symmetric, asymmetric, per-channel, per-tensor, scale, zero-point

# Dependency graph
requires:
  - phase: 01-core-quantization-foundation
    plan: 01
    provides: project structure, package configuration
provides:
  - Quantization scheme classes (SymmetricScheme, AsymmetricScheme)
  - Scale/zero-point calculation functions (per-tensor, per-channel)
  - dtype range mapping utility
affects: 01-03, 01-04 (quantization operations depend on these math primitives)

# Tech tracking
tech-stack:
  added: [PyTorch 2.10]
  patterns: [abstract base class for quantization schemes, functional API for mappers]

key-files:
  created: [src/mono_quant/core/schemes.py, src/mono_quant/core/mappers.py]
  modified: [src/mono_quant/core/__init__.py]

key-decisions:
  - "Per-channel reduction over all dimensions EXCEPT axis (standard for nn.Linear/Conv2d)"
  - "Scale clamped BEFORE zero_point calculation to handle zero-range edge case"
  - "qint4 removed from dtype range - PyTorch 2.10 doesn't support it yet"

patterns-established:
  - "Pattern 1: Abstract base class defines interface, concrete implementations provide algorithms"
  - "Pattern 2: Per-channel quantization assumes standard PyTorch weight layout (axis=0)"
  - "Pattern 3: Edge case handling with clamping before dependent calculations"

# Metrics
duration: 13min
completed: 2026-02-03
---

# Phase 1 Plan 2: Core Quantization Math Summary

**Symmetric and asymmetric quantization schemes with per-tensor/per-channel scale/zero-point calculation using pure PyTorch operations**

## Performance

- **Duration:** 13 min
- **Started:** 2026-02-03T09:23:29Z
- **Completed:** 2026-02-03T09:37:17Z
- **Tasks:** 2
- **Files modified:** 3

## Accomplishments

- Implemented abstract QuantizationScheme base class with calculate() interface
- Created SymmetricScheme with zero_point=0 and max_abs/qmax formula
- Created AsymmetricScheme with affine quantization (range-based scale/zp)
- Implemented functional mapper API: calculate_scale_zp_per_tensor() and calculate_scale_zp_per_channel()
- Added get_dtype_range() utility for qmin/qmax lookup
- All functions handle edge cases: zero-range tensors, FP16 (returns None), per-channel axis specification

## Task Commits

Each task was committed atomically:

1. **Task 1: Create quantization scheme classes** - `b51da43` (feat)
2. **Task 2: Create scale and zero-point mapper functions** - `9395635` (feat)

_Note: Plan metadata commit will follow in final step_

## Files Created/Modified

- `src/mono_quant/core/schemes.py` - QuantizationScheme ABC, SymmetricScheme, AsymmetricScheme classes
- `src/mono_quant/core/mappers.py` - calculate_scale_zp_per_tensor(), calculate_scale_zp_per_channel(), get_dtype_range()
- `src/mono_quant/core/__init__.py` - Exports for schemes and mappers

## Decisions Made

- Per-channel reduction: reduce over all dimensions EXCEPT the specified axis (not just `dim=axis`). For axis=0 on shape (64, 128), this correctly produces (64,) scales.
- Scale clamping order: clamp scale BEFORE calculating zero_point to avoid overflow when range is 0
- Removed qint4/quint4 from dtype ranges: PyTorch 2.10 doesn't have these dtypes yet (will add when available)
- FP16 returns (None, None): FP16 quantization is simple dtype casting, no scale/zero-point needed

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Installed PyTorch dependency**

- **Found during:** Initial verification (before task execution)
- **Issue:** PyTorch was not installed, imports failed
- **Fix:** Ran `pip install torch` to install PyTorch 2.10.0+cpu
- **Files modified:** None (package installed to environment, not project files)
- **Verification:** `python -c "import torch; print(torch.__version__)"` succeeded
- **Committed in:** N/A (environment setup, not code change)

**2. [Rule 1 - Bug] Fixed per-channel reduction dimension calculation**

- **Found during:** Task 1 verification (per-channel shape test)
- **Issue:** Using `amax(dim=axis, keepdim=True)` was reducing along axis instead of over all other dimensions. For (64, 128) with axis=0, produced (1, 128) instead of (64,).
- **Fix:** Changed to reduce over tuple of all dimensions EXCEPT axis: `tuple(i for i in range(tensor.dim()) if i != axis)`
- **Files modified:** src/mono_quant/core/schemes.py (SymmetricScheme.calculate and AsymmetricScheme.calculate)
- **Verification:** Per-channel test now produces correct shape (64,) for axis=0
- **Committed in:** `b51da43` (part of Task 1 commit)

**3. [Rule 1 - Bug] Fixed scale clamping order to prevent overflow**

- **Found during:** Task 2 verification (constant tensor edge case)
- **Issue:** When tensor has zero range (all same values), scale becomes 0, and zero_point calculation `qmin - (min_val / scale)` produces infinity
- **Fix:** Clamp scale to min=1e-8 BEFORE calculating zero_point, not after
- **Files modified:** src/mono_quant/core/mappers.py (both per-tensor and per-channel functions)
- **Verification:** Constant tensor test now completes without overflow
- **Committed in:** `9395635` (part of Task 2 commit)

**4. [Rule 2 - Missing Critical] Removed unsupported dtypes from get_dtype_range()**

- **Found during:** Task 2 verification (dtype range lookup)
- **Issue:** torch.qint4 and torch.quint4 don't exist in PyTorch 2.10, causing AttributeError
- **Fix:** Removed qint4/quint4 from dtype_ranges dict, kept only supported dtypes (qint8, quint8)
- **Files modified:** src/mono_quant/core/mappers.py (get_dtype_range function)
- **Verification:** get_dtype_range(torch.qint8) returns (-128, 127) without error
- **Committed in:** `9395635` (part of Task 2 commit)

---

**Total deviations:** 4 auto-fixed (1 blocking, 3 bugs/missing critical)
**Impact on plan:** All fixes were necessary for correctness. Per-channel reduction logic was key algorithm fix. Clamping order fix prevents crash on degenerate inputs.

## Issues Encountered

None - all issues were auto-fixed via deviation rules.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- Core quantization math is complete and tested
- SymmetricScheme and AsymmetricScheme provide both OOP and functional APIs
- Per-tensor and per-channel calculations verified for correct shapes
- Ready for 01-03: Quantization operations that use these primitives to quantize actual weights

---
*Phase: 01-core-quantization-foundation*
*Completed: 2026-02-03*
