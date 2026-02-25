# Implementation Log: T-032

## Summary
Fixed two independent bugs in `HistogramObserver`: incorrect histogram
accumulation across batches (Bug A) and wrong scale/zero-point formula (Bug B).

## What Was Done

### Bug A — Consistent histogram bin range
The old code called `torch.histogram(x.flatten(), bins=N)` which computed bin
edges from each batch's own min/max. Adding counts from batches with different
bin edges produced numerically meaningless histograms. The KL divergence
minimisation that followed operated on garbage data.

Fix: use `torch.histc(x.float(), bins=N, min=running_min, max=running_max)` so
all batches share the same bin edges. When the global range expands (new batch
falls outside current running_min/max), reset the histogram — previous counts
used incompatible bin edges and cannot be merged.

Also removed `self.histogram_counts` and `self.bin_edges` attributes; replaced
with a single `self.histogram` (the counts tensor; edges are implicit in
running `min_val`/`max_val`).

### Bug B — Scale and zero-point formula
The old code computed:
```python
scale = optimal_threshold / 255        # off by 2× for symmetric [-T, T]
zero_point = qmin - ((-T / 2) / scale) # wrong negation and /2 term
```

For a symmetric clipping range `[-T, T]`, the correct scale is `2T/255` (the
range spans `T - (-T) = 2T`). The old formula used `T/255`, halving the
representable range. The zero-point formula also had a spurious `/2` division
that happened to produce ≈ 0 for symmetric data but was algebraically wrong.

Fix:
```python
min_val = -optimal_threshold
max_val = optimal_threshold
scale = (max_val - min_val) / (qmax - qmin)  # 2T / 255
zero_point = round(qmin - min_val / scale)    # standard formula
zero_point = clamp(zero_point, qmin, qmax)
```

## Files Changed

| File | Change Type | What Changed |
|------|-------------|--------------|
| src/mono_quant/core/observers.py | Modified | Bug A + Bug B fixes; attribute rename |
| tests/test_bugfixes.py | Modified | 4 new T-032 tests |
| docs/dev/tasks/T-032/IMPL_LOG.md | Created | This file |

## Detailed Changes

### src/mono_quant/core/observers.py
- `__init__`: removed `histogram_counts` and `bin_edges`; added `histogram`
- `forward`: replaced `torch.histogram` with `torch.histc` + fixed-range
  accumulation; added range-expansion detection and histogram reset
- `calculate_qparams`: renamed `histogram_counts` → `histogram`; fixed scale
  to `2T/255` and zero-point to standard `round(qmin - min_val/scale)` formula
- `reset`: removed `bin_edges = None`; renamed `histogram_counts` → `histogram`
- Docstrings updated throughout

### tests/test_bugfixes.py
- `test_histogram_observer_consistent_bins_same_range`: two same-range batches
  → histogram.sum() == 200 (Bug A positive case)
- `test_histogram_observer_range_expands_tracks_full_range`: two different-range
  batches → min_val/max_val cover [0, 6] (Bug A)
- `test_histogram_observer_zero_point_positive_activations`: feed [0,1],
  verify zero_point > qmin = -128 (Bug B)
- `test_histogram_observer_zero_point_symmetric_activations`: feed [-1,1],
  verify zero_point ≈ 0 (Bug B)

## Testing Results
- Unit: `test_histogram_observer_consistent_bins_same_range` — PASS
- Unit: `test_histogram_observer_range_expands_tracks_full_range` — PASS
- Unit: `test_histogram_observer_zero_point_positive_activations` — PASS
- Unit: `test_histogram_observer_zero_point_symmetric_activations` — PASS
- All 32 tests passing

## Final State
**Status**: DONE | **Date**: 2026-02-25
