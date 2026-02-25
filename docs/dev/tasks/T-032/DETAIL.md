# T-032: Fix/rewrite `HistogramObserver` — incorrect histogram accumulation and wrong zero-point formula

## Status
TODO

## Audit Reference
C5 (Critical)

## Problem
`HistogramObserver` in `core/observers.py` has two independent bugs:

### Bug A — Histogram accumulation across batches
`forward()` adds histogram bin counts from successive batches:

```python
self.histogram += torch.histc(x, bins=self.bins)
```

`torch.histc` assigns all values into a fixed range `[min, max]` determined
by the **current batch** (or by explicit `min`/`max` args). When successive
batches have different value ranges, the bin edges are different. Adding counts
from incompatible bin edges produces numerically meaningless histograms — the
KL divergence minimisation that follows operates on garbage data.

### Bug B — Asymmetric zero-point formula
`calculate_qparams()` uses:

```python
zero_point = qmin - ((-optimal_threshold / 2) / scale)
```

The `/2` term and the negation assume a symmetric distribution centred at zero.
For asymmetric data (e.g. activations after ReLU which are all positive), this
formula computes wrong zero-points, causing a permanent offset in quantized
values.

The correct formula for asymmetric quantization:
```python
zero_point = round(qmin - min_val / scale)
zero_point = clamp(zero_point, qmin, qmax)
```

## Requirements
1. Fix histogram accumulation to track a consistent bin range across batches.
2. Fix zero-point formula for asymmetric data.
3. KL divergence minimisation must operate on correctly-formed histograms.
4. Symmetric path (where zero_point=0) must remain correct.

## Decisions
- **Decision (Bug A):** Track `running_min` and `running_max` across batches;
  recompute the histogram over the full accumulated range on each forward call.
  Alternatively (simpler): collect raw samples and histogram at `calculate_qparams`
  time. The simplest correct approach: maintain a fixed global `[running_min,
  running_max]` range and always call `torch.histc(x, bins=N, min=running_min,
  max=running_max)` so bins are comparable.

- **Decision (Bug B):** Use the standard PyTorch asymmetric zero-point formula:
  `zp = round(qmin - min_val / scale)` clamped to `[qmin, qmax]`.
  Use `min_val = -optimal_threshold` and `max_val = +optimal_threshold` as the
  KL-selected thresholds.

## Success Criteria
- [ ] Two successive calibration batches with different value ranges produce
      a merged histogram where total count equals total element count across batches
- [ ] Zero-point for all-positive activations (post-ReLU) is non-zero
      (not anchored at zero)
- [ ] KL divergence minimisation selects a reasonable threshold on a
      multimodal distribution
- [ ] MinMaxObserver and MovingAverageMinMaxObserver are unchanged

## Dependencies
- T-031 (calibration dead code fix) — HistogramObserver only matters once
  calibration stats are actually used

## Implementation Guidance

### Fix A: Consistent histogram range

```python
class HistogramObserver(BaseObserver):
    def __init__(self, bins=2048, ...):
        self.bins = bins
        self.histogram = None
        self.running_min = float('inf')
        self.running_max = float('-inf')

    def forward(self, x):
        x_min = x.min().item()
        x_max = x.max().item()
        # Update global range
        new_min = min(self.running_min, x_min)
        new_max = max(self.running_max, x_max)
        if new_min != self.running_min or new_max != self.running_max:
            # Range expanded — rehistogram from scratch if we have old data
            # Simplest: just reset if range changes (conservative approach)
            self.histogram = torch.zeros(self.bins)
            self.running_min = new_min
            self.running_max = new_max
        self.histogram += torch.histc(
            x.float(), bins=self.bins,
            min=self.running_min, max=self.running_max
        )
```

### Fix B: Asymmetric zero-point

```python
def calculate_qparams(self):
    # ... KL divergence finds optimal_threshold (the symmetric clipping range)
    min_val = -optimal_threshold
    max_val = +optimal_threshold
    scale = (max_val - min_val) / (self.qmax - self.qmin)
    zero_point = round(self.qmin - min_val / scale)
    zero_point = max(self.qmin, min(self.qmax, zero_point))
    return torch.tensor([scale]), torch.tensor([zero_point], dtype=torch.int32)
```

## Testing Requirements
- Unit: Feed two batches with ranges [0,1] and [5,6]; verify histogram has
  entries across the full [0,6] range (not two separate [0,1] ranges summed)
- Unit: Feed all-positive activations [0, 1]; verify zero_point > qmin
- Unit: Feed symmetric activations [-1, 1]; verify zero_point ≈ 0 (symmetric)
- Coverage target: histogram accumulation, both symmetric and asymmetric
  zero-point paths

## Open Questions
<!-- MUST be empty before implementation begins -->
