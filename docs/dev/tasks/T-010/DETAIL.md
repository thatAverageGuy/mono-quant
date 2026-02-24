# T-010: Advanced Observers (MovingAverageMinMax, Histogram)

## Status
DONE

## Phase
03-02 — Phase 3: Advanced Calibration & INT4

## Requirements
- CAL-02: MovingAverageMinMaxObserver
- CAL-03: HistogramObserver with KL divergence

## Decisions
- MovingAverageMinMax uses exponential moving average (momentum=0.1)
- HistogramObserver with bins=2048; KL divergence for optimal range selection
- Factory function create_observer() supports 'minmax', 'movingaverage', 'histogram', 'auto'

## Success Criteria
- [x] MovingAverageMinMax reduces outlier sensitivity vs MinMax
- [x] HistogramObserver with configurable bins
- [x] Factory function dispatches correctly

## Files
- `src/mono_quant/core/observers.py` — all three observer classes
