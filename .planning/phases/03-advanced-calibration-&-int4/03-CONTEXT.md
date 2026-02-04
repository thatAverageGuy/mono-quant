# Phase 3: Advanced Calibration & INT4 - Context

**Gathered:** 2026-02-03
**Status:** Ready for planning

## Phase Boundary

Extend quantization capabilities with INT4 support using group-wise scaling, advanced calibration observers (MovingAverageMinMax, Histogram), layer skipping mechanism to protect sensitive components, and accuracy warning system for aggressive quantization scenarios.

**In scope:**
- INT4 quantization with group-wise scaling for maximum compression
- Advanced calibration observers (MovingAverageMinMax, Histogram) for robust calibration
- Layer skipping to exclude sensitive layers (embeddings, normalization) from quantization
- Accuracy warning system to alert users when aggressive quantization may impact quality

**Out of scope:**
- New quantization types beyond INT4
- Dynamic quantization enhancements
- Model architecture changes

## Implementation Decisions

### INT4 Group-Wise Scaling
- **Group size**: Configurable parameter with default of 128 channels (industry standard balance)
- **Axis configuration**: Configurable with default axis=0 (consistent with existing per-channel convention)
- **Small layer handling**: thatAverageGuy's discretion - choose safest approach for layers smaller than group size
- **API design**: Add `group_size` parameter to quantization functions, user can tune tradeoff

### Layer Skipping Behavior
- **Default skip list for INT4**:
  - Embedding layers (word, positional embeddings - often sensitive to quantization)
  - Normalization layers (LayerNorm, BatchNorm - can cause instability)
  - Layers < 512 parameters (overhead outweighs compression benefit)
- **User configurability**: All defaults are user-configurable and overridable
- **Skip mechanism API**: thatAverageGuy's discretion - choose most intuitive interface (type-based, name-based, or unified)
- **Default behavior**: Skipped layers remain in original precision (FP32/FP16)
- **Advanced option**: Users can specify target precision for skipped layers (e.g., INT8 fallback instead of full skip)

### Observer Selection Strategy
- **Default observer**: MinMaxObserver (proven, reliable)
- **Advanced observers**: MovingAverageMinMax and Histogram available as opt-in
- **Documentation**: Describe scenarios where each observer type is helpful
- **Auto-selection strategy**: Experimental feature with smart selection based on data characteristics
  - thatAverageGuy's discretion on criteria (dataset size, distribution analysis, outliers)
  - Must be clearly marked as experimental if not fool-proof
  - If auto-selection proves unreliable, keep as experimental with user opt-in

### Accuracy Warning System
- **Metrics**: thatAverageGuy's discretion - choose best accuracy preservation indicators
  - Default metrics should be well-documented
  - User-configurable threshold options with detailed explanations
- **Threshold philosophy**: thatAverageGuy's discretion - set sensible defaults
- **Warning behavior**: thatAverageGuy's discretion - choose most practical approach for CI/CD workflows
  - Consider log-only vs. configurable (error/warn/ignore) vs. interactive
  - Must not break automated pipelines

### thatAverageGuy's Discretion
**Layer skipping API:**
- Type-based (`skip_layer_types=['Embedding']`)
- Name-based (`skip_layer_names=['lm_head']`)
- Pattern-based (regex support)
- Or unified parameter accepting all formats

**Small layer handling for INT4:**
- Fail with error if layer < group_size
- Fallback to per-channel quantization
- Skip quantizing that layer
- Any other safe approach

**Observer auto-selection criteria:**
- Dataset size thresholds
- Data distribution analysis
- Outlier detection
- Any combination of robust indicators

**MovingAverageMinMaxObserver averaging constant:**
- Default value (e.g., 10)
- Configurable via parameter
- Adaptive based on characteristics

**Accuracy warning implementation:**
- Which specific metrics to track (SQNR, distribution drift, outliers, etc.)
- Threshold values for warnings
- Warning delivery mechanism (log, error, configurable)
- Integration with existing validation system

## Specific Ideas

- Industry-standard group size of 128 is preferred baseline
- User wants configurability for all major decisions with sensible defaults
- Auto-selection features should be marked as experimental until proven reliable
- Documentation should guide users on when to use each observer type
- CI/CD compatibility is important for warning system design

## Deferred Ideas

None - discussion stayed within phase scope.

---

*Phase: 03-advanced-calibration-&-int4*
*Context gathered: 2026-02-03*
