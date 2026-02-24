# T-020: AWQ Checkpoint Export with Activation-Aware Weights

## Status
TODO

## Phase
06-03 — Phase 6: GPTQ/AWQ Checkpoint Exports

## Requirements
- AWQ-01: Export to AWQ checkpoint format
- AWQ-02: Include activation-aware weight information in checkpoint
- AWQ-03: Include quantization_config.json for vLLM/SGLang compatibility

## Context
AWQ (Activation-aware Weight Quantization) uses per-channel scaling factors based
on activation magnitudes. The format stores weights with these activation scales
embedded. AWQ format is somewhat less standardized than GPTQ; study vLLM's AWQ
loader implementation for the expected structure.

## Required quantization_config.json Fields

```json
{
  "quant_method": "awq",
  "zero_point": true,
  "q_group_size": 128,
  "w_bit": 4,
  "version": "GEMM"
}
```

## AWQ Checkpoint Structure

```python
# AWQ weight tensors (per linear layer)
{
  "layer.N.qweight": packed_int4_weights,   # [in_features, out_features // 8]
  "layer.N.scales": per_channel_scales,      # [in_features // group_size, out_features]
  "layer.N.qzeros": zero_points,             # packed zero-points
}
```

## Important: Activation-Aware Scaling
AWQ's key differentiation is per-channel activation scales. If mono-quant
didn't use activation-aware calibration, we approximate by:
- Using the calibration-derived scales as proxy for activation importance
- Document this approximation clearly (not true AWQ, but AWQ-compatible format)

## Critical Pitfall: SGLang vs vLLM AWQ
SGLang requires vLLM operators for AWQ. Document this dependency explicitly:
> AWQ exports require vLLM kernels at inference time. Install vLLM before serving.

## Success Criteria
- [ ] AWQ checkpoint directory written with correct tensor names
- [ ] quantization_config.json with quant_method="awq" and correct fields
- [ ] vLLM loads AWQ checkpoint without errors
- [ ] SGLang loads AWQ checkpoint (or documents vLLM dependency requirement)
- [ ] Approximation (non-true-AWQ scales) documented in output and README

## Dependencies
- T-018 (shares INT4 packing infrastructure with GPTQ)
- T-019 (shares checkpoint directory writing pattern)

## Testing Requirements
- Unit: checkpoint structure matches expected schema
- Integration: vLLM AWQ load test
- Documentation: clearly document approximation vs true AWQ

## Open Questions
- [ ] Should we support "GEMM" vs "GEMV" AWQ versions? (GEMM is more common)
- [ ] How to handle models not quantized with activation-aware calibration?
- [ ] MoE model support? (Requires awq_marlin format in vLLM)

## Implementation Guidance

1. Create `src/mono_quant/export/awq.py` with `AWQExporter(BaseExporter)`
2. Reuse packing logic from T-018 (GPTQExporter._pack_int4_to_int32)
3. Implement AWQ-specific tensor naming and config structure
4. Write integration test against vLLM AWQ loader
5. Document approximation prominently
