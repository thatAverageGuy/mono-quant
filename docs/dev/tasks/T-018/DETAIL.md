# T-018: 4-Bit Packing Format Matching AutoGPTQ Reference

## Status
TODO

## Phase
06-01 — Phase 6: GPTQ/AWQ Checkpoint Exports

## Requirements
- GPTQ-01: Export to GPTQ checkpoint format with packed INT4 weights
- GPTQ-02: Include per-group scales in GPTQ checkpoint
- GPTQ-03: Include zero-points in GPTQ checkpoint

## Context
GPTQ format packs 4-bit values into 32-bit integers with a specific bit ordering.
The packing format must match AutoGPTQ's reference implementation exactly — incorrect
bit ordering produces models that load but output garbage. Study AutoGPTQ source
for exact packing scheme before implementing.

## Decisions (to finalize during planning)
- Reference: AutoGPTQ / GPTQModel packing format
- group_size=128 is the GPTQ standard
- Symmetric quantization (most GPTQ loaders assume symmetric)
- Tensor shapes: qweight should be (in_features, out_features // 8) for 4-bit

## Success Criteria
- [ ] 4-bit values packed correctly into 32-bit integers (LSB-first, verify exact ordering)
- [ ] per-group scales stored correctly (shape: [in_features // group_size, out_features])
- [ ] zero-points stored correctly
- [ ] Round-trip: unpack → values match original INT4 weights
- [ ] Load test: AutoGPTQ can load the packed format without errors

## Dependencies
- T-017 (Phase 5 complete — export infrastructure established)
- External: AutoGPTQ source code as reference

## Pitfalls (from research)
- CRITICAL: Wrong bit ordering produces garbage outputs silently
- Verify exact bit order against AutoGPTQ source (not documentation)
- qweight shape must be exact: (in_features, out_features // 8)
- Missing quantization_config.json causes vLLM to fail silently

## Testing Requirements
- Unit: test pack/unpack round-trip for various tensor shapes
- Integration: load packed checkpoint with AutoGPTQ, verify outputs match original
- Integration: load with vLLM (different loader than AutoGPTQ)
- Coverage target: all packing code paths + edge cases (non-divisible shapes)

## Open Questions
- [ ] What is the exact bit ordering (LSB vs MSB first)? Verify against AutoGPTQ source.
- [ ] Should we support both symmetric and asymmetric in initial implementation?
- [ ] Does group_size need to be configurable or always 128?

## State Machine

```
quantized model (QuantizedLinearInt4)
    │
    ▼
EXTRACT_WEIGHTS
  ├── get _quantized_weight (packed INT8 storage)
  ├── get scale (per-group, shape [groups, out_features])
  └── get zero_point (per-group)
    │
    ▼
UNPACK_INT4
  ├── unpack INT8 storage → INT4 values
  └── shape: [in_features, out_features]
    │
    ▼
REPACK_GPTQ
  ├── pack 8 × INT4 values into 1 × INT32
  ├── bit ordering: [val0 in bits 0-3, val1 in bits 4-7, ...]
  └── shape: [in_features, out_features // 8]
    │
    ▼
BUILD_CHECKPOINT
  ├── qweight: [in_features, out_features // 8]
  ├── scales: [in_features // group_size, out_features]
  └── qzeros: [in_features // group_size, out_features // 8] (packed zero-points)
```

## Implementation Guidance

1. Create `src/mono_quant/export/gptq.py` with `GPTQExporter(BaseExporter)`
2. Implement `_pack_int4_to_int32(weights: Tensor) -> Tensor` — exact AutoGPTQ bit order
3. Implement `_build_gptq_checkpoint(model, info) -> dict` — checkpoint structure
4. Add round-trip test: pack → unpack → compare to original
5. Add AutoGPTQ load test (requires AutoGPTQ installed in test env)
6. Wire into export/__init__.py and CLI export command
