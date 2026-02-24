# T-023: Q4_K_M and Q4_K_S Quantization Type Support in GGUF

## Status
TODO

## Phase
07-02 — Phase 7: GGUF Binary Format Export

## Requirements
- GGUF-02: Support Q4_K_M quantization type
- GGUF-03: Support Q4_K_S quantization type

## Context
GGUF defines specific quantization types with exact bit packing and group sizes.
Q4_K_M and Q4_K_S are the most common types for 4-bit quantization in llama.cpp.
The K variants use a "k-quantization" scheme with super-groups.

## Quantization Type Specifications

### Q4_K_M
- 4-bit weights with k-quant method
- group_size = 32 (inner group)
- super_group_size = 8 groups = 256 weights
- Scales stored as FP16 per group
- Mixed: some blocks use Q6_K for more important weights (M = medium)
- GGML type code: GGML_TYPE_Q4_K (12)

### Q4_K_S
- 4-bit weights with k-quant method
- Same structure as Q4_K_M but all blocks Q4_K (S = small)
- GGML type code: GGML_TYPE_Q4_K (12, same as M but different block usage)

## Critical Note
Q4_K_M and Q4_K_S have implementation details defined in llama.cpp source:
`ggml/src/ggml-quants.c` — `quantize_row_q4_K_ref()` is the reference.
DO NOT guess the format. Read this source before implementing.

## group_size Gotcha
Each GGUF quantization type uses a specific group size. DO NOT assume
the group_size used in mono-quant's INT4 (128) matches GGUF's requirements.
Q4_K uses 32-weight inner groups, NOT 128.

## Success Criteria
- [ ] Q4_K_M weight packing matches llama.cpp reference implementation
- [ ] Q4_K_S weight packing matches llama.cpp reference implementation
- [ ] Correct GGML type codes embedded in tensor info
- [ ] llama.cpp loads and runs inference (no garbage output)

## Dependencies
- T-022 (GGUF binary writer)

## Testing Requirements
- Unit: pack/unpack round-trip against reference values from llama.cpp tests
- Integration: llama.cpp inference produces non-garbage output
- Coverage: both Q4_K_M and Q4_K_S paths

## Open Questions
- [ ] Do we need to re-quantize from mono-quant's INT4 (group_size=128) to GGUF's group_size=32?
- [ ] How to handle mono-quant models that weren't quantized with k-quant compatible settings?

## Implementation Guidance

1. Add `src/mono_quant/export/gguf/quant_types.py` — GGUF type codes and block structures
2. Implement `quantize_to_q4_k(weights, type='M') -> bytes` per llama.cpp spec
3. Re-quantize from mono-quant's internal format to GGUF block structure if needed
4. Reference: https://github.com/ggerganov/llama.cpp/blob/master/ggml/src/ggml-quants.c
