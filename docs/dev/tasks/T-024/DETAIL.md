# T-024: Architecture-Specific Tensor Naming Conventions

## Status
TODO

## Phase
07-03 — Phase 7: GGUF Binary Format Export

## Requirements
- GGUF-04: Include required GGUF metadata (arch, quant type, tensor info)
- GGUF-05: Handle architecture parameter for correct tensor naming

## Context
llama.cpp uses architecture-specific tensor names in GGUF files. A tensor named
`model.layers.0.self_attn.q_proj.weight` in PyTorch must be named
`blk.0.attn_q.weight` in GGUF for the llama architecture. Without correct naming,
llama.cpp fails to map tensors to the model graph.

## Architecture Mappings

### LLaMA / Mistral
```
PyTorch name                           → GGUF name
model.embed_tokens.weight              → token_embd.weight
model.layers.N.self_attn.q_proj.weight → blk.N.attn_q.weight
model.layers.N.self_attn.k_proj.weight → blk.N.attn_k.weight
model.layers.N.self_attn.v_proj.weight → blk.N.attn_v.weight
model.layers.N.self_attn.o_proj.weight → blk.N.attn_output.weight
model.layers.N.mlp.gate_proj.weight    → blk.N.ffn_gate.weight
model.layers.N.mlp.up_proj.weight      → blk.N.ffn_up.weight
model.layers.N.mlp.down_proj.weight    → blk.N.ffn_down.weight
model.norm.weight                      → output_norm.weight
lm_head.weight                         → output.weight
```

### GPT-2
```
transformer.wte.weight              → token_embd.weight
transformer.h.N.attn.c_attn.weight  → blk.N.attn_qkv.weight
transformer.h.N.mlp.c_fc.weight     → blk.N.ffn_up.weight
transformer.h.N.mlp.c_proj.weight   → blk.N.ffn_down.weight
```

## Generic/Unknown Architecture
When architecture is unknown or model is generic:
- Pass `--architecture generic` flag
- Use sequential naming: `blk.N.weight_0`, `blk.N.weight_1`, etc.
- Document that llama.cpp may not load generic models correctly

## Success Criteria
- [ ] LLaMA architecture tensor naming correct
- [ ] GPT-2 architecture tensor naming correct
- [ ] Generic/unknown architecture falls back gracefully with warning
- [ ] --architecture CLI flag exposed

## Dependencies
- T-022 (GGUF binary writer)

## Testing Requirements
- Unit: tensor name mapping for LLaMA and GPT-2 architectures
- Integration: llama.cpp loads LLaMA-named tensors correctly

## Open Questions
- [ ] Which architectures to support in v2.0? (LLaMA + generic minimum)
- [ ] Should we auto-detect architecture from tensor names?

## Implementation Guidance

1. Add `src/mono_quant/export/gguf/tensor_map.py` — architecture name mappings
2. Implement `map_tensor_name(pytorch_name, architecture) -> gguf_name`
3. Add `--architecture` flag to `monoquant export --format gguf`
4. Default to auto-detect from tensor name patterns; fall back to generic
