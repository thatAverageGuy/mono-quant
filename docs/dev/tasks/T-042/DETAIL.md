# T-042: Expand GGUF arch_maps — 12 new architectures

## Status
DONE

## Requirements

Extend `src/mono_quant/export/gguf/arch_maps.py` to support popular LLM
architectures beyond the original five (llama, mistral, qwen2, gpt2, deepseek_v2).

Architectures requested: OPT, Phi-2, Phi-3/4, ChatGLM/GLM-4, Falcon, Gemma,
Gemma2, StarCoder, StarCoder2, BLOOM, MPT, Command-R.

Each architecture requires:
1. Entry in `_HF_TYPE_TO_ARCH` mapping HF `model_type` → GGUF arch string
2. Tensor name map (`_ARCH_TENSOR_MAP`) — regex patterns mapping PyTorch names → GGUF names
3. Config map (`_ARCH_CONFIG_MAP`) — HF config.json keys → GGUF KV metadata keys
4. Entries in `_ARCH_TENSOR_MAPS` and `_ARCH_CONFIG_MAPS` dispatch tables

## Decisions

- **`(weight|bias)` capture group**: Use as `\2` in new patterns so a single
  regex covers both weight and bias tensors. Reduces pattern count and duplication.
  | Decision | Why |
  |----------|-----|
  | `(weight|bias)` capture group as `\2` | Avoids duplicating each pattern for weight+bias |
  | OPT exported as "opt" arch string | Structurally valid GGUF; llama.cpp has no OPT loader — documented |
  | Mixtral → "llama" | Mixtral dense layers identical to LLaMA; MoE expert routing falls to generic fallback |
  | Phi-2 dual layout | Both "new" (`model.layers.*`) and "old" (`transformer.h.*`) layouts in same map |
  | Phi-3 `gate_up_proj` → `ffn_up` | Fused tensor; llama.cpp splits at inference time |
  | Gemma2 prepends extras before Gemma base | Overrides `post_attention_layernorm` for Gemma2 semantics |
  | Keys in `_HF_TYPE_TO_ARCH` lowercase | `detect_architecture` calls `.lower()` on input; keys must match |

## Success Criteria

- [x] All 12 architectures added to `_HF_TYPE_TO_ARCH`
- [x] All 12 architecture tensor maps defined
- [x] All 12 config maps defined
- [x] `_ARCH_TENSOR_MAPS` and `_ARCH_CONFIG_MAPS` updated
- [x] All 117 existing tests pass
- [x] `detect_architecture` key bug fixed (RefinedWebModel/RefinedWeb lowercased)

## Dependencies
- T-024 (GGUFExporter + arch_maps original implementation)

## Implementation Guidance

See IMPL_LOG.md for full details.

## Open Questions
*None — all resolved during implementation.*
