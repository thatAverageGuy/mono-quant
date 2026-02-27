# Implementation Log: T-042

## Summary

Expanded `arch_maps.py` from 5 architectures to 17. Added 12 new LLM architecture
tensor-name and config-key mappings for GGUF export. All 117 tests pass.

## What Was Done

Rewrote `src/mono_quant/export/gguf/arch_maps.py` to add the following architectures:

| Arch string  | Model family                             | HF model_type(s)                        |
|--------------|------------------------------------------|-----------------------------------------|
| `opt`        | Meta OPT                                 | `opt`                                   |
| `phi2`       | Microsoft Phi-1, Phi-1.5, Phi-2          | `phi`, `phi-msft`                       |
| `phi3`       | Microsoft Phi-3, Phi-3.5, Phi-4          | `phi3`                                  |
| `chatglm`    | ChatGLM-2/3, GLM-4                       | `chatglm`, `glm4`                       |
| `falcon`     | Falcon-7B/40B/180B                       | `falcon`, `refinedwebmodel`, `refinedweb` |
| `gemma`      | Google Gemma                             | `gemma`                                 |
| `gemma2`     | Google Gemma 2                           | `gemma2`                                |
| `starcoder`  | StarCoder-1, SantaCoder                  | `gpt_bigcode`                           |
| `starcoder2` | StarCoder-2                              | `starcoder2`                            |
| `bloom`      | BLOOM, BLOOMZ                            | `bloom`                                 |
| `mpt`        | MPT (MosaicML)                           | `mpt`                                   |
| `command-r`  | Command R, Command R+                    | `cohere`                                |

Also added `mixtral → "llama"` (dense layers identical; MoE experts fall to generic).

## How It Was Done

1. **Research**: Web search for each architecture's HF tensor name layout (PyTorch
   parameter names from `named_parameters()`) and the GGUF tensor names expected
   by llama.cpp (from llama.cpp source: `gguf-py/gguf/constants.py` and
   `convert_hf_to_gguf.py` model-specific writers).

2. **Pattern design**: Each tensor map uses `(weight|bias)` as capture group `\2`
   where both weight and bias tensors exist, reducing total patterns. Anchored
   with `^...$` to prevent partial matches.

3. **Gemma2 structure**: `_GEMMA2_TENSOR_MAP = _GEMMA2_EXTRA + _GEMMA_TENSOR_MAP`
   — prepends 3 Gemma2-specific norm patterns before the shared Gemma base map.
   Since first match wins, Gemma2's `post_attention_layernorm` overrides Gemma's.

4. **OPT note**: Added with full tensor map but documented that llama.cpp has no
   OPT model loader (`MODEL_ARCH.OPT` does not exist). GGUF file will be structurally
   valid but non-loadable in llama.cpp until upstream OPT support is added.

5. **Bug fix**: `detect_architecture()` calls `.lower()` on input but `_HF_TYPE_TO_ARCH`
   had mixed-case keys `"RefinedWebModel"` / `"RefinedWeb"` — changed to
   `"refinedwebmodel"` / `"refinedweb"`.

## Files Changed

| File | Change Type | What Changed |
|------|-------------|--------------|
| `src/mono_quant/export/gguf/arch_maps.py` | Modified (full rewrite) | Added 12 architectures + bug fix |
| `tests/test_gguf_export.py` | Modified | 2 new tests: detect_architecture (24 cases) + tensor names (40 cases) |
| `docs/dev/tasks/T-042/DETAIL.md` | Created | Task planning doc |
| `docs/dev/tasks/T-042/IMPL_LOG.md` | Created | This file |
| `docs/dev/tasks/TASKS.md` | Modified | T-042 added to completed |
| `CHANGELOG.md` | Modified | Entry for T-042 |
| `CONTEXT.md` | Modified | Updated current state |
| `README.md` | Modified | GGUF supported architectures listed |

## Detailed Changes Per File

### `src/mono_quant/export/gguf/arch_maps.py` (rewritten)

**`_HF_TYPE_TO_ARCH`** expanded from 5 to 20 entries (+15 model_type keys).

**New tensor maps added (12)**:
- `_OPT_TENSOR_MAP` — 12 patterns for `model.decoder.layers.N.*` layout
- `_PHI2_TENSOR_MAP` — 17 patterns covering both new (`model.layers.*`) and old (`transformer.h.*`) Phi-2 layouts
- `_PHI3_TENSOR_MAP` — 9 patterns; fused `qkv_proj` and `gate_up_proj` → single GGUF tensors
- `_CHATGLM_TENSOR_MAP` — 9 patterns for `transformer.encoder.layers.N.*` layout
- `_FALCON_TENSOR_MAP` — 10 patterns; covers both 7B (single `input_layernorm`) and 40B/180B (separate `ln_attn` + `ln_mlp`)
- `_GEMMA_TENSOR_MAP` — 11 patterns (weight-only; Gemma has no bias)
- `_GEMMA2_EXTRA` + `_GEMMA2_TENSOR_MAP` — 3 extra norms prepended before Gemma base
- `_STARCODER_TENSOR_MAP` — 10 patterns (GPT-2 style + `wpe` position embeddings)
- `_STARCODER2_TENSOR_MAP` — 11 patterns (LLaMA-style but with `c_fc`/`c_proj` FFN names)
- `_BLOOM_TENSOR_MAP` — 10 patterns; includes `word_embeddings_layernorm` (post-embedding norm)
- `_MPT_TENSOR_MAP` — 11 patterns; `transformer.blocks.N` (not `.h.N` or `model.layers.N`)
- `_COMMAND_R_TENSOR_MAP` — 12 patterns; LLaMA-style + per-head `q_norm`/`k_norm`

**New config maps added (10)**: OPT, Phi-2, Phi-3, ChatGLM, Falcon, Gemma, StarCoder, StarCoder2, BLOOM, MPT, Command-R. (`gemma2` reuses `_GEMMA_CONFIG_MAP` — same config keys.)

**Bug fixed**: `"RefinedWebModel"` → `"refinedwebmodel"`, `"RefinedWeb"` → `"refinedweb"` (dict keys must be lowercase since `detect_architecture` calls `.lower()` on input).

## Why These Choices Were Made

- **`(weight|bias)` capture group**: Single pattern per tensor pair vs duplicating for weight+bias — cleaner and less error-prone.
- **Phi-2 dual layout**: Both old and new layouts coexist safely — patterns use distinct anchors (model.layers vs transformer.h) so there's no ambiguity.
- **Phi-3 `gate_up_proj → ffn_up`**: llama.cpp's Phi-3 loader expects fused tensor as `ffn_up`; it splits at inference time. No `ffn_gate` entry.
- **Mixtral → "llama"**: Mixtral's dense layer tensor names are identical to LLaMA. MoE expert tensors (`mlp.experts.N.*`) are not in the LLaMA map and fall through to generic — acceptable since mono-quant doesn't pack MoE experts.
- **OPT exported as "opt" arch string**: Gives structurally named tensors even though llama.cpp can't load them. Better than "generic" sequential names when a future OPT loader appears.

## Testing Results

- Unit: 119/119 passing (2 new tests added)
  - `test_gguf_arch_maps_detect_architecture` — 24 cases covering all model_type mappings incl. mixed-case RefinedWebModel
  - `test_gguf_arch_maps_new_tensor_names` — 40 cases covering representative tensors for all 12 new architectures
- Integration: all passing
- Coverage: no regression; verify_arch_maps.py (mq_manual_test/) ran 88 checks, all pass

## Issues Encountered

- **`detect_architecture` lowercase bug**: Caught during review before committing. Keys `"RefinedWebModel"` and `"RefinedWeb"` would never match because the function `.lower()`s the input. Fixed by lowercasing the keys.

## Impact

- GGUF export now recognizes 17 architectures (was 5) — models from Phi, ChatGLM, Falcon, Gemma, StarCoder, BLOOM, MPT, Command-R families will get correctly named GGUF tensors instead of falling through to generic sequential naming.
- No changes to public API, CLI, or file format.
- OPT is exported with named tensors but llama.cpp inference is not yet possible (no upstream OPT loader).

## Final State

**Status**: DONE | **Commit**: (pending) | **Date**: 2026-02-27
