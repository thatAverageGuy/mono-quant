"""Architecture-specific tensor name mappings and HF config → GGUF KV mappings.

Supported architectures:
    llama       — LLaMA, LLaMA-2/3, Mistral, Mixtral (dense layers only)
    qwen2       — Qwen2 / Qwen2.5 (same HF tensor names as LLaMA)
    deepseek2   — DeepSeek-V2 (MLA attention tensors + dense MLP)
    gpt2        — GPT-2
    opt         — Meta OPT (NOTE: llama.cpp has no OPT loader; structural export only)
    phi2        — Microsoft Phi-1, Phi-1.5, Phi-2 (both old/new weight layouts)
    phi3        — Microsoft Phi-3, Phi-3.5, Phi-4
    chatglm     — ChatGLM-2/3, GLM-4 (THUDM)
    falcon      — Falcon-7B, Falcon-40B, Falcon-180B (TII)
    gemma       — Google Gemma (1st generation)
    gemma2      — Google Gemma 2
    starcoder   — StarCoder-1 / SantaCoder (GPTBigCode)
    starcoder2  — StarCoder-2
    bloom       — BLOOM / BLOOMZ (BigScience)
    mpt         — MPT (MosaicML)
    command-r   — Command R / Command R+ (Cohere)
    generic     — Unknown architecture; sequential blk.N.weight_M fallback

GGUF KV file_type values:
    GGML_FTYPE_MOSTLY_Q4_K_S = 14

Tensor name pattern format:
    Each map is a list of (regex_pattern, gguf_template) tuples.
    First match wins. \\1, \\2, etc. refer to regex capture groups.
    (weight|bias) is used as \\2 where both weight and bias tensors exist
    so a single pattern covers both suffixes.
"""

import re
import warnings
from typing import Any, Dict, List, Optional, Tuple

# HF model_type → GGUF architecture string
_HF_TYPE_TO_ARCH: Dict[str, str] = {
    # LLaMA family
    "llama":            "llama",
    "mistral":          "llama",    # Mistral uses LLaMA arch in GGUF
    "mixtral":          "llama",    # Mixtral dense layers identical; MoE experts fall to generic
    # Qwen
    "qwen2":            "qwen2",
    # DeepSeek
    "deepseek_v2":      "deepseek2",
    # GPT-2
    "gpt2":             "gpt2",
    # OPT
    "opt":              "opt",      # NOTE: llama.cpp has no OPT model loader; structural export only
    # Phi
    "phi":              "phi2",     # Phi-1, Phi-1.5, Phi-2
    "phi-msft":         "phi2",     # alternate HF model_type seen on some Phi-2 checkpoints
    "phi3":             "phi3",     # Phi-3, Phi-3.5, Phi-4
    # ChatGLM / GLM-4
    "chatglm":          "chatglm",  # ChatGLM-2, ChatGLM-3, GLM-4
    "glm4":             "chatglm",  # some GLM-4 checkpoints use this model_type
    # Falcon
    "falcon":           "falcon",
    "refinedwebmodel":  "falcon",   # early Falcon checkpoint naming (lowercased at lookup)
    "refinedweb":       "falcon",
    # Gemma
    "gemma":            "gemma",
    "gemma2":           "gemma2",
    # StarCoder
    "gpt_bigcode":      "starcoder",
    "starcoder2":       "starcoder2",
    # BLOOM
    "bloom":            "bloom",
    # MPT
    "mpt":              "mpt",
    # Command-R
    "cohere":           "command-r",
}

# GGML file_type values
GGML_FTYPE_MOSTLY_Q4_K_S = 14

# ---------------------------------------------------------------------------
# Tensor name mappings — list of (regex_pattern, gguf_template)
# First match wins. \1, \2, etc. refer to capture groups.
# ---------------------------------------------------------------------------

# ── LLaMA / Mistral / Qwen2 / DeepSeek-V2 dense layers ───────────────────
_LLAMA_TENSOR_MAP: List[Tuple[str, str]] = [
    (r"^model\.embed_tokens\.weight$",                                "token_embd.weight"),
    (r"^model\.layers\.(\d+)\.self_attn\.q_proj\.weight$",           r"blk.\1.attn_q.weight"),
    (r"^model\.layers\.(\d+)\.self_attn\.k_proj\.weight$",           r"blk.\1.attn_k.weight"),
    (r"^model\.layers\.(\d+)\.self_attn\.v_proj\.weight$",           r"blk.\1.attn_v.weight"),
    (r"^model\.layers\.(\d+)\.self_attn\.o_proj\.weight$",           r"blk.\1.attn_output.weight"),
    (r"^model\.layers\.(\d+)\.mlp\.gate_proj\.weight$",              r"blk.\1.ffn_gate.weight"),
    (r"^model\.layers\.(\d+)\.mlp\.up_proj\.weight$",                r"blk.\1.ffn_up.weight"),
    (r"^model\.layers\.(\d+)\.mlp\.down_proj\.weight$",              r"blk.\1.ffn_down.weight"),
    (r"^model\.layers\.(\d+)\.input_layernorm\.weight$",             r"blk.\1.attn_norm.weight"),
    (r"^model\.layers\.(\d+)\.post_attention_layernorm\.weight$",    r"blk.\1.ffn_norm.weight"),
    (r"^model\.norm\.weight$",                                        "output_norm.weight"),
    (r"^lm_head\.weight$",                                            "output.weight"),
]

# ── DeepSeek-V2: MLA attention + MoE experts (prepend before LLaMA patterns)
_DEEPSEEK_EXTRA: List[Tuple[str, str]] = [
    (r"^model\.layers\.(\d+)\.self_attn\.q_a_proj\.weight$",                r"blk.\1.attn_q_a.weight"),
    (r"^model\.layers\.(\d+)\.self_attn\.q_b_proj\.weight$",                r"blk.\1.attn_q_b.weight"),
    (r"^model\.layers\.(\d+)\.self_attn\.kv_a_proj_with_mqa\.weight$",      r"blk.\1.attn_kv_a.weight"),
    (r"^model\.layers\.(\d+)\.self_attn\.kv_b_proj\.weight$",               r"blk.\1.attn_kv_b.weight"),
    (r"^model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.gate_proj\.weight$",     r"blk.\1.ffn_gate.\2.weight"),
    (r"^model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.up_proj\.weight$",       r"blk.\1.ffn_up.\2.weight"),
    (r"^model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.down_proj\.weight$",     r"blk.\1.ffn_down.\2.weight"),
]

_DEEPSEEK_TENSOR_MAP: List[Tuple[str, str]] = _DEEPSEEK_EXTRA + _LLAMA_TENSOR_MAP

# ── GPT-2 ─────────────────────────────────────────────────────────────────
_GPT2_TENSOR_MAP: List[Tuple[str, str]] = [
    (r"^transformer\.wte\.weight$",                                   "token_embd.weight"),
    (r"^transformer\.h\.(\d+)\.attn\.c_attn\.(weight|bias)$",        r"blk.\1.attn_qkv.\2"),
    (r"^transformer\.h\.(\d+)\.attn\.c_proj\.(weight|bias)$",        r"blk.\1.attn_output.\2"),
    (r"^transformer\.h\.(\d+)\.mlp\.c_fc\.(weight|bias)$",           r"blk.\1.ffn_up.\2"),
    (r"^transformer\.h\.(\d+)\.mlp\.c_proj\.(weight|bias)$",         r"blk.\1.ffn_down.\2"),
    (r"^transformer\.h\.(\d+)\.ln_1\.(weight|bias)$",                r"blk.\1.attn_norm.\2"),
    (r"^transformer\.h\.(\d+)\.ln_2\.(weight|bias)$",                r"blk.\1.ffn_norm.\2"),
    (r"^transformer\.ln_f\.(weight|bias)$",                           r"output_norm.\1"),
    (r"^lm_head\.(weight|bias)$",                                     r"output.\1"),
]

# ── OPT ───────────────────────────────────────────────────────────────────
# NOTE: llama.cpp has no OPT model loader. GGUF will be structurally valid
# but llama.cpp cannot run inference on it. Tensor names follow the standard
# blk.N.* convention in anticipation of future llama.cpp OPT support.
_OPT_TENSOR_MAP: List[Tuple[str, str]] = [
    (r"^model\.decoder\.embed_tokens\.(weight|bias)$",               r"token_embd.\1"),
    (r"^model\.decoder\.embed_positions\.(weight|bias)$",            r"position_embd.\1"),
    (r"^model\.decoder\.final_layer_norm\.(weight|bias)$",           r"output_norm.\1"),
    (r"^model\.decoder\.layers\.(\d+)\.self_attn\.q_proj\.(weight|bias)$",        r"blk.\1.attn_q.\2"),
    (r"^model\.decoder\.layers\.(\d+)\.self_attn\.k_proj\.(weight|bias)$",        r"blk.\1.attn_k.\2"),
    (r"^model\.decoder\.layers\.(\d+)\.self_attn\.v_proj\.(weight|bias)$",        r"blk.\1.attn_v.\2"),
    (r"^model\.decoder\.layers\.(\d+)\.self_attn\.out_proj\.(weight|bias)$",      r"blk.\1.attn_output.\2"),
    (r"^model\.decoder\.layers\.(\d+)\.self_attn_layer_norm\.(weight|bias)$",     r"blk.\1.attn_norm.\2"),
    (r"^model\.decoder\.layers\.(\d+)\.fc1\.(weight|bias)$",                      r"blk.\1.ffn_up.\2"),
    (r"^model\.decoder\.layers\.(\d+)\.fc2\.(weight|bias)$",                      r"blk.\1.ffn_down.\2"),
    (r"^model\.decoder\.layers\.(\d+)\.final_layer_norm\.(weight|bias)$",         r"blk.\1.ffn_norm.\2"),
    (r"^lm_head\.(weight|bias)$",                                                  r"output.\1"),
]

# ── Phi-2 (microsoft/phi-2) ───────────────────────────────────────────────
# Two weight layouts exist in the wild:
#   "new" layout:  model.layers.N.*  (most HF checkpoints)
#   "old" layout:  transformer.h.N.* (Phi-1, Phi-1.5, early Phi-2)
# Both are included; patterns are distinct and do not conflict.
_PHI2_TENSOR_MAP: List[Tuple[str, str]] = [
    # New layout (model.layers.*)
    (r"^model\.embed_tokens\.(weight|bias)$",                          r"token_embd.\1"),
    (r"^model\.layers\.(\d+)\.input_layernorm\.(weight|bias)$",        r"blk.\1.attn_norm.\2"),
    (r"^model\.layers\.(\d+)\.self_attn\.q_proj\.(weight|bias)$",      r"blk.\1.attn_q.\2"),
    (r"^model\.layers\.(\d+)\.self_attn\.k_proj\.(weight|bias)$",      r"blk.\1.attn_k.\2"),
    (r"^model\.layers\.(\d+)\.self_attn\.v_proj\.(weight|bias)$",      r"blk.\1.attn_v.\2"),
    (r"^model\.layers\.(\d+)\.self_attn\.dense\.(weight|bias)$",       r"blk.\1.attn_output.\2"),
    (r"^model\.layers\.(\d+)\.post_layernorm\.(weight|bias)$",         r"blk.\1.ffn_norm.\2"),
    (r"^model\.layers\.(\d+)\.mlp\.fc1\.(weight|bias)$",               r"blk.\1.ffn_up.\2"),
    (r"^model\.layers\.(\d+)\.mlp\.fc2\.(weight|bias)$",               r"blk.\1.ffn_down.\2"),
    (r"^model\.final_layernorm\.(weight|bias)$",                        r"output_norm.\1"),
    (r"^lm_head\.linear\.(weight|bias)$",                               r"output.\1"),
    # Old layout (transformer.h.*) — Phi-1 / Phi-1.5 / early Phi-2
    (r"^transformer\.embd\.wte\.(weight|bias)$",                        r"token_embd.\1"),
    (r"^transformer\.h\.(\d+)\.ln\.(weight|bias)$",                     r"blk.\1.attn_norm.\2"),
    (r"^transformer\.h\.(\d+)\.mixer\.Wqkv\.(weight|bias)$",            r"blk.\1.attn_qkv.\2"),
    (r"^transformer\.h\.(\d+)\.mixer\.out_proj\.(weight|bias)$",        r"blk.\1.attn_output.\2"),
    (r"^transformer\.h\.(\d+)\.mlp\.fc1\.(weight|bias)$",               r"blk.\1.ffn_up.\2"),
    (r"^transformer\.h\.(\d+)\.mlp\.fc2\.(weight|bias)$",               r"blk.\1.ffn_down.\2"),
    (r"^lm_head\.ln\.(weight|bias)$",                                    r"output_norm.\1"),
    # lm_head.linear matches both layouts
]

# ── Phi-3 / Phi-3.5 / Phi-4 ───────────────────────────────────────────────
# Uses fused qkv_proj and fused gate_up_proj.
# gate_up_proj is a single tensor (gate + up interleaved); stored as
# blk.N.ffn_up.weight — llama.cpp splits it at inference time.
_PHI3_TENSOR_MAP: List[Tuple[str, str]] = [
    (r"^model\.embed_tokens\.(weight|bias)$",                              r"token_embd.\1"),
    (r"^model\.layers\.(\d+)\.input_layernorm\.(weight|bias)$",            r"blk.\1.attn_norm.\2"),
    (r"^model\.layers\.(\d+)\.self_attn\.qkv_proj\.(weight|bias)$",        r"blk.\1.attn_qkv.\2"),
    (r"^model\.layers\.(\d+)\.self_attn\.o_proj\.(weight|bias)$",          r"blk.\1.attn_output.\2"),
    (r"^model\.layers\.(\d+)\.post_attention_layernorm\.(weight|bias)$",   r"blk.\1.ffn_norm.\2"),
    (r"^model\.layers\.(\d+)\.mlp\.gate_up_proj\.(weight|bias)$",          r"blk.\1.ffn_up.\2"),
    (r"^model\.layers\.(\d+)\.mlp\.down_proj\.(weight|bias)$",             r"blk.\1.ffn_down.\2"),
    (r"^model\.norm\.(weight|bias)$",                                       r"output_norm.\1"),
    (r"^lm_head\.(weight|bias)$",                                           r"output.\1"),
]

# ── ChatGLM-2 / ChatGLM-3 / GLM-4 ────────────────────────────────────────
# HF tensor names include the full "transformer." prefix.
# Uses fused query_key_value and encoder.layers (not model.layers).
_CHATGLM_TENSOR_MAP: List[Tuple[str, str]] = [
    (r"^transformer\.embedding\.word_embeddings\.(weight|bias)$",                           r"token_embd.\1"),
    (r"^transformer\.encoder\.final_layernorm\.(weight|bias)$",                             r"output_norm.\1"),
    (r"^transformer\.output_layer\.(weight|bias)$",                                         r"output.\1"),
    (r"^transformer\.encoder\.layers\.(\d+)\.input_layernorm\.(weight|bias)$",              r"blk.\1.attn_norm.\2"),
    (r"^transformer\.encoder\.layers\.(\d+)\.self_attention\.query_key_value\.(weight|bias)$", r"blk.\1.attn_qkv.\2"),
    (r"^transformer\.encoder\.layers\.(\d+)\.self_attention\.dense\.(weight|bias)$",        r"blk.\1.attn_output.\2"),
    (r"^transformer\.encoder\.layers\.(\d+)\.post_attention_layernorm\.(weight|bias)$",     r"blk.\1.ffn_norm.\2"),
    (r"^transformer\.encoder\.layers\.(\d+)\.mlp\.dense_h_to_4h\.(weight|bias)$",          r"blk.\1.ffn_up.\2"),
    (r"^transformer\.encoder\.layers\.(\d+)\.mlp\.dense_4h_to_h\.(weight|bias)$",          r"blk.\1.ffn_down.\2"),
]

# ── Falcon ────────────────────────────────────────────────────────────────
# Falcon-7B uses a single input_layernorm.
# Falcon-40B/180B use separate ln_attn + ln_mlp (both patterns included).
_FALCON_TENSOR_MAP: List[Tuple[str, str]] = [
    (r"^transformer\.word_embeddings\.(weight|bias)$",                              r"token_embd.\1"),
    (r"^transformer\.ln_f\.(weight|bias)$",                                         r"output_norm.\1"),
    (r"^lm_head\.(weight|bias)$",                                                   r"output.\1"),
    # Falcon-7B: single input_layernorm → attn_norm
    (r"^transformer\.h\.(\d+)\.input_layernorm\.(weight|bias)$",                    r"blk.\1.attn_norm.\2"),
    # Falcon-40B/180B: separate attn + mlp norms
    (r"^transformer\.h\.(\d+)\.ln_attn\.(weight|bias)$",                            r"blk.\1.attn_norm.\2"),
    (r"^transformer\.h\.(\d+)\.ln_mlp\.(weight|bias)$",                             r"blk.\1.attn_norm_2.\2"),
    # Attention (fused QKV)
    (r"^transformer\.h\.(\d+)\.self_attention\.query_key_value\.(weight|bias)$",    r"blk.\1.attn_qkv.\2"),
    (r"^transformer\.h\.(\d+)\.self_attention\.dense\.(weight|bias)$",              r"blk.\1.attn_output.\2"),
    # FFN
    (r"^transformer\.h\.(\d+)\.mlp\.dense_h_to_4h\.(weight|bias)$",                r"blk.\1.ffn_up.\2"),
    (r"^transformer\.h\.(\d+)\.mlp\.dense_4h_to_h\.(weight|bias)$",                r"blk.\1.ffn_down.\2"),
]

# ── Gemma (1st generation) ────────────────────────────────────────────────
# lm_head.weight is tied to token_embd.weight — not exported as a separate tensor.
# NOTE: llama.cpp expects norm weights stored as (w + 1). If loading fails with
# incorrect outputs, re-export after adding 1.0 to all norm weight tensors.
_GEMMA_TENSOR_MAP: List[Tuple[str, str]] = [
    (r"^model\.embed_tokens\.weight$",                                "token_embd.weight"),
    (r"^model\.norm\.weight$",                                        "output_norm.weight"),
    (r"^model\.layers\.(\d+)\.input_layernorm\.weight$",             r"blk.\1.attn_norm.weight"),
    (r"^model\.layers\.(\d+)\.self_attn\.q_proj\.weight$",           r"blk.\1.attn_q.weight"),
    (r"^model\.layers\.(\d+)\.self_attn\.k_proj\.weight$",           r"blk.\1.attn_k.weight"),
    (r"^model\.layers\.(\d+)\.self_attn\.v_proj\.weight$",           r"blk.\1.attn_v.weight"),
    (r"^model\.layers\.(\d+)\.self_attn\.o_proj\.weight$",           r"blk.\1.attn_output.weight"),
    (r"^model\.layers\.(\d+)\.post_attention_layernorm\.weight$",    r"blk.\1.ffn_norm.weight"),
    (r"^model\.layers\.(\d+)\.mlp\.gate_proj\.weight$",              r"blk.\1.ffn_gate.weight"),
    (r"^model\.layers\.(\d+)\.mlp\.up_proj\.weight$",                r"blk.\1.ffn_up.weight"),
    (r"^model\.layers\.(\d+)\.mlp\.down_proj\.weight$",              r"blk.\1.ffn_down.weight"),
]

# ── Gemma 2 ───────────────────────────────────────────────────────────────
# Adds pre/post feedforward norms and post-attention norm on top of Gemma base.
_GEMMA2_EXTRA: List[Tuple[str, str]] = [
    (r"^model\.layers\.(\d+)\.post_attention_layernorm\.weight$",      r"blk.\1.post_attention_norm.weight"),
    (r"^model\.layers\.(\d+)\.pre_feedforward_layernorm\.weight$",     r"blk.\1.ffn_norm.weight"),
    (r"^model\.layers\.(\d+)\.post_feedforward_layernorm\.weight$",    r"blk.\1.post_ffw_norm.weight"),
]

_GEMMA2_TENSOR_MAP: List[Tuple[str, str]] = _GEMMA2_EXTRA + _GEMMA_TENSOR_MAP

# ── StarCoder-1 / SantaCoder (GPTBigCode) ─────────────────────────────────
# GPT-2 style names with position embeddings and fused c_attn QKV.
_STARCODER_TENSOR_MAP: List[Tuple[str, str]] = [
    (r"^transformer\.wte\.(weight|bias)$",                             r"token_embd.\1"),
    (r"^transformer\.wpe\.(weight|bias)$",                             r"position_embd.\1"),
    (r"^transformer\.ln_f\.(weight|bias)$",                            r"output_norm.\1"),
    (r"^lm_head\.(weight|bias)$",                                      r"output.\1"),
    (r"^transformer\.h\.(\d+)\.ln_1\.(weight|bias)$",                  r"blk.\1.attn_norm.\2"),
    (r"^transformer\.h\.(\d+)\.attn\.c_attn\.(weight|bias)$",          r"blk.\1.attn_qkv.\2"),
    (r"^transformer\.h\.(\d+)\.attn\.c_proj\.(weight|bias)$",          r"blk.\1.attn_output.\2"),
    (r"^transformer\.h\.(\d+)\.ln_2\.(weight|bias)$",                  r"blk.\1.ffn_norm.\2"),
    (r"^transformer\.h\.(\d+)\.mlp\.c_fc\.(weight|bias)$",             r"blk.\1.ffn_up.\2"),
    (r"^transformer\.h\.(\d+)\.mlp\.c_proj\.(weight|bias)$",           r"blk.\1.ffn_down.\2"),
]

# ── StarCoder-2 ───────────────────────────────────────────────────────────
# LLaMA-style model.layers with separate Q/K/V and c_fc/c_proj FFN names.
_STARCODER2_TENSOR_MAP: List[Tuple[str, str]] = [
    (r"^model\.embed_tokens\.(weight|bias)$",                              r"token_embd.\1"),
    (r"^model\.norm\.(weight|bias)$",                                      r"output_norm.\1"),
    (r"^lm_head\.(weight|bias)$",                                          r"output.\1"),
    (r"^model\.layers\.(\d+)\.input_layernorm\.(weight|bias)$",            r"blk.\1.attn_norm.\2"),
    (r"^model\.layers\.(\d+)\.self_attn\.q_proj\.(weight|bias)$",          r"blk.\1.attn_q.\2"),
    (r"^model\.layers\.(\d+)\.self_attn\.k_proj\.(weight|bias)$",          r"blk.\1.attn_k.\2"),
    (r"^model\.layers\.(\d+)\.self_attn\.v_proj\.(weight|bias)$",          r"blk.\1.attn_v.\2"),
    (r"^model\.layers\.(\d+)\.self_attn\.o_proj\.(weight|bias)$",          r"blk.\1.attn_output.\2"),
    (r"^model\.layers\.(\d+)\.post_attention_layernorm\.(weight|bias)$",   r"blk.\1.ffn_norm.\2"),
    (r"^model\.layers\.(\d+)\.mlp\.c_fc\.(weight|bias)$",                  r"blk.\1.ffn_up.\2"),
    (r"^model\.layers\.(\d+)\.mlp\.c_proj\.(weight|bias)$",                r"blk.\1.ffn_down.\2"),
]

# ── BLOOM / BLOOMZ ────────────────────────────────────────────────────────
# Full "transformer." prefix is part of the PyTorch tensor names.
# The word_embeddings_layernorm is a post-embedding norm unique to BLOOM.
_BLOOM_TENSOR_MAP: List[Tuple[str, str]] = [
    (r"^transformer\.word_embeddings\.(weight|bias)$",                              r"token_embd.\1"),
    (r"^transformer\.word_embeddings_layernorm\.(weight|bias)$",                    r"token_embd_norm.\1"),
    (r"^transformer\.ln_f\.(weight|bias)$",                                         r"output_norm.\1"),
    (r"^lm_head\.(weight|bias)$",                                                   r"output.\1"),
    (r"^transformer\.h\.(\d+)\.input_layernorm\.(weight|bias)$",                    r"blk.\1.attn_norm.\2"),
    (r"^transformer\.h\.(\d+)\.self_attention\.query_key_value\.(weight|bias)$",    r"blk.\1.attn_qkv.\2"),
    (r"^transformer\.h\.(\d+)\.self_attention\.dense\.(weight|bias)$",              r"blk.\1.attn_output.\2"),
    (r"^transformer\.h\.(\d+)\.post_attention_layernorm\.(weight|bias)$",           r"blk.\1.ffn_norm.\2"),
    (r"^transformer\.h\.(\d+)\.mlp\.dense_h_to_4h\.(weight|bias)$",                r"blk.\1.ffn_up.\2"),
    (r"^transformer\.h\.(\d+)\.mlp\.dense_4h_to_h\.(weight|bias)$",                r"blk.\1.ffn_down.\2"),
]

# ── MPT ───────────────────────────────────────────────────────────────────
# Uses transformer.blocks.N (not transformer.h.N or model.layers.N).
_MPT_TENSOR_MAP: List[Tuple[str, str]] = [
    (r"^transformer\.wte\.(weight|bias)$",                                   r"token_embd.\1"),
    (r"^transformer\.norm_f\.(weight|bias)$",                                r"output_norm.\1"),
    (r"^lm_head\.(weight|bias)$",                                            r"output.\1"),
    (r"^transformer\.blocks\.(\d+)\.norm_1\.(weight|bias)$",                 r"blk.\1.attn_norm.\2"),
    (r"^transformer\.blocks\.(\d+)\.attn\.Wqkv\.(weight|bias)$",             r"blk.\1.attn_qkv.\2"),
    (r"^transformer\.blocks\.(\d+)\.attn\.out_proj\.(weight|bias)$",         r"blk.\1.attn_output.\2"),
    (r"^transformer\.blocks\.(\d+)\.attn\.q_ln\.(weight|bias)$",             r"blk.\1.attn_q_norm.\2"),
    (r"^transformer\.blocks\.(\d+)\.attn\.k_ln\.(weight|bias)$",             r"blk.\1.attn_k_norm.\2"),
    (r"^transformer\.blocks\.(\d+)\.norm_2\.(weight|bias)$",                 r"blk.\1.ffn_norm.\2"),
    (r"^transformer\.blocks\.(\d+)\.ffn\.up_proj\.(weight|bias)$",           r"blk.\1.ffn_up.\2"),
    (r"^transformer\.blocks\.(\d+)\.ffn\.down_proj\.(weight|bias)$",         r"blk.\1.ffn_down.\2"),
]

# ── Command-R / Command-R+ (Cohere) ──────────────────────────────────────
# LLaMA-style tensor names plus per-head Q/K norms.
# lm_head.weight is tied to token_embd.weight — not exported separately.
_COMMAND_R_TENSOR_MAP: List[Tuple[str, str]] = [
    (r"^model\.embed_tokens\.(weight|bias)$",                              r"token_embd.\1"),
    (r"^model\.norm\.(weight|bias)$",                                      r"output_norm.\1"),
    (r"^model\.layers\.(\d+)\.input_layernorm\.(weight|bias)$",            r"blk.\1.attn_norm.\2"),
    (r"^model\.layers\.(\d+)\.self_attn\.q_proj\.(weight|bias)$",          r"blk.\1.attn_q.\2"),
    (r"^model\.layers\.(\d+)\.self_attn\.k_proj\.(weight|bias)$",          r"blk.\1.attn_k.\2"),
    (r"^model\.layers\.(\d+)\.self_attn\.v_proj\.(weight|bias)$",          r"blk.\1.attn_v.\2"),
    (r"^model\.layers\.(\d+)\.self_attn\.o_proj\.(weight|bias)$",          r"blk.\1.attn_output.\2"),
    (r"^model\.layers\.(\d+)\.self_attn\.q_norm\.(weight|bias)$",          r"blk.\1.attn_q_norm.\2"),
    (r"^model\.layers\.(\d+)\.self_attn\.k_norm\.(weight|bias)$",          r"blk.\1.attn_k_norm.\2"),
    (r"^model\.layers\.(\d+)\.mlp\.gate_proj\.(weight|bias)$",             r"blk.\1.ffn_gate.\2"),
    (r"^model\.layers\.(\d+)\.mlp\.up_proj\.(weight|bias)$",               r"blk.\1.ffn_up.\2"),
    (r"^model\.layers\.(\d+)\.mlp\.down_proj\.(weight|bias)$",             r"blk.\1.ffn_down.\2"),
]

# ---------------------------------------------------------------------------
# Master dispatch table
# ---------------------------------------------------------------------------

_ARCH_TENSOR_MAPS: Dict[str, List[Tuple[str, str]]] = {
    "llama":       _LLAMA_TENSOR_MAP,
    "qwen2":       _LLAMA_TENSOR_MAP,       # same HF tensor names as LLaMA
    "deepseek2":   _DEEPSEEK_TENSOR_MAP,
    "gpt2":        _GPT2_TENSOR_MAP,
    "opt":         _OPT_TENSOR_MAP,
    "phi2":        _PHI2_TENSOR_MAP,
    "phi3":        _PHI3_TENSOR_MAP,
    "chatglm":     _CHATGLM_TENSOR_MAP,
    "falcon":      _FALCON_TENSOR_MAP,
    "gemma":       _GEMMA_TENSOR_MAP,
    "gemma2":      _GEMMA2_TENSOR_MAP,
    "starcoder":   _STARCODER_TENSOR_MAP,
    "starcoder2":  _STARCODER2_TENSOR_MAP,
    "bloom":       _BLOOM_TENSOR_MAP,
    "mpt":         _MPT_TENSOR_MAP,
    "command-r":   _COMMAND_R_TENSOR_MAP,
}

# Counter used for generic sequential naming
_generic_layer_counter: Dict[str, int] = {}


def detect_architecture(hf_model_type: Optional[str]) -> str:
    """Map a HuggingFace model_type string to a GGUF architecture string.

    Args:
        hf_model_type: Value of 'model_type' from HF config.json, or None.

    Returns:
        GGUF architecture string (e.g. "llama", "phi3", "generic").
    """
    if hf_model_type is None:
        return "generic"
    return _HF_TYPE_TO_ARCH.get(hf_model_type.lower(), "generic")


def map_tensor_name(pytorch_name: str, architecture: str) -> str:
    """Return the GGUF tensor name for a given PyTorch module name.

    Args:
        pytorch_name: Full dotted module name as from model.named_modules()
                      plus ".weight" or ".bias" suffix.
        architecture: GGUF architecture string (e.g. "llama", "phi3", "generic").

    Returns:
        GGUF tensor name string.  Falls back to "blk.N.weight" sequential naming
        for unknown architectures or unmatched names, with a warning.
    """
    patterns = _ARCH_TENSOR_MAPS.get(architecture)
    if patterns is not None:
        for pattern, template in patterns:
            m = re.match(pattern, pytorch_name)
            if m:
                return re.sub(pattern, template, pytorch_name)

    # Generic / unmatched — emit warning on first occurrence only
    if architecture != "generic":
        warnings.warn(
            f"Tensor '{pytorch_name}' did not match any pattern for architecture "
            f"'{architecture}'. Using generic sequential name. "
            "llama.cpp inference may not work correctly.",
            stacklevel=3,
        )
    idx = _generic_layer_counter.get(architecture, 0)
    _generic_layer_counter[architecture] = idx + 1
    return f"blk.{idx}.weight"


def reset_generic_counter(architecture: str = "generic") -> None:
    """Reset the sequential naming counter (used between exports and in tests)."""
    _generic_layer_counter.pop(architecture, None)


# ---------------------------------------------------------------------------
# HF config.json → GGUF KV mapping
# ---------------------------------------------------------------------------

# Each entry: (hf_key, gguf_key_template, value_type)
# {arch} in gguf_key_template is replaced with the GGUF architecture string.
# value_type: "uint32", "float32", "string"

_LLAMA_CONFIG_MAP: List[Tuple[str, str, str]] = [
    ("max_position_embeddings", "{arch}.context_length",                   "uint32"),
    ("hidden_size",             "{arch}.embedding_length",                 "uint32"),
    ("num_hidden_layers",       "{arch}.block_count",                      "uint32"),
    ("num_attention_heads",     "{arch}.attention.head_count",             "uint32"),
    ("num_key_value_heads",     "{arch}.attention.head_count_kv",          "uint32"),
    ("intermediate_size",       "{arch}.feed_forward_length",              "uint32"),
    ("rope_theta",              "{arch}.rope.freq_base",                   "float32"),
    ("rms_norm_eps",            "{arch}.attention.layer_norm_rms_epsilon", "float32"),
    ("vocab_size",              "{arch}.vocab_size",                       "uint32"),
]

_GPT2_CONFIG_MAP: List[Tuple[str, str, str]] = [
    ("n_embd",        "{arch}.embedding_length",    "uint32"),
    ("n_layer",       "{arch}.block_count",         "uint32"),
    ("n_head",        "{arch}.attention.head_count","uint32"),
    ("n_positions",   "{arch}.context_length",      "uint32"),
]

_OPT_CONFIG_MAP: List[Tuple[str, str, str]] = [
    ("max_position_embeddings", "{arch}.context_length",        "uint32"),
    ("hidden_size",             "{arch}.embedding_length",      "uint32"),
    ("num_hidden_layers",       "{arch}.block_count",           "uint32"),
    ("num_attention_heads",     "{arch}.attention.head_count",  "uint32"),
    ("ffn_dim",                 "{arch}.feed_forward_length",   "uint32"),
    ("vocab_size",              "{arch}.vocab_size",            "uint32"),
]

_PHI2_CONFIG_MAP: List[Tuple[str, str, str]] = [
    ("max_position_embeddings", "{arch}.context_length",                   "uint32"),
    ("hidden_size",             "{arch}.embedding_length",                 "uint32"),
    ("num_hidden_layers",       "{arch}.block_count",                      "uint32"),
    ("num_attention_heads",     "{arch}.attention.head_count",             "uint32"),
    ("num_key_value_heads",     "{arch}.attention.head_count_kv",          "uint32"),
    ("intermediate_size",       "{arch}.feed_forward_length",              "uint32"),
    ("layer_norm_eps",          "{arch}.attention.layer_norm_epsilon",     "float32"),
    ("vocab_size",              "{arch}.vocab_size",                       "uint32"),
]

_PHI3_CONFIG_MAP: List[Tuple[str, str, str]] = [
    ("max_position_embeddings", "{arch}.context_length",                   "uint32"),
    ("hidden_size",             "{arch}.embedding_length",                 "uint32"),
    ("num_hidden_layers",       "{arch}.block_count",                      "uint32"),
    ("num_attention_heads",     "{arch}.attention.head_count",             "uint32"),
    ("num_key_value_heads",     "{arch}.attention.head_count_kv",          "uint32"),
    ("intermediate_size",       "{arch}.feed_forward_length",              "uint32"),
    ("rms_norm_eps",            "{arch}.attention.layer_norm_rms_epsilon", "float32"),
    ("rope_theta",              "{arch}.rope.freq_base",                   "float32"),
    ("vocab_size",              "{arch}.vocab_size",                       "uint32"),
]

_CHATGLM_CONFIG_MAP: List[Tuple[str, str, str]] = [
    ("seq_length",              "{arch}.context_length",        "uint32"),
    ("hidden_size",             "{arch}.embedding_length",      "uint32"),
    ("num_layers",              "{arch}.block_count",           "uint32"),
    ("num_attention_heads",     "{arch}.attention.head_count",  "uint32"),
    ("multi_query_group_num",   "{arch}.attention.head_count_kv", "uint32"),
    ("ffn_hidden_size",         "{arch}.feed_forward_length",   "uint32"),
    ("vocab_size",              "{arch}.vocab_size",            "uint32"),
]

_FALCON_CONFIG_MAP: List[Tuple[str, str, str]] = [
    ("hidden_size",         "{arch}.embedding_length",          "uint32"),
    ("num_hidden_layers",   "{arch}.block_count",               "uint32"),
    ("num_attention_heads", "{arch}.attention.head_count",      "uint32"),
    ("num_kv_heads",        "{arch}.attention.head_count_kv",   "uint32"),
    ("vocab_size",          "{arch}.vocab_size",                "uint32"),
]

_GEMMA_CONFIG_MAP: List[Tuple[str, str, str]] = [
    ("max_position_embeddings", "{arch}.context_length",                   "uint32"),
    ("hidden_size",             "{arch}.embedding_length",                 "uint32"),
    ("num_hidden_layers",       "{arch}.block_count",                      "uint32"),
    ("num_attention_heads",     "{arch}.attention.head_count",             "uint32"),
    ("num_key_value_heads",     "{arch}.attention.head_count_kv",          "uint32"),
    ("intermediate_size",       "{arch}.feed_forward_length",              "uint32"),
    ("head_dim",                "{arch}.attention.key_length",             "uint32"),
    ("rms_norm_eps",            "{arch}.attention.layer_norm_rms_epsilon", "float32"),
    ("vocab_size",              "{arch}.vocab_size",                       "uint32"),
]

_STARCODER_CONFIG_MAP: List[Tuple[str, str, str]] = [
    ("n_embd",              "{arch}.embedding_length",          "uint32"),
    ("n_layer",             "{arch}.block_count",               "uint32"),
    ("n_head",              "{arch}.attention.head_count",      "uint32"),
    ("n_inner",             "{arch}.feed_forward_length",       "uint32"),
    ("n_positions",         "{arch}.context_length",            "uint32"),
    ("layer_norm_epsilon",  "{arch}.attention.layer_norm_epsilon", "float32"),
    ("vocab_size",          "{arch}.vocab_size",                "uint32"),
]

_STARCODER2_CONFIG_MAP: List[Tuple[str, str, str]] = [
    ("max_position_embeddings", "{arch}.context_length",                   "uint32"),
    ("hidden_size",             "{arch}.embedding_length",                 "uint32"),
    ("num_hidden_layers",       "{arch}.block_count",                      "uint32"),
    ("num_attention_heads",     "{arch}.attention.head_count",             "uint32"),
    ("num_key_value_heads",     "{arch}.attention.head_count_kv",          "uint32"),
    ("intermediate_size",       "{arch}.feed_forward_length",              "uint32"),
    ("norm_epsilon",            "{arch}.attention.layer_norm_epsilon",     "float32"),
    ("rope_theta",              "{arch}.rope.freq_base",                   "float32"),
    ("vocab_size",              "{arch}.vocab_size",                       "uint32"),
]

_BLOOM_CONFIG_MAP: List[Tuple[str, str, str]] = [
    ("hidden_size",         "{arch}.embedding_length",              "uint32"),
    ("n_layer",             "{arch}.block_count",                   "uint32"),
    ("n_head",              "{arch}.attention.head_count",          "uint32"),
    ("layer_norm_epsilon",  "{arch}.attention.layer_norm_epsilon",  "float32"),
    ("vocab_size",          "{arch}.vocab_size",                    "uint32"),
]

_MPT_CONFIG_MAP: List[Tuple[str, str, str]] = [
    ("max_seq_len",     "{arch}.context_length",        "uint32"),
    ("d_model",         "{arch}.embedding_length",      "uint32"),
    ("n_layers",        "{arch}.block_count",           "uint32"),
    ("n_heads",         "{arch}.attention.head_count",  "uint32"),
    ("vocab_size",      "{arch}.vocab_size",            "uint32"),
]

_COMMAND_R_CONFIG_MAP: List[Tuple[str, str, str]] = [
    ("max_position_embeddings", "{arch}.context_length",                   "uint32"),
    ("hidden_size",             "{arch}.embedding_length",                 "uint32"),
    ("num_hidden_layers",       "{arch}.block_count",                      "uint32"),
    ("num_attention_heads",     "{arch}.attention.head_count",             "uint32"),
    ("num_key_value_heads",     "{arch}.attention.head_count_kv",          "uint32"),
    ("intermediate_size",       "{arch}.feed_forward_length",              "uint32"),
    ("layer_norm_eps",          "{arch}.attention.layer_norm_rms_epsilon", "float32"),
    ("rope_theta",              "{arch}.rope.freq_base",                   "float32"),
    ("vocab_size",              "{arch}.vocab_size",                       "uint32"),
]

_ARCH_CONFIG_MAPS: Dict[str, List[Tuple[str, str, str]]] = {
    "llama":      _LLAMA_CONFIG_MAP,
    "qwen2":      _LLAMA_CONFIG_MAP,
    "deepseek2":  _LLAMA_CONFIG_MAP,
    "gpt2":       _GPT2_CONFIG_MAP,
    "opt":        _OPT_CONFIG_MAP,
    "phi2":       _PHI2_CONFIG_MAP,
    "phi3":       _PHI3_CONFIG_MAP,
    "chatglm":    _CHATGLM_CONFIG_MAP,
    "falcon":     _FALCON_CONFIG_MAP,
    "gemma":      _GEMMA_CONFIG_MAP,
    "gemma2":     _GEMMA_CONFIG_MAP,    # same keys as Gemma
    "starcoder":  _STARCODER_CONFIG_MAP,
    "starcoder2": _STARCODER2_CONFIG_MAP,
    "bloom":      _BLOOM_CONFIG_MAP,
    "mpt":        _MPT_CONFIG_MAP,
    "command-r":  _COMMAND_R_CONFIG_MAP,
}


def build_kv_entries(
    config: Dict[str, Any],
    architecture: str,
    quantization_type: str,
    model_name: str = "mono-quant-export",
) -> List[Tuple[str, str, Any]]:
    """Build a list of GGUF KV entries from a merged config dict.

    Args:
        config:            Merged HF config dict (model_params overrides config.json).
        architecture:      GGUF architecture string.
        quantization_type: Quantization type string (e.g. "q4_k_s").
        model_name:        Human-readable model name for general.name.

    Returns:
        List of (gguf_key, value_type, value) tuples.
        value_type is one of: "string", "uint32", "float32".
    """
    entries: List[Tuple[str, str, Any]] = []

    # Standard general.* keys
    entries.append(("general.architecture",          "string", architecture))
    entries.append(("general.name",                  "string", model_name))
    entries.append(("general.quantization_version",  "uint32", 2))

    if quantization_type == "q4_k_s":
        entries.append(("general.file_type", "uint32", GGML_FTYPE_MOSTLY_Q4_K_S))

    # Architecture-specific keys
    arch_map = _ARCH_CONFIG_MAPS.get(architecture, [])
    for hf_key, gguf_key_tpl, val_type in arch_map:
        gguf_key = gguf_key_tpl.replace("{arch}", architecture)
        if hf_key in config:
            entries.append((gguf_key, val_type, config[hf_key]))
        else:
            warnings.warn(
                f"Missing config key '{hf_key}' for GGUF KV entry '{gguf_key}'. "
                "Skipping. Pass it via config_path or model_params.",
                stacklevel=4,
            )

    return entries
