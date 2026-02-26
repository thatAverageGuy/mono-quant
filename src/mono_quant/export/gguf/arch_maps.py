"""Architecture-specific tensor name mappings and HF config → GGUF KV mappings.

Supported architectures:
    llama     — LLaMA, Mistral, Qwen2, DeepSeek-V2 (dense layers)
    qwen2     — Qwen2 (same HF tensor names as LLaMA; different GGUF arch string)
    deepseek_v2 — DeepSeek-V2 (MLA attention tensors + dense MLP)
    gpt2      — GPT-2
    generic   — Unknown architecture; sequential blk.N.weight_M fallback

GGUF KV file_type values:
    GGML_FTYPE_MOSTLY_Q4_K_S = 14
"""

import re
import warnings
from typing import Any, Dict, List, Optional, Tuple

# HF model_type → GGUF architecture string
_HF_TYPE_TO_ARCH: Dict[str, str] = {
    "llama":        "llama",
    "mistral":      "llama",   # Mistral uses LLaMA arch in GGUF
    "qwen2":        "qwen2",
    "deepseek_v2":  "deepseek2",
    "gpt2":         "gpt2",
}

# GGML file_type values
GGML_FTYPE_MOSTLY_Q4_K_S = 14

# ---------------------------------------------------------------------------
# Tensor name mappings — list of (regex_pattern, gguf_template)
# First match wins. \1, \2, etc. refer to capture groups.
# ---------------------------------------------------------------------------

# LLaMA / Mistral / Qwen2 / DeepSeek-V2 dense layers
_LLAMA_TENSOR_MAP: List[Tuple[str, str]] = [
    (r"^model\.embed_tokens\.weight$",                           "token_embd.weight"),
    (r"^model\.layers\.(\d+)\.self_attn\.q_proj\.weight$",      r"blk.\1.attn_q.weight"),
    (r"^model\.layers\.(\d+)\.self_attn\.k_proj\.weight$",      r"blk.\1.attn_k.weight"),
    (r"^model\.layers\.(\d+)\.self_attn\.v_proj\.weight$",      r"blk.\1.attn_v.weight"),
    (r"^model\.layers\.(\d+)\.self_attn\.o_proj\.weight$",      r"blk.\1.attn_output.weight"),
    (r"^model\.layers\.(\d+)\.mlp\.gate_proj\.weight$",         r"blk.\1.ffn_gate.weight"),
    (r"^model\.layers\.(\d+)\.mlp\.up_proj\.weight$",           r"blk.\1.ffn_up.weight"),
    (r"^model\.layers\.(\d+)\.mlp\.down_proj\.weight$",         r"blk.\1.ffn_down.weight"),
    (r"^model\.layers\.(\d+)\.input_layernorm\.weight$",        r"blk.\1.attn_norm.weight"),
    (r"^model\.layers\.(\d+)\.post_attention_layernorm\.weight$", r"blk.\1.ffn_norm.weight"),
    (r"^model\.norm\.weight$",                                   "output_norm.weight"),
    (r"^lm_head\.weight$",                                       "output.weight"),
]

# DeepSeek-V2 MLA attention tensors (prepended before the LLaMA patterns)
_DEEPSEEK_EXTRA: List[Tuple[str, str]] = [
    (r"^model\.layers\.(\d+)\.self_attn\.q_a_proj\.weight$",             r"blk.\1.attn_q_a.weight"),
    (r"^model\.layers\.(\d+)\.self_attn\.q_b_proj\.weight$",             r"blk.\1.attn_q_b.weight"),
    (r"^model\.layers\.(\d+)\.self_attn\.kv_a_proj_with_mqa\.weight$",   r"blk.\1.attn_kv_a.weight"),
    (r"^model\.layers\.(\d+)\.self_attn\.kv_b_proj\.weight$",            r"blk.\1.attn_kv_b.weight"),
    # MoE expert tensors
    (r"^model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.gate_proj\.weight$",  r"blk.\1.ffn_gate.\2.weight"),
    (r"^model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.up_proj\.weight$",    r"blk.\1.ffn_up.\2.weight"),
    (r"^model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.down_proj\.weight$",  r"blk.\1.ffn_down.\2.weight"),
]

_DEEPSEEK_TENSOR_MAP: List[Tuple[str, str]] = _DEEPSEEK_EXTRA + _LLAMA_TENSOR_MAP

_GPT2_TENSOR_MAP: List[Tuple[str, str]] = [
    (r"^transformer\.wte\.weight$",                             "token_embd.weight"),
    (r"^transformer\.h\.(\d+)\.attn\.c_attn\.weight$",         r"blk.\1.attn_qkv.weight"),
    (r"^transformer\.h\.(\d+)\.attn\.c_proj\.weight$",         r"blk.\1.attn_output.weight"),
    (r"^transformer\.h\.(\d+)\.mlp\.c_fc\.weight$",            r"blk.\1.ffn_up.weight"),
    (r"^transformer\.h\.(\d+)\.mlp\.c_proj\.weight$",          r"blk.\1.ffn_down.weight"),
    (r"^transformer\.h\.(\d+)\.ln_1\.weight$",                 r"blk.\1.attn_norm.weight"),
    (r"^transformer\.h\.(\d+)\.ln_2\.weight$",                 r"blk.\1.ffn_norm.weight"),
    (r"^transformer\.ln_f\.weight$",                            "output_norm.weight"),
    (r"^lm_head\.weight$",                                      "output.weight"),
]

_ARCH_TENSOR_MAPS: Dict[str, List[Tuple[str, str]]] = {
    "llama":      _LLAMA_TENSOR_MAP,
    "qwen2":      _LLAMA_TENSOR_MAP,   # same HF tensor names
    "deepseek2":  _DEEPSEEK_TENSOR_MAP,
    "gpt2":       _GPT2_TENSOR_MAP,
}

# Counter used for generic sequential naming
_generic_layer_counter: Dict[str, int] = {}


def detect_architecture(hf_model_type: Optional[str]) -> str:
    """Map a HuggingFace model_type string to a GGUF architecture string.

    Args:
        hf_model_type: Value of 'model_type' from HF config.json, or None.

    Returns:
        GGUF architecture string (e.g. "llama", "gpt2", "generic").
    """
    if hf_model_type is None:
        return "generic"
    return _HF_TYPE_TO_ARCH.get(hf_model_type.lower(), "generic")


def map_tensor_name(pytorch_name: str, architecture: str) -> str:
    """Return the GGUF tensor name for a given PyTorch module name.

    Args:
        pytorch_name: Full dotted module name as from model.named_modules()
                      plus ".weight" suffix (e.g. "model.layers.0.self_attn.q_proj.weight").
        architecture: GGUF architecture string (e.g. "llama", "gpt2", "generic").

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
    ("max_position_embeddings", "{arch}.context_length",                    "uint32"),
    ("hidden_size",             "{arch}.embedding_length",                  "uint32"),
    ("num_hidden_layers",       "{arch}.block_count",                       "uint32"),
    ("num_attention_heads",     "{arch}.attention.head_count",              "uint32"),
    ("num_key_value_heads",     "{arch}.attention.head_count_kv",           "uint32"),
    ("intermediate_size",       "{arch}.feed_forward_length",               "uint32"),
    ("rope_theta",              "{arch}.rope.freq_base",                    "float32"),
    ("rms_norm_eps",            "{arch}.attention.layer_norm_rms_epsilon",  "float32"),
    ("vocab_size",              "{arch}.vocab_size",                        "uint32"),
]

_GPT2_CONFIG_MAP: List[Tuple[str, str, str]] = [
    ("n_embd",      "{arch}.embedding_length",   "uint32"),
    ("n_layer",     "{arch}.block_count",         "uint32"),
    ("n_head",      "{arch}.attention.head_count","uint32"),
    ("n_positions", "{arch}.context_length",      "uint32"),
]

_ARCH_CONFIG_MAPS: Dict[str, List[Tuple[str, str, str]]] = {
    "llama":     _LLAMA_CONFIG_MAP,
    "qwen2":     _LLAMA_CONFIG_MAP,
    "deepseek2": _LLAMA_CONFIG_MAP,
    "gpt2":      _GPT2_CONFIG_MAP,
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
