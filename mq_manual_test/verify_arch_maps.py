"""Verification script for T-042 arch_maps additions."""
import sys
sys.path.insert(0, "../src")

from mono_quant.export.gguf.arch_maps import detect_architecture, map_tensor_name, reset_generic_counter

PASS = 0
FAIL = 0

def check(label, got, expected):
    global PASS, FAIL
    if got == expected:
        PASS += 1
    else:
        FAIL += 1
        print(f"  FAIL  {label}")
        print(f"        expected: {expected!r}")
        print(f"        got:      {got!r}")

def check_tensor(arch, pt_name, expected):
    reset_generic_counter(arch)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        got = map_tensor_name(pt_name, arch)
    check(f"{arch}: {pt_name}", got, expected)

# ── detect_architecture ───────────────────────────────────────────────────
print("=== detect_architecture ===")
detect_cases = [
    ("llama",           "llama"),
    ("mistral",         "llama"),
    ("mixtral",         "llama"),
    ("qwen2",           "qwen2"),
    ("deepseek_v2",     "deepseek2"),
    ("gpt2",            "gpt2"),
    ("opt",             "opt"),
    ("phi",             "phi2"),
    ("phi-msft",        "phi2"),
    ("phi3",            "phi3"),
    ("chatglm",         "chatglm"),
    ("glm4",            "chatglm"),
    ("falcon",          "falcon"),
    ("RefinedWebModel", "falcon"),
    ("RefinedWeb",      "falcon"),
    ("gemma",           "gemma"),
    ("gemma2",          "gemma2"),
    ("gpt_bigcode",     "starcoder"),
    ("starcoder2",      "starcoder2"),
    ("bloom",           "bloom"),
    ("mpt",             "mpt"),
    ("cohere",          "command-r"),
    ("unknown_xyz",     "generic"),
    (None,              "generic"),
]
for model_type, expected in detect_cases:
    got = detect_architecture(model_type)
    check(f"detect({model_type!r})", got, expected)
print(f"  {PASS} pass, {FAIL} fail\n")

# ── map_tensor_name ───────────────────────────────────────────────────────
PASS = FAIL = 0
print("=== map_tensor_name ===")

# OPT
check_tensor("opt", "model.decoder.embed_tokens.weight",                          "token_embd.weight")
check_tensor("opt", "model.decoder.layers.0.self_attn.q_proj.weight",            "blk.0.attn_q.weight")
check_tensor("opt", "model.decoder.layers.0.self_attn.q_proj.bias",              "blk.0.attn_q.bias")
check_tensor("opt", "model.decoder.layers.2.fc1.weight",                         "blk.2.ffn_up.weight")
check_tensor("opt", "model.decoder.layers.2.fc2.bias",                           "blk.2.ffn_down.bias")
check_tensor("opt", "model.decoder.final_layer_norm.weight",                     "output_norm.weight")
check_tensor("opt", "lm_head.weight",                                             "output.weight")

# Phi-2 new layout
check_tensor("phi2", "model.embed_tokens.weight",                                 "token_embd.weight")
check_tensor("phi2", "model.layers.0.self_attn.q_proj.weight",                   "blk.0.attn_q.weight")
check_tensor("phi2", "model.layers.0.self_attn.q_proj.bias",                     "blk.0.attn_q.bias")
check_tensor("phi2", "model.layers.1.mlp.fc1.weight",                            "blk.1.ffn_up.weight")
check_tensor("phi2", "model.final_layernorm.weight",                             "output_norm.weight")
check_tensor("phi2", "lm_head.linear.weight",                                    "output.weight")
# Phi-2 old layout
check_tensor("phi2", "transformer.embd.wte.weight",                              "token_embd.weight")
check_tensor("phi2", "transformer.h.0.mixer.Wqkv.weight",                       "blk.0.attn_qkv.weight")
check_tensor("phi2", "transformer.h.0.mlp.fc1.bias",                            "blk.0.ffn_up.bias")

# Phi-3
check_tensor("phi3", "model.embed_tokens.weight",                                "token_embd.weight")
check_tensor("phi3", "model.layers.0.self_attn.qkv_proj.weight",                "blk.0.attn_qkv.weight")
check_tensor("phi3", "model.layers.0.mlp.gate_up_proj.weight",                  "blk.0.ffn_up.weight")
check_tensor("phi3", "model.layers.0.mlp.down_proj.weight",                     "blk.0.ffn_down.weight")
check_tensor("phi3", "lm_head.weight",                                           "output.weight")

# ChatGLM
check_tensor("chatglm", "transformer.embedding.word_embeddings.weight",          "token_embd.weight")
check_tensor("chatglm", "transformer.encoder.layers.0.self_attention.query_key_value.weight",
             "blk.0.attn_qkv.weight")
check_tensor("chatglm", "transformer.encoder.layers.0.mlp.dense_h_to_4h.weight", "blk.0.ffn_up.weight")
check_tensor("chatglm", "transformer.encoder.final_layernorm.weight",            "output_norm.weight")
check_tensor("chatglm", "transformer.output_layer.weight",                       "output.weight")

# Falcon 7B (single layernorm)
check_tensor("falcon", "transformer.word_embeddings.weight",                     "token_embd.weight")
check_tensor("falcon", "transformer.h.0.input_layernorm.weight",                "blk.0.attn_norm.weight")
check_tensor("falcon", "transformer.h.0.self_attention.query_key_value.weight", "blk.0.attn_qkv.weight")
check_tensor("falcon", "transformer.h.0.mlp.dense_h_to_4h.weight",             "blk.0.ffn_up.weight")
check_tensor("falcon", "transformer.ln_f.weight",                               "output_norm.weight")
check_tensor("falcon", "lm_head.weight",                                         "output.weight")
# Falcon 40B separate norms
check_tensor("falcon", "transformer.h.0.ln_attn.weight",                        "blk.0.attn_norm.weight")
check_tensor("falcon", "transformer.h.0.ln_mlp.weight",                         "blk.0.attn_norm_2.weight")

# Gemma
check_tensor("gemma", "model.embed_tokens.weight",                               "token_embd.weight")
check_tensor("gemma", "model.layers.0.self_attn.q_proj.weight",                 "blk.0.attn_q.weight")
check_tensor("gemma", "model.layers.0.mlp.gate_proj.weight",                    "blk.0.ffn_gate.weight")
check_tensor("gemma", "model.norm.weight",                                       "output_norm.weight")

# Gemma2 (extra norms + base Gemma patterns)
check_tensor("gemma2", "model.layers.0.pre_feedforward_layernorm.weight",       "blk.0.ffn_norm.weight")
check_tensor("gemma2", "model.layers.0.post_feedforward_layernorm.weight",      "blk.0.post_ffw_norm.weight")
check_tensor("gemma2", "model.layers.0.self_attn.q_proj.weight",                "blk.0.attn_q.weight")

# StarCoder
check_tensor("starcoder", "transformer.wte.weight",                              "token_embd.weight")
check_tensor("starcoder", "transformer.wpe.weight",                              "position_embd.weight")
check_tensor("starcoder", "transformer.h.0.attn.c_attn.weight",                 "blk.0.attn_qkv.weight")
check_tensor("starcoder", "transformer.h.0.mlp.c_fc.bias",                      "blk.0.ffn_up.bias")
check_tensor("starcoder", "transformer.ln_f.weight",                             "output_norm.weight")

# StarCoder2
check_tensor("starcoder2", "model.embed_tokens.weight",                          "token_embd.weight")
check_tensor("starcoder2", "model.layers.0.self_attn.q_proj.weight",            "blk.0.attn_q.weight")
check_tensor("starcoder2", "model.layers.0.mlp.c_fc.weight",                    "blk.0.ffn_up.weight")
check_tensor("starcoder2", "model.norm.weight",                                  "output_norm.weight")

# BLOOM
check_tensor("bloom", "transformer.word_embeddings.weight",                      "token_embd.weight")
check_tensor("bloom", "transformer.word_embeddings_layernorm.weight",            "token_embd_norm.weight")
check_tensor("bloom", "transformer.h.0.self_attention.query_key_value.weight",   "blk.0.attn_qkv.weight")
check_tensor("bloom", "transformer.h.0.mlp.dense_h_to_4h.weight",               "blk.0.ffn_up.weight")
check_tensor("bloom", "transformer.ln_f.bias",                                   "output_norm.bias")

# MPT
check_tensor("mpt", "transformer.wte.weight",                                    "token_embd.weight")
check_tensor("mpt", "transformer.blocks.0.attn.Wqkv.weight",                    "blk.0.attn_qkv.weight")
check_tensor("mpt", "transformer.blocks.0.ffn.up_proj.weight",                   "blk.0.ffn_up.weight")
check_tensor("mpt", "transformer.norm_f.weight",                                 "output_norm.weight")

# Command-R
check_tensor("command-r", "model.embed_tokens.weight",                           "token_embd.weight")
check_tensor("command-r", "model.layers.0.self_attn.q_proj.weight",              "blk.0.attn_q.weight")
check_tensor("command-r", "model.layers.0.self_attn.q_norm.weight",              "blk.0.attn_q_norm.weight")
check_tensor("command-r", "model.layers.0.mlp.gate_proj.weight",                 "blk.0.ffn_gate.weight")
check_tensor("command-r", "model.norm.weight",                                   "output_norm.weight")

print(f"  {PASS} pass, {FAIL} fail\n")
print(f"=== TOTAL tensor: {PASS} pass, {FAIL} fail ===")
sys.exit(FAIL)
