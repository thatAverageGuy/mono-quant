# T-024: GGUFExporter — Tensor Naming, Config Parsing, Public API, CLI, Tests

## Status
TODO

## Phase
07-03 — Phase 7: GGUF Binary Format Export

## Requirements
- Architecture-aware tensor name mapping (PyTorch → GGUF names)
- Config parsing: accept HuggingFace `config.json` path OR a `model_params` dict (or both)
- `GGUFExporter(BaseExporter)` class that orchestrates the full export pipeline
- Public API: `export_to_gguf()` added to `src/mono_quant/export/__init__.py`
- CLI: `monoquant export-gguf` subcommand
- Tests: `tests/test_gguf_export.py` (T-022 + T-024 tests live here; T-023 tests are a subset)

## Decisions
- **Both config_path and model_params accepted**: dict values override config.json on collision
- **Architecture auto-detected** from `config.json["model_type"]` if not passed explicitly
- **Supported architectures (v2.0)**: llama, mistral, qwen2, deepseek_v2, gpt2, generic
- **Generic fallback**: unknown architecture → sequential GGUF names (`blk.N.weight_M`),
  warning emitted, llama.cpp inference not guaranteed
- **DeepSeek-V2**: dense layers map correctly; MLA attention tensors (q_a_proj, kv_a_proj)
  mapped best-effort; MoE expert tensors fall back to generic naming with a warning
- **in_features % 256 != 0**: emit warning and skip layer (write as FP32 GGML_TYPE_F32)
  so the file is still valid; document limitation

## Architecture-Specific Tensor Name Mappings

All mappings are: `(pytorch_name_pattern, gguf_name_template)`.
`N` = layer index, `M` = expert index (MoE only).

### llama / mistral  (model_type: "llama" or "mistral")

| PyTorch name                              | GGUF name                    |
|-------------------------------------------|------------------------------|
| model.embed_tokens.weight                 | token_embd.weight            |
| model.layers.N.self_attn.q_proj.weight    | blk.N.attn_q.weight          |
| model.layers.N.self_attn.k_proj.weight    | blk.N.attn_k.weight          |
| model.layers.N.self_attn.v_proj.weight    | blk.N.attn_v.weight          |
| model.layers.N.self_attn.o_proj.weight    | blk.N.attn_output.weight     |
| model.layers.N.mlp.gate_proj.weight       | blk.N.ffn_gate.weight        |
| model.layers.N.mlp.up_proj.weight         | blk.N.ffn_up.weight          |
| model.layers.N.mlp.down_proj.weight       | blk.N.ffn_down.weight        |
| model.layers.N.input_layernorm.weight     | blk.N.attn_norm.weight       |
| model.layers.N.post_attention_layernorm.weight | blk.N.ffn_norm.weight   |
| model.norm.weight                         | output_norm.weight           |
| lm_head.weight                            | output.weight                |

### qwen2  (model_type: "qwen2")
Same as llama/mistral — Qwen2 uses identical HuggingFace tensor naming.
Set `general.architecture = "qwen2"`.

### deepseek_v2  (model_type: "deepseek_v2")
Dense MLP layers use llama-style mapping. MLA attention adds:

| PyTorch name                                | GGUF name                    |
|---------------------------------------------|------------------------------|
| model.layers.N.self_attn.q_a_proj.weight    | blk.N.attn_q_a.weight        |
| model.layers.N.self_attn.q_b_proj.weight    | blk.N.attn_q_b.weight        |
| model.layers.N.self_attn.kv_a_proj_with_mqa.weight | blk.N.attn_kv_a.weight |
| model.layers.N.self_attn.kv_b_proj.weight   | blk.N.attn_kv_b.weight       |
| model.layers.N.self_attn.o_proj.weight      | blk.N.attn_output.weight     |
| model.layers.N.mlp.experts.M.gate_proj.weight | blk.N.ffn_gate.M.weight   |
| model.layers.N.mlp.experts.M.up_proj.weight   | blk.N.ffn_up.M.weight     |
| model.layers.N.mlp.experts.M.down_proj.weight  | blk.N.ffn_down.M.weight  |

Tensors not matched by any pattern → generic naming with `warnings.warn`.

### gpt2  (model_type: "gpt2")

| PyTorch name                          | GGUF name                    |
|---------------------------------------|------------------------------|
| transformer.wte.weight                | token_embd.weight            |
| transformer.h.N.attn.c_attn.weight    | blk.N.attn_qkv.weight        |
| transformer.h.N.attn.c_proj.weight    | blk.N.attn_output.weight     |
| transformer.h.N.mlp.c_fc.weight       | blk.N.ffn_up.weight          |
| transformer.h.N.mlp.c_proj.weight     | blk.N.ffn_down.weight        |
| transformer.ln_1.N.weight             | blk.N.attn_norm.weight       |
| transformer.ln_f.weight               | output_norm.weight           |
| lm_head.weight                        | output.weight                |

### generic  (unknown model_type or explicit `--architecture generic`)
Sequential naming: `blk.N.weight_M` where N = linear layer index within block,
M = weight index. Emit `warnings.warn` once. llama.cpp inference not guaranteed.

## HuggingFace config.json → GGUF KV Mapping

### LLaMA / Mistral / Qwen2 / DeepSeek-V2

| HF config key               | GGUF KV key                                    | Type    |
|-----------------------------|------------------------------------------------|---------|
| model_type (mapped)         | general.architecture                           | string  |
| (model name or path)        | general.name                                   | string  |
| (constant: 2)               | general.quantization_version                   | uint32  |
| (GGML file type enum)       | general.file_type                              | uint32  |
| max_position_embeddings     | {arch}.context_length                          | uint32  |
| hidden_size                 | {arch}.embedding_length                        | uint32  |
| num_hidden_layers           | {arch}.block_count                             | uint32  |
| num_attention_heads         | {arch}.attention.head_count                    | uint32  |
| num_key_value_heads         | {arch}.attention.head_count_kv                 | uint32  |
| intermediate_size           | {arch}.feed_forward_length                     | uint32  |
| rope_theta (default: 10000) | {arch}.rope.freq_base                          | float32 |
| rms_norm_eps                | {arch}.attention.layer_norm_rms_epsilon        | float32 |
| vocab_size                  | {arch}.vocab_size                              | uint32  |

`{arch}` = GGUF architecture string (e.g., `"llama"`, `"qwen2"`). HF model_type values
map to GGUF architecture strings via: `"mistral" → "llama"`, all others use as-is.

### GPT-2

| HF config key  | GGUF KV key                      | Type   |
|----------------|----------------------------------|--------|
| model_type     | general.architecture ("gpt2")    | string |
| n_embd         | gpt2.embedding_length            | uint32 |
| n_layer        | gpt2.block_count                 | uint32 |
| n_head         | gpt2.attention.head_count        | uint32 |
| n_positions    | gpt2.context_length              | uint32 |

### GGML file_type values for general.file_type
```python
GGML_FTYPE_MOSTLY_Q4_K_S = 14   # used when quantization_type = "q4_k_s"
```

### Missing config keys
If a required key is missing from config.json/model_params:
- Emit `warnings.warn(f"Missing config key '{k}', skipping GGUF KV entry")`
- Continue (do not raise); the file is still structurally valid

## Architecture: GGUFExporter

**File**: `src/mono_quant/export/gguf.py`

```python
class GGUFExporter(BaseExporter):

    def export(
        self,
        model: nn.Module,
        path: Union[str, Path],
        quantization_type: str = "q4_k_s",
        architecture: Optional[str] = None,
        config_path: Optional[Union[str, Path]] = None,
        model_params: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:

    def validate_compatibility(self, model: nn.Module) -> None:
        # raise TypeError if not nn.Module

    def build_metadata(self, model: nn.Module, **kwargs: Any) -> Dict[str, Any]:
        # returns merged config dict (config.json + model_params override)
```

**Thin wrapper**: `src/mono_quant/export/gguf_impl.py` (mirrors `gptq_impl.py`):
```python
def export_to_gguf_impl(model, path, quantization_type, architecture,
                         config_path, model_params, **kwargs) -> None:
    GGUFExporter().export(model, path, quantization_type=quantization_type,
                          architecture=architecture, config_path=config_path,
                          model_params=model_params)
```

## State Machine: GGUFExporter.export() flow

```
        [export() called]
               │
               ▼
  [validate_compatibility]──fail──→ [TypeError]
               │
               ▼
  [resolve_config]
    config_path provided?
      → read JSON → base_config
    model_params provided?
      → override base_config keys
    neither → base_config = {}
    warn if both empty
               │
               ▼
  [detect_architecture]
    architecture arg? → use it
    else config["model_type"]? → map to GGUF arch
    else → "generic" + warn
               │
               ▼
  [revert_to_standard_modules]
    → fp32_model (all nn.Linear)
               │
               ▼
  [build_kv_metadata]
    map base_config keys → GGUF KV
    add mono_quant_version
               │
               ▼
  [create GGUFWriter]
  [add all KV entries]
               │
    for each nn.Linear in fp32_model:
               │
               ▼
  [map_tensor_name(pytorch_name, arch)]
               │
               ▼
  [quantize_to_q4_k_s(weight)]──incompat──→ [warn + write FP32]
               │
               ▼
  [writer.add_tensor(gguf_name, data, shape, GGML_TYPE_Q4_K)]
               │
  [all layers done]
               │
               ▼
  [writer.write(path / "model.gguf")]
               │
               ▼
             [DONE]
```

## Public API

**File**: `src/mono_quant/export/__init__.py` — add:
```python
def export_to_gguf(
    model: nn.Module,
    path: Union[str, Path],
    quantization_type: str = "q4_k_s",
    architecture: Optional[str] = None,
    config_path: Optional[Union[str, Path]] = None,
    model_params: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> None:
    """Export a model to GGUF format for use with llama.cpp.

    Args:
        model: Any nn.Module (quantized or plain FP32).
        path: Output directory. model.gguf written inside.
        quantization_type: Quantization format. Currently only "q4_k_s".
        architecture: GGUF architecture string. Auto-detected from config if None.
        config_path: Path to HuggingFace config.json. Optional.
        model_params: Dict of model hyperparameters (same keys as HF config.json).
                      Overrides config_path values on collision.
    """
    from mono_quant.export.gguf_impl import export_to_gguf_impl
    export_to_gguf_impl(model, path, quantization_type=quantization_type,
                        architecture=architecture, config_path=config_path,
                        model_params=model_params)
```

**File**: `src/mono_quant/__init__.py` — add `export_to_gguf` to `__all__` and import.

## CLI

**File**: `src/mono_quant/cli/main.py` — add `export-gguf` subcommand:

```
monoquant export-gguf MODEL_PATH OUTPUT_DIR
  --quantization-type  q4_k_s (default)
  --architecture       auto-detect (default), or llama/mistral/qwen2/deepseek_v2/gpt2/generic
  --config             path to HuggingFace config.json
  --model-param        KEY=VALUE  (repeatable; overrides config.json)
```

`--model-param` parses `KEY=VALUE` into a dict and passes as `model_params`.
Values are parsed as int if possible, then float, then string.

Example:
```
monoquant export-gguf ./model ./output \
  --config ./model/config.json \
  --architecture llama
```

## Architecture: arch_maps.py

**File**: `src/mono_quant/export/gguf/arch_maps.py`

Two dictionaries per architecture:
1. `TENSOR_MAPS: Dict[str, List[Tuple[str, str]]]` — list of (regex_pattern, gguf_template)
   patterns applied in order; first match wins
2. `CONFIG_MAPS: Dict[str, List[Tuple[str, str, str]]]` — list of (hf_key, gguf_key, gguf_arch)

Public function:
```python
def map_tensor_name(pytorch_name: str, architecture: str) -> str:
    """Return the GGUF tensor name for a given PyTorch name and architecture.
    Returns a generic sequential name if no pattern matches, with a warning."""

def build_kv_entries(
    config: Dict[str, Any],
    architecture: str,
    quantization_type: str,
) -> List[Tuple[str, str, Any]]:
    """Return list of (gguf_key, type_hint, value) tuples for GGUFWriter.add_* calls."""
```

Use `re.sub` for pattern replacement (e.g., replace `r'model\.layers\.(\d+)\.'` → `blk.\1.`).

## Dependencies
- T-022 (GGUFWriter) — must be complete
- T-023 (quantize_to_q4_k_s) — must be complete

## Success Criteria
- [ ] `export_to_gguf(model, path)` writes `path/model.gguf`
- [ ] KV metadata populated from config.json when provided
- [ ] model_params dict values override config.json on collision
- [ ] LLaMA tensor names map correctly (verified by gguf-py reader)
- [ ] Qwen2 tensor names map correctly
- [ ] Unknown architecture falls back to generic naming with warning
- [ ] `monoquant export-gguf` CLI works end-to-end
- [ ] `export_to_gguf` accessible from `import mono_quant`

## Testing Requirements

File: `tests/test_gguf_export.py`

T-022 tests (writer unit tests):
- `test_gguf_writer_magic_and_version`
- `test_gguf_writer_kv_string`
- `test_gguf_writer_kv_uint32`
- `test_gguf_writer_kv_float32`
- `test_gguf_writer_tensor_alignment`
- `test_gguf_writer_tensor_offsets`

T-023 tests (quant_types unit tests):
- `test_q4k_block_size`
- `test_q4k_scale_packing_round_trip`
- `test_q4k_nibble_packing_round_trip`
- `test_q4k_reconstruction_error`
- `test_q4k_invalid_in_features`

T-024 integration tests:
- `test_gguf_export_creates_file` — `model.gguf` exists after export
- `test_gguf_export_ggufpy_reads_file` — gguf-py opens without error
- `test_gguf_export_tensor_count` — gguf-py tensor count matches nn.Linear count
- `test_gguf_export_config_path` — model_params from config.json written as KV
- `test_gguf_export_model_params_override` — model_params dict overrides config.json
- `test_gguf_export_llama_tensor_names` — map_tensor_name produces correct GGUF names
- `test_gguf_export_unknown_arch_warns` — warning emitted for unknown architecture
- `test_gguf_export_cli_runs` — CLI invocation produces model.gguf

Total target: ~19 tests (6 T-022 + 5 T-023 + 8 T-024)
Coverage target: all public functions in writer.py, quant_types.py, arch_maps.py, gguf.py

## Open Questions
None.
