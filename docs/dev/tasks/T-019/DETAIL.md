# T-019: GPTQ Checkpoint Export with quantization_config.json

## Status
TODO

## Phase
06-02 — Phase 6: GPTQ/AWQ Checkpoint Exports

## Requirements
- GPTQ-04: Include quantization_config.json for vLLM/SGLang compatibility

## Context
A GPTQ checkpoint without quantization_config.json will fail to load in vLLM and
text-generation-webui. The config must include specific fields that loaders use
to detect quantization type and configure their quantization kernels.

## Required quantization_config.json Fields

```json
{
  "quant_type": "gptq",
  "bits": 4,
  "group_size": 128,
  "sym": true,
  "desc_act": false,
  "model_type": "auto",
  "quantization_config": {
    "bits": 4,
    "group_size": 128,
    "damp_percent": 0.01,
    "desc_act": false,
    "static_groups": false,
    "sym": true,
    "true_sequential": false,
    "model_name_or_path": null,
    "model_file_base_name": null,
    "quant_method": "gptq"
  }
}
```

## Checkpoint Directory Structure

```
model_gptq/
├── quantization_config.json    ← required for vLLM
├── config.json                 ← optional: HuggingFace-style model config
└── model.safetensors           ← packed weights (qweight, scales, qzeros)
```

## Success Criteria
- [ ] quantization_config.json written with all required vLLM fields
- [ ] config.json written with model metadata
- [ ] Checkpoint directory structure matches vLLM expectation
- [ ] vLLM loads checkpoint without errors
- [ ] text-generation-webui loads checkpoint without errors

## Dependencies
- T-018 (4-bit packing must work first)

## Testing Requirements
- Unit: verify JSON structure matches schema
- Integration: load with vLLM (if available in test env), else schema validation
- Coverage: config generation for models with different architectures

## Open Questions
- [ ] Should we support exporting as directory (HF-style) or single file?
- [ ] What model metadata to include in config.json?
- [ ] Support for desc_act=True (activation ordering)?

## Implementation Guidance

1. Add `_write_quantization_config(path, info, group_size)` to GPTQExporter
2. Add `_write_checkpoint_dir(path, checkpoint_dict, config)` for directory structure
3. Ensure safetensors format for weight file (vLLM prefers safetensors)
4. Integration test: load directory with vLLM AutoModelForCausalLM.from_pretrained()
