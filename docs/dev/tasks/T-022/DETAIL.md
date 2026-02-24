# T-022: GGUF Binary Format Writer — Header + KV Metadata

## Status
TODO

## Phase
07-01 — Phase 7: GGUF Binary Format Export

## Requirements
- GGUF-01: Export to GGUF binary format (header + metadata KV + tensor data)

## Context
GGUF (GGML Universal File) is a binary format used by llama.cpp. The format has a
specific binary structure that must be followed exactly. Unlike GPTQ/AWQ (which are
just JSON + safetensors), GGUF requires writing a custom binary file.

Reference: https://github.com/ggml-org/ggml/blob/master/docs/gguf.md

## GGUF File Structure

```
┌──────────────────────────────┐
│ Header                        │
│  magic:    0x46554747 ("GGUF")│
│  version:  3 (uint32)         │
│  n_tensors: (uint64)          │
│  n_kv:      (uint64)          │
├──────────────────────────────┤
│ Metadata KV pairs             │
│  key: string                  │
│  type: gguf_metadata_value_t  │
│  value: type-dependent        │
├──────────────────────────────┤
│ Tensor Info Array             │
│  name: string                 │
│  n_dims: uint32               │
│  dims: [uint64]               │
│  type: ggml_type              │
│  offset: uint64               │
├──────────────────────────────┤
│ Padding (32-byte alignment)   │
├──────────────────────────────┤
│ Tensor Data                   │
│  (raw bytes, 32-byte aligned) │
└──────────────────────────────┘
```

## Required KV Metadata (Minimum)
```
general.architecture: string ("llama" / "gpt2" / etc.)
general.quantization_version: uint32 (2)
general.file_type: uint32 (GGUF_TYPE for quantization)
general.name: string (model name)
```

## Success Criteria
- [ ] GGUF file written with correct magic number and version 3
- [ ] KV metadata section written with required keys
- [ ] Tensor info array written with correct offsets
- [ ] Tensor data section 32-byte aligned
- [ ] gguf-py library can read written file (basic validation)

## Dependencies
- T-021 (Phase 6 patterns established)

## Critical Pitfall
GGUF metadata format is underspecified publicly. Use llama.cpp source and gguf-py
as ground truth, not documentation. Missing or wrong keys cause silent load failures.

## Testing Requirements
- Unit: verify binary file structure (magic, version, counts)
- Unit: verify KV pairs are readable by gguf-py
- Integration: llama.cpp can load the file (manual test if llama.cpp unavailable in CI)

## Open Questions
- [ ] GGUF version: v2 or v3? (Use v3 — current standard)
- [ ] Which metadata keys are mandatory vs optional for llama.cpp?

## Implementation Guidance

1. Create `src/mono_quant/export/gguf/` directory
2. Create `src/mono_quant/export/gguf/writer.py` — binary GGUF file writer
3. Implement `GGUFWriter` class with `add_kv()`, `add_tensor()`, `write(path)` methods
4. Use Python `struct` module for binary packing
5. Test against gguf-py: `pip install gguf` for validation
