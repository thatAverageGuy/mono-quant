# T-025: llama.cpp Validation Testing

## Status
TODO

## Phase
07-04 — Phase 7: GGUF Binary Format Export

## Requirements
- GGUF-06: Validate GGUF exports work correctly with llama.cpp

## Context
Same as T-021 but for GGUF/llama.cpp. llama.cpp has stricter validation than ONNX
Runtime — files that look structurally correct can still fail with "unknown tensor type"
or produce garbage outputs due to incorrect quantization metadata.

## Validation Approach

### Level 1: gguf-py Validation (automated)
```python
import gguf
reader = gguf.GGUFReader(path)
# Verify: magic, version, tensor count, KV metadata
```

### Level 2: llama.cpp Load Test (manual / CI if available)
```bash
./llama-cli -m model.gguf -p "Hello" -n 10
```
- Check: no error messages
- Check: output is not garbage (contains real words)

### Level 3: Accuracy Benchmark (recommended)
- Compare perplexity on small dataset vs. FP32 model
- Q4_K_M typically within 0.1-0.3 perplexity points of FP32

## Success Criteria
- [ ] gguf-py reads file without errors (automated)
- [ ] llama.cpp loads file without "invalid quantization format" error
- [ ] llama.cpp generates coherent text (not garbage)
- [ ] Tensor count matches expected model structure

## Dependencies
- T-023 (quantization types complete)
- T-024 (tensor naming complete)

## Testing Requirements
- Unit: gguf-py validation (pip install gguf)
- Integration: llama.cpp CLI test (manual test procedure in IMPL_LOG if not in CI)
- Coverage: Q4_K_M and Q4_K_S both validated

## Open Questions
- [ ] Can CI/CD run llama.cpp? (Probably not — document manual procedure)
- [ ] Should we add a `--validate-runtime llama.cpp` flag?

## Implementation Guidance

1. Add `validate_gguf_with_gguf_py(path)` to validators.py
2. Document manual llama.cpp test procedure in IMPL_LOG
3. Add `--validate-runtime llamacpp` CLI option that prints test instructions
4. Add gguf to optional deps: `pip install mono-quant[gguf]`
