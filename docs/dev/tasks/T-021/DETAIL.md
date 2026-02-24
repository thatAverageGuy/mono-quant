# T-021: vLLM/SGLang Validation and Accuracy Benchmarking

## Status
TODO

## Phase
06-04 — Phase 6: GPTQ/AWQ Checkpoint Exports

## Requirements
- GPTQ-05: Validate GPTQ exports work correctly with vLLM
- AWQ-04: Validate AWQ exports work correctly with vLLM / SGLang

## Context
Format correctness (T-018 to T-020) is necessary but not sufficient. We need to
verify that exported models actually run and produce correct outputs in target runtimes.
Silent correctness issues (model loads but outputs garbage) are the key risk.

## Validation Levels

### Level 1: Load Validation (mandatory)
- vLLM can instantiate the model without error
- SGLang can instantiate the model without error
- No "format mismatch" or "unsupported type" errors

### Level 2: Output Validation (mandatory)
- Export original FP32 model outputs on test inputs
- Export quantized PyTorch model outputs on same test inputs
- Export quantized vLLM outputs on same test inputs
- Compare: vLLM output should be within 2% SQNR of PyTorch quantized output

### Level 3: Accuracy Benchmark (recommended)
- Run on a small benchmark dataset (100-500 samples)
- Report accuracy delta vs FP32: GPTQ/AWQ typically <1% on standard benchmarks
- Fail export if accuracy delta >5% (configurable threshold)

## Validation CLI Integration

```bash
# New CLI option
monoquant export model.pt --format gptq --validate-runtime vllm
monoquant export model.pt --format awq --validate-runtime sglang
```

## Success Criteria
- [ ] vLLM GPTQ load and inference verified
- [ ] vLLM AWQ load and inference verified
- [ ] SGLang AWQ load verified (or dependency documented)
- [ ] Accuracy delta reported and threshold enforced
- [ ] --validate-runtime CLI option added

## Dependencies
- T-019 (GPTQ checkpoint complete)
- T-020 (AWQ checkpoint complete)

## Testing Requirements
- Integration: requires vLLM/SGLang installed (skip if not available)
- Unit: accuracy delta calculation logic
- Document manual test procedure if automated testing is not possible in CI

## Open Questions
- [ ] Can CI/CD environment have vLLM installed? (May require manual test procedure)
- [ ] What accuracy delta threshold is appropriate? (2% proposed)
- [ ] Which vLLM version to target? (Format has evolved across versions)

## Implementation Guidance

1. Add `--validate-runtime [vllm|sglang|none]` to `monoquant export` CLI
2. Implement `validate_gptq_with_vllm(checkpoint_path, test_inputs)` in validators.py
3. Implement `validate_awq_with_vllm(checkpoint_path, test_inputs)` in validators.py
4. Add accuracy delta computation and configurable threshold
5. If vLLM not installed: print instructions for manual validation with test procedure
