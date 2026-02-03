# Pitfalls Research

**Domain:** Model Quantization Tools / PyTorch Packages
**Researched:** 2025-02-03
**Confidence:** MEDIUM

## Critical Pitfalls

### Pitfall 1: Blind Quantization Without Selectivity

**What goes wrong:**
Quantizing all layers indiscriminately causes significant accuracy degradation, especially with normalization layers, embeddings, or operations sensitive to precision loss. Research shows naive quantization can cause 10-35% accuracy drops depending on model architecture.

**Why it happens:**
Developers equate "quantization = everything to INT4/INT8" without understanding that different layer types have different sensitivity to precision loss. The desire for maximum compression overrides accuracy considerations.

**How to avoid:**
- Implement selective quantization: only quantize weights and activations for appropriate layer types (linear layers, attention mechanisms)
- Exclude sensitive layers: normalization, layer norms, embeddings often need higher precision
- Provide hybrid options: allow users to quantize weights but keep some activations in float
- Default to conservative quantization (INT8 or FP16) with opt-in for aggressive compression (INT4)

**Warning signs:**
- Accuracy drops >2% on validation set after quantization
- NaN or Inf values appearing during inference
- Perplexity scores degrading significantly on language models

**Phase to address:**
Phase 1 (Core Quantization) - Implement selective quantization from the start, not as an afterthought

**Severity:** CRITICAL

---

### Pitfall 2: Insufficient Validation on Target Hardware

**What goes wrong:**
Models quantized and validated on development workstations (often with GPUs) fail or behave differently when deployed to CPU servers, edge devices, or different GPU architectures. Performance gains may not materialize due to missing kernel support.

**Why it happens:**
Development environments differ from production environments. Developers test on powerful machines with different instruction sets, BLAS libraries, or CUDA capabilities. Integer kernels may not be available on all target hardware.

**How to avoid:**
- Always validate quantized models on actual deployment hardware during development
- Provide clear documentation of hardware requirements (CPU extensions, GPU compute capability)
- Include hardware detection in the CLI/API with helpful error messages
- Test on at least two different hardware configurations if possible

**Warning signs:**
- Quantization works on dev machine but fails in CI/CD or production
- Latency actually increases after quantization (integer kernels not available)
- "Kernels not found" or unsupported operation errors

**Phase to address:**
Phase 2 (CLI & Validation) - Add hardware detection and target-hardware testing capabilities

**Severity:** CRITICAL

---

### Pitfall 3: Serialization/Derialization Breaking Quantization State

**What goes wrong:**
Quantized models fail to save/load correctly, losing scale factors, zero points, or quantization configuration. PyTorch has documented issues with quantized model serialization, and torch.ao.quantization is being deprecated.

**Why it happens:**
Quantization requires storing additional metadata (scale, zero-point, quantization scheme) beyond just the weights. Standard torch.save/load may not preserve this correctly. Different quantization formats (GPTQ, AWQ, bitsandbytes) have incompatible serialization.

**How to avoid:**
- Use state_dict-based serialization (explicitly documented as the supported method)
- Store quantization parameters alongside weights in a structured format
- Validate quantized model after save/load cycle with same inference output
- Document the serialization format clearly for interoperability
- Consider storing both original and quantized weights for recovery

**Warning signs:**
- Loading a saved quantized model produces different outputs than before saving
- "Missing key" or "size mismatch" errors when loading quantized state dicts
- Scale/zero-point values are None or missing after loading

**Phase to address:**
Phase 1 (Core Quantization) - Implement robust save/load with validation tests from day one

**Severity:** CRITICAL

---

### Pitfall 4: Calibration Data Mismatch

**What goes wrong:**
Using non-representative calibration data for static quantization causes poor scale/zero-point selection, leading to accuracy loss or clipping errors. This is especially problematic for dynamic quantization of activations.

**Why it happens:**
Developers use small, random, or convenience datasets for calibration instead of representative production data. Activation distributions vary significantly across input domains, making calibration sensitive to data choice.

**How to avoid:**
- Document calibration data requirements clearly (size, diversity, representativeness)
- Provide guidance on minimum calibration samples (typically 100-500 examples)
- Implement warnings when calibration data appears insufficient
- For dynamic quantization (activations), clearly explain it's data-independent

**Warning signs:**
- High clipping rates (many values at quantization range boundaries)
- Accuracy varies significantly with different calibration datasets
- Per-layer statistics show extreme outliers dominating scale calculation

**Phase to address:**
Phase 2 (CLI & Validation) - Add calibration quality checks and user guidance

**Severity:** HIGH

---

### Pitfall 5: Format Incompatibility Hell

**What goes wrong:**
Different quantization formats (GPTQ, AWQ, NF4, INT4, EXL2, GGUF) are incompatible with each other and with different inference frameworks. Users get locked into ecosystems or cannot deploy quantized models to their target inference engine.

**Why it happens:**
The quantization ecosystem is fragmented with competing formats optimized for different runtimes (vLLM, TensorRT, llama.cpp). Tools pick one format without considering deployment targets.

**How to avoid:**
- Use standard PyTorch quantization formats as the foundation
- Clearly document compatibility with major inference frameworks
- Consider providing export options for common deployment targets
- Avoid introducing proprietary or custom quantization formats

**Warning signs:**
- Users report "format not supported" errors with their inference framework
- Need to convert between multiple quantization formats
- Models quantized with one tool cannot be loaded by another

**Phase to address:**
Phase 1 (Core Quantization) - Design for interoperability from the start

**Severity:** HIGH

---

## Technical Debt Patterns

Shortcuts that seem reasonable but create long-term problems.

| Shortcut | Immediate Benefit | Long-term Cost | When Acceptable |
|----------|-------------------|----------------|-----------------|
| Hardcoding quantization parameters | Faster initial implementation | Inflexible API, can't adapt to different models | Never |
| Skipping scale/zero-point validation | Simpler code, faster development | Subtle bugs, hard-to-debug accuracy issues | Never |
| Using internal PyTorch APIs | Access to more features | Breaking changes with PyTorch updates | Only if documented as stable |
| Single-threaded quantization | Simpler implementation | Poor scalability for large models | MVP only |
| No per-layer configuration | Simpler API | Cannot handle sensitive layers | MVP with planned enhancement |

## Integration Gotchas

Common mistakes when connecting to external services.

| Integration | Common Mistake | Correct Approach |
|-------------|----------------|------------------|
| PyTorch versions | Assuming quantization works across all versions | Pin minimum PyTorch version, test on multiple |
| HuggingFace integration | Adding transformers as required dependency | Keep optional, accept any loaded model |
| CUDA/ROCM | Assuming GPU is always available | Graceful fallback to CPU with clear messaging |
| Apple Silicon | Ignoring macOS entirely | Document platform limitations clearly |
| CI/CD pipelines | Quantizing during deployment instead of build phase | Design as build-time tool, document workflow |

## Performance Traps

Patterns that work at small scale but fail as usage grows.

| Trap | Symptoms | Prevention | When It Breaks |
|------|----------|------------|----------------|
| In-memory quantization of entire model | OOM on large models (7B+ params) | Stream quantization per-layer | Models >2GB |
| No progress reporting | Appears frozen during quantization | Show progress for each layer | Models >100M params |
| Synchronous blocking in CLI | Can't quantize multiple models | Add async options | Batch processing |
| Repeated calibration data loading | Slow quantization startup | Cache calibration statistics | Multiple quantizations |

## API Design Pitfalls

Common API mistakes specific to quantization tools.

| Pitfall | User Impact | Better Approach |
|---------|-------------|-----------------|
| Too many quantization options | Paralysis, confusion, wrong choices | Sensible defaults, single primary method |
| Inconsistent naming (int8 vs INT8 vs i8) | Errors, frustration | Standardized type naming |
| No way to preview quantization | Fear of breaking models | Dry-run mode showing what will change |
| All-or-nothing quantization | Can't find accuracy bottleneck | Per-layer quantization control |
| Hidden calibration data requirements | Unexpected failures | Clear errors with guidance |

## "Looks Done But Isn't" Checklist

Things that appear complete but are missing critical pieces.

- [ ] **INT8 quantization:** Often missing scale/zero-point validation — verify quantized model produces same outputs as original within tolerance
- [ ] **INT4 quantization:** Often missing accuracy warnings on aggressive compression — always validate accuracy and warn if >2% degradation
- [ ] **Save/load:** Often missing serialization of quantization metadata — verify round-trip preserves all parameters
- [ ] **CLI:** Often missing helpful error messages for unsupported models — detect and explain what's not supported
- [ ] **Model-agnostic claim:** Often fails on custom architectures — test on non-Transformer architectures
- [ ] **Calibration:** Often missing guidance on data requirements — document size and diversity needs

## Recovery Strategies

When pitfalls occur despite prevention, how to recover.

| Pitfall | Recovery Cost | Recovery Steps |
|---------|---------------|----------------|
| Accuracy loss from blind quantization | MEDIUM | Add per-layer exclusion options, re-quantize selectively |
| Serialization broke quantized model | LOW if original preserved | Keep original weights, re-quantize with fixed code |
| Hardware incompatibility discovered late | HIGH | Document limitations, add hardware detection early |
| Poor calibration data choice | LOW | Re-calibrate with better data, no retraining needed |
| Format lock-in | HIGH | Add export/conversion utilities, or document migration path |

## Pitfall-to-Phase Mapping

How roadmap phases should address these pitfalls.

| Pitfall | Prevention Phase | Verification |
|---------|------------------|--------------|
| Blind quantization | Phase 1 - Core Quantization | Unit tests with layer exclusion, accuracy benchmarks |
| Hardware validation | Phase 2 - CLI & Validation | CI tests on multiple hardware types |
| Serialization issues | Phase 1 - Core Quantization | Round-trip tests in CI, save/load validation |
| Calibration mismatch | Phase 2 - CLI & Validation | Calibration quality checks, warnings |
| Format incompatibility | Phase 1 - Core Quantization | Interoperability tests with common runtimes |
| Progress reporting | Phase 2 - CLI & Validation | User testing with large models |
| API confusion | Phase 3 - API Polish | Usability testing, documentation review |

## Quantization-Specific Anti-Patterns

Patterns unique to quantization that should be avoided.

### Anti-Pattern 1: Assuming Uniform Sensitivity
**What:** Treating all layers as equally quantizable
**Why bad:** Some layers (embeddings, layer norms) are extremely sensitive to precision loss
**Instead:** Implement layer-type-aware quantization with sensible defaults

### Anti-Pattern 2: Ignoring Outlier Values
**What:** Using simple min/max scaling when data has extreme outliers
**Why bad:** Outliers force wide quantization ranges, wasting precision on normal values
**Instead:** Consider percentile-based clipping or per-channel quantization

### Anti-Pattern 3: No Accuracy Baseline
**What:** Quantizing without measuring original model accuracy
**Why bad:** Cannot determine if quantization caused accuracy loss
**Instead:** Always measure baseline accuracy, compare after quantization

### Anti-Pattern 4: Mixing Precision Without Tracking
**What:** Using different precisions in different places without documenting
**Why bad:** Impossible to debug, difficult to reproduce results
**Instead:** Log quantization scheme per layer, make inspectable

## Severity Assessment Guide

For prioritizing which pitfalls to address first:

**CRITICAL (address in Phase 1):**
- Blind quantization without selectivity
- Serialization/deserialization issues
- Format incompatibility

**HIGH (address in Phase 2):**
- Hardware validation gaps
- Calibration data mismatch
- API confusion

**MEDIUM (address in Phase 3):**
- Progress reporting for large models
- Recovery tooling
- Advanced per-layer configuration

**LOW (nice-to-have):**
- Additional format export options
- Advanced calibration strategies

## Sources

### HIGH Confidence (Official Documentation)
- [PyTorch Quantization Documentation](https://docs.pytorch.org/docs/stable/quantization.html) - Official quantization API, deprecation warnings
- [NVIDIA Developer Blog - Model Quantization](https://developer.nvidia.com/blog/model-quantization-concepts-methods-and-why-it-matters/) - Quantization algorithms, granularity, approaches

### MEDIUM Confidence (Verified Sources)
- [ShadeCoder - Dynamic Quantization Guide](https://www.shadecoder.com/topics/dynamic-quantization-a-comprehensive-guide-for-2025) - Common mistakes with dynamic quantization (Mistakes 1-6)
- [PyTorch Discuss - Quantized model save/load](https://discuss.pytorch.org/t/question-about-quantized-model-save-load/219109) - Serialization issues
- [vLLM Forums - Quantization frustration](https://discuss.vllm.ai/t/a-bit-of-frustration-with-quantization/1720) - Hardware compatibility challenges
- [HuggingFace Discuss - bitsandbytes conflicts](https://discuss.huggingface.co/t/bitsandbytes-conflict-with-accelerate/150275) - Library integration issues

### LOW Confidence (Community/Single Source - Verify Before Use)
- [GitHub - bitsandbytes macOS issues](https://github.com/bitsandbytes-foundation/bitsandbytes/issues/1460) - Platform limitations
- [Medium - Demystifying LLM Quantization Suffixes](https://medium.com/@paul.ilvez/demystifying-llm-quantization-suffixes-what-q4-k-m-q8-0-and-q6-k-really-mean-0ec2770f17d3) - Format confusion issues

### Research Papers (For Future Reference)
- [EMNLP 2025 - Why Do Some Inputs Break Low-Bit LLM Quantization?](https://aclanthology.org/2025.emnlp-main.168.pdf) - Quantization failure modes
- [arXiv 2026 - Mitigating Quantization Error via Regenerating Calibration](https://arxiv.org/html/2601.11200v1) - Calibration error mitigation
- [arXiv 2025 - Mixed-Precision Quantization for Language Models](https://arxiv.org/html/2510.16805v1) - INT4/INT8 accuracy preservation

---
*Pitfalls research for: Mono Quant - Ultra-lightweight quantization package*
*Researched: 2025-02-03*
