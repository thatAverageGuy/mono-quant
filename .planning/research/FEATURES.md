# Feature Research

**Domain:** Model Quantization Tools (PyTorch)
**Researched:** 2026-02-03
**Confidence:** MEDIUM

## Feature Landscape

### Table Stakes (Users Expect These)

Features users assume exist. Missing these = product feels incomplete.

| Feature | Why Expected | Complexity | Notes |
|---------|--------------|------------|-------|
| **INT8 Quantization** | Industry standard for model compression; users expect 8-bit as baseline | MEDIUM | Provides ~4x model size reduction with minimal accuracy loss; supported by all major frameworks |
| **INT4 Quantization** | Popular for LLMs; 4-bit is increasingly standard for memory-constrained deployment | MEDIUM | More aggressive compression; may require calibration for accuracy preservation |
| **FP16 Quantization** | Users expect float16 option as "middle ground" between FP32 and INT8 | LOW | Simple dtype conversion; minimal accuracy loss; often used for intermediate precision |
| **Dynamic Quantization** | Easiest quantization method; users expect "quantize without calibration data" | LOW | Weights quantized, activations quantized on-the-fly; good for LSTMs and Transformers with Linear layers |
| **Per-Channel Quantization** | Standard for weights; per-tensor is insufficient for accuracy | MEDIUM | Different scale per output channel; significantly better accuracy than per-tensor |
| **Scale and Zero-Point Parameters** | Fundamental to affine quantization; users expect control over these | LOW | Scale factor and zero-point bias required for proper FP32 <-> INT8 mapping |
| **Symmetric Quantization Option** | Users expect choice between symmetric and asymmetric schemes | LOW | Symmetric (zero-point = 0) is simpler; asymmetric better for non-negative activations |
| **Save/Export Quantized Weights** | Users need to persist quantized models for deployment | MEDIUM | Save quantized weights + config; support common formats (safetensors, .pt, .bin) |
| **Python API** | Programmatic access required for automation and CI/CD | LOW | Simple function call to quantize; returns quantized model or saves to path |
| **Layer Type Selection** | Users expect to specify which layer types to quantize | LOW | E.g., "quantize Linear layers only" or include Conv2d, LSTM, etc. |
| **Basic Calibration Support** | Even basic tools need min/max observer for scale computation | MEDIUM | MinMaxObserver is baseline; MovingAverageMinMaxObserver is nice-to-have |
| **Quantization Config Persistence** | Quantization parameters must be saved with model | LOW | Config.json or similar storing bits, group_size, sym/asym, scale/zp |

### Differentiators (Competitive Advantage)

Features that set the product apart. Not required, but valuable.

| Feature | Value Proposition | Complexity | Notes |
|---------|-------------------|------------|-------|
| **True Model-Agnostic Design** | Most tools are HF/Transformers locked; we accept any PyTorch model | HIGH | Key differentiator: user loads model how they want, we just quantize weights |
| **Minimal Dependencies (torch-only)** | bitsandbytes, AutoGPTQ require heavy CUDA deps; we keep it light | MEDIUM | Only torch required; everything else optional or std lib |
| **Group-Wise Quantization** | Finer granularity than per-channel; better accuracy for INT4 | HIGH | Divide channels into groups (e.g., 128); each group has own scale/zp |
| **Outlier Detection** | LLM.int8() style handling of outlier features | HIGH | Identify outlier activations, process in higher precision; critical for LLM accuracy |
| **Layer Skipping / Selective Quantization** | Skip sensitive layers to preserve accuracy | MEDIUM | User can specify modules to skip (e.g., "lm_head") or auto-detect sensitivity |
| **Mixed Precision** | Different bit widths for different layers | HIGH | INT8 for sensitive layers, INT4 for robust layers; requires sensitivity analysis |
| **CLI Interface** | Build-phase tools need CLI for CI/CD pipelines | LOW | `monoquant quantize --model model.pt --output model_int8.pt --bits 8` |
| **Calibration Dataset API** | For static quantization, users need easy way to provide calibration data | MEDIUM | Accept list of tensors or data loader; run calibration automatically |
| **Quantization Metrics** | Built-in accuracy/quality metrics help users validate | MEDIUM | SQNR, MSE, perplexity comparison vs FP32; helps identify problematic layers |
| **Dequantization Support** | Convert quantized model back to FP32 | LOW | May lose quality but useful for debugging or format conversion |
| **Safetensors Format** | Modern standard; safer than pickle | LOW | Optional but increasingly expected for model sharing |
| **Module Fusion** | Fuse Conv-BN-ReLU before quantizing for better accuracy | MEDIUM | Reduces quantization error by eliminating intermediate operations |
| **Multiple Observer Types** | MinMax, MovingAverage, Histogram observers | MEDIUM | Different calibration strategies for different use cases |
| **Per-Layer Config Override** | Fine-grained control over quantization parameters | MEDIUM | Override bits, scheme, observer for specific layers/modules |
| **Progress Logging** | Quantization can be slow; users want progress feedback | LOW | Progress bars, layer-by-layer status, estimated time remaining |

### Anti-Features (Commonly Requested, Often Problematic)

Features that seem good but create problems.

| Feature | Why Requested | Why Problematic | Alternative |
|---------|---------------|-----------------|-------------|
| **Quantization-Aware Training (QAT)** | Users want best possible accuracy after quantization | Requires full training pipeline; conflicts with "build-phase only" scope; adds massive complexity | Focus on PTQ with good calibration; point users to QAT libraries if they need training |
| **HuggingFace Transformers Integration** | "Make it work like bitsandbytes" | Locks into HF ecosystem; adds heavy dependency; conflicts with model-agnostic design | Accept any loaded PyTorch model; let user handle loading |
| **Runtime Inference Engine** | "Run the quantized model for me" | Outside scope; quantization is build-time; inference serving is separate problem | Quantize weights, save to standard format; let inference engines handle serving |
| **Automatic Model Loading from Hub** | Convenience for HF users | Adds HF transformers dependency; limits to HF models; not model-agnostic | User loads model however they want; we quantize what's given |
| **Custom CUDA Kernels** | Faster quantization/inference | Breaks "minimal dependencies" principle; platform-specific; complex build process | Use PyTorch native operations; portable across platforms |
| **Model Architecture Modification** | "Optimize the model structure" | Requires architecture knowledge; breaks model-agnostic principle | Weights-only approach; assume user knows their model |
| **Fancy UI/Dashboard** | Visual feedback for quantization process | Build-phase tool doesn't need GUI; adds complexity and dependencies | CLI + Python API with structured logging |
| **Support for Non-PyTorch Frameworks** | "Support TensorFlow, JAX, etc." | Dilutes focus; each framework has different internals | Focus on PyTorch; point to framework-specific tools for others |
| **Automatic Hyperparameter Tuning** | "Find best quantization settings automatically" | Requires evaluation pipeline, dataset, compute; massive scope creep | Provide sensible defaults; expose parameters for manual tuning |
| **Model Zoo Integration** | "Download and quantize popular models" | Requires storage, bandwidth, maintenance; not core value | Documentation examples showing how to quantize common models |
| **ONNX/TFLite Export** | "Export to other formats" | Adds heavy dependencies; conversion is separate problem | Quantize to PyTorch format; use dedicated conversion tools for export |

## Feature Dependencies

```
[Calibration Dataset API]
    └──requires──> [Static Quantization]
                       └──enhances──> [Accuracy Metrics]

[Per-Channel Quantization]
    └──requires──> [Scale/Zero-Point Support]

[Group-Wise Quantization]
    └──requires──> [Per-Channel Quantization]

[Outlier Detection]
    └──enhances──> [INT4 Quantization]
    └──enhances──> [Mixed Precision]

[Layer Skipping]
    └──requires──> [Per-Layer Config Override]
    └──enhances──> [Mixed Precision]

[Mixed Precision]
    └──requires──> [Per-Layer Config Override]
    └──requires──> [Sensitivity Analysis]
                       └──requires──> [Accuracy Metrics]

[CLI Interface]
    └──requires──> [Python API]

[Safetensors Format]
    └──enhances──> [Save/Export Quantized Weights]
```

### Dependency Notes

- **Calibration Dataset API requires Static Quantization:** Dynamic quantization doesn't need calibration data; static quantization pre-computes activation ranges using representative data.
- **Per-Channel required for Group-Wise:** Group-wise quantization is an extension of per-channel where each channel has multiple scale/zero-point pairs.
- **Outlier Detection enhances INT4:** INT4 is more sensitive to outliers; outlier detection (LLM.int8() style) processes outliers in higher precision.
- **Layer Skipping requires Per-Layer Config:** Need mechanism to override quantization settings for specific modules.
- **Mixed Precision requires Sensitivity Analysis:** To intelligently apply different precision to different layers, need to identify which layers are sensitive.
- **CLI requires Python API:** CLI is a thin wrapper around the core Python API.
- **Safetensors enhances Save/Export:** Alternative format to PyTorch's native serialization; safer and increasingly standard.

## MVP Definition

### Launch With (v1)

Minimum viable product — what's needed to validate the concept.

- [x] **INT8 Dynamic Quantization** — Simplest quantization; validates core workflow
- [x] **INT8 Static Quantization** — Requires calibration; validates calibration API
- [x] **Per-Channel Quantization** — Required for acceptable accuracy
- [x] **Python API** — Core interface; `quantize(model, bits=8)` style
- [x] **Save Quantized Weights** — Persist results; standard PyTorch format
- [x] **Basic Calibration (MinMaxObserver)** — Compute scale/zp from calibration data
- [x] **Linear Layer Support** — Most common layer type for quantization
- [x] **Model-Agnostic Input** — Accept any PyTorch state_dict or model

### Add After Validation (v1.x)

Features to add once core is working.

- [ ] **CLI Interface** — Build-phase tool needs command-line access
- [ ] **INT4 Quantization** — More aggressive compression
- [ ] **Group-Wise Quantization** — Better INT4 accuracy
- [ ] **Safetensors Support** — Modern format for safer serialization
- [ ] **Conv2d/LSTM Support** — Beyond Linear layers
- [ ] **Per-Layer Skipping** — Skip sensitive modules
- [ ] **Basic Metrics** — SQNR, size comparison

### Future Consideration (v2+)

Features to defer until product-market fit is established.

- [ ] **Mixed Precision** — Different bits per layer; requires sensitivity analysis
- [ ] **Outlier Detection** — LLM.int8() style; complex but valuable for LLMs
- [ ] **Module Fusion** — Conv-BN-ReLU fusion; improves accuracy
- [ ] **Advanced Observers** — MovingAverage, Histogram calibration
- [ ] **Quantization Metrics Dashboard** — Perplexity, accuracy comparison
- [ ] **Layer Sensitivity Analysis** — Automatic detection of quantization-sensitive layers

## Feature Prioritization Matrix

| Feature | User Value | Implementation Cost | Priority |
|---------|------------|---------------------|----------|
| INT8 Dynamic Quantization | HIGH | LOW | P1 |
| Per-Channel Quantization | HIGH | MEDIUM | P1 |
| Python API | HIGH | LOW | P1 |
| Save Quantized Weights | HIGH | LOW | P1 |
| Model-Agnostic Design | HIGH | MEDIUM | P1 |
| INT8 Static Quantization | HIGH | MEDIUM | P1 |
| CLI Interface | MEDIUM | LOW | P2 |
| INT4 Quantization | HIGH | MEDIUM | P2 |
| Calibration Dataset API | MEDIUM | MEDIUM | P2 |
| Safetensors Support | MEDIUM | LOW | P2 |
| Per-Layer Skipping | MEDIUM | MEDIUM | P2 |
| Basic Metrics | MEDIUM | LOW | P2 |
| Group-Wise Quantization | MEDIUM | HIGH | P3 |
| Mixed Precision | HIGH | HIGH | P3 |
| Outlier Detection | HIGH | HIGH | P3 |
| Module Fusion | MEDIUM | HIGH | P3 |
| Advanced Observers | LOW | MEDIUM | P3 |
| Sensitivity Analysis | MEDIUM | HIGH | P3 |

**Priority key:**
- P1: Must have for launch (MVP)
- P2: Should have, add when possible (v1.x)
- P3: Nice to have, future consideration (v2+)

## Competitor Feature Analysis

| Feature | PyTorch Native | bitsandbytes | AutoGPTQ | GGUF/llama.cpp | Mono Quant |
|---------|---------------|--------------|----------|----------------|------------|
| INT8 Quantization | Yes | Yes (LLM.int8) | Yes | Yes | Yes |
| INT4 Quantization | Limited (torchao) | Yes (QLoRA/NF4) | Yes | Yes | Yes |
| FP16 Quantization | Yes (dtype cast) | No | No | Yes | Yes |
| Dynamic Quantization | Yes | No | No | No | Yes |
| Static Quantization | Yes | No | Yes | No | Yes |
| Per-Channel | Yes | Yes | Yes | Yes | Yes |
| Group-Wise | Limited | No (128 fixed) | Yes (configurable) | Yes (K-quants) | Planned |
| Model-Agnostic | Yes | Via HF only | Via HF only | Via conversion only | Yes (core value) |
| Dependencies | torch | torch + CUDA | torch + CUDA | C++ (cpp) | torch only |
| CLI | No | No (via HF) | No | Yes | Yes |
| Python API | Yes | Yes (HF integration) | Yes | Yes (binding) | Yes |
| Calibration Options | Multiple observers | No | No | No | Multiple planned |
| Layer Skipping | Yes (qconfig_dict) | Yes (skip_modules) | No | No | Planned |
| Mixed Precision | Limited | No | No | Yes (per-block) | Planned |
| Outlier Detection | No | Yes (LLM.int8) | No | No | Planned |
| QAT | Yes | No | No | No | Out of scope |

## Sources

### Official Documentation (HIGH Confidence)

- [PyTorch Quantization in Practice](https://pytorch.org/blog/quantization-in-practice/) — Core quantization concepts, per-channel vs per-tensor, observers, calibration
- [PyTorch AO (torch/ao)](https://github.com/pytorch/ao) — Native PyTorch quantization library; INT8/INT4 support
- [Hugging Face Bitsandbytes Documentation](https://huggingface.co/docs/transformers/quantization/bitsandbytes) — LLM.int8(), QLoRA/NF4, outlier detection, skipping modules

### Research Papers (HIGH Confidence)

- [LLM.int8() Paper (NeurIPS 2022)](https://proceedings.neurips.cc/paper_files/paper/2022/file/c3ba4962c05c49636d4c6206a97e9c8a-Paper-Conference.pdf) — 8-bit matrix multiplication with outlier detection
- [Quantizing Deep Convolutional Networks (Krishnamoorthi, 2018)](https://arxiv.org/pdf/1806.08342) — Per-channel vs per-tensor, best practices
- [A Survey of Quantization Methods (Gholami et al., 2021)](https://arxiv.org/abs/2103.13630) — Comprehensive quantization techniques survey

### Tool Comparisons (MEDIUM Confidence)

- [AutoGPTQ GitHub](https://github.com/AutoGPTQ/AutoGPTQ) — GPTQ-based quantization package; features and limitations (Note: project marked unmaintained)
- [Quantization Without Tears: QLoRA vs AWQ vs GPTQ](https://medium.com/@hadiyolworld007/quantization-without-tears-qlora-vs-awq-vs-gptq-1a904d21a46a) — Comparison of popular LLM quantization methods
- [Which Quantization Method Is Best: GGUF, GPTQ, AWQ](https://www.e2enetworks.com/blog/which-quantization-method-is-best-for-you-gguf-gptq-or-awq) — Format comparison for LLM inference

### Technical Deep Dives (MEDIUM Confidence)

- [Per-Channel vs Per-Tensor Quantization](https://www.oreateai.com/blog/understanding-perchannel-vs-pertensor-quantization-a-deep-dive/b8a88e1024816bb93518f9bad4c5f62c) — Granularity explanation and impact
- [Quantization Granularity](https://medium.com/@curiositydeck/quantization-granularity-aec2dd7a0bb4) — Per-tensor, per-channel, per-group comparison
- [Understanding LLM.int8()](https://picovoice.ai/blog/understanding-llm-int8/) — Outlier detection explanation
- [Model Quantization: Concepts and Methods](https://developer.nvidia.com/blog/model-quantization-concepts-methods-and-why-it-matters/) — NVIDIA's quantization fundamentals

### Quality and Evaluation (LOW-MEDIUM Confidence)

- [On the Impact of Calibration Data in PTQ](https://liner.com/review/on-impact-calibration-data-in-posttraining-quantization-and-pruning) — Calibration data best practices
- [Metrics for Quantized LLM Evaluation](https://apxml.com/courses/practical-llm-quantization/chapter-6-evaluating-deploying-quantized-llms/evaluating-quantized-models) — Perplexity and accuracy metrics
- [Post-Training Quantization vs Quantization-Aware Training](https://ai.plainenglish.io/post-training-quantization-vs-quantization-aware-training-a-hands-on-comparison-with-a-small-llama-bc53e1fbb6d2) — PTQ vs QAT comparison

---
*Feature research for: Mono Quant - Model Quantization Tools*
*Researched: 2026-02-03*
