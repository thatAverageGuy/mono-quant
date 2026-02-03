# Project Research Summary

**Project:** Mono Quant - Ultra-lightweight, model-agnostic quantization package for PyTorch
**Domain:** Model Quantization Tools / PyTorch Packages
**Researched:** 2026-02-03
**Confidence:** HIGH

## Executive Summary

Mono Quant is a build-phase model quantization package for PyTorch that compresses neural networks to INT4/INT8/FP16 precision. Expert quantization tools follow a layered architecture with clean separation between core quantization logic (observers, mappers, quantizers) and user interfaces (CLI/API). The recommended approach leverages PyTorch's native `torch.ao.quantization` APIs while maintaining strict model-agnostic design—accepting any PyTorch `nn.Module` without tying to HuggingFace or other ecosystems.

Based on research, Mono Quant should launch with a minimalist foundation: only torch as a required dependency, INT8 dynamic/static quantization, per-channel quantization (critical for accuracy), and a Python API. The key differentiator is "model-agnostic + minimal dependencies" at a time when competitors (bitsandbytes, AutoGPTQ) lock users into specific ecosystems. Critical risks include blind quantization without layer selectivity (causes 10-35% accuracy drops), serialization failures losing scale/zero-point metadata, and insufficient hardware validation. These are mitigated by implementing selective quantization from day one, using state_dict-based serialization with round-trip validation, and documenting hardware requirements clearly.

## Key Findings

### Recommended Stack

The stack is intentionally minimal to support the core value proposition: ultra-lightweight quantization that works anywhere PyTorch works. PyTorch 2.9+ provides all necessary quantization APIs (`torch.ao.quantization`) including dynamic, static, and quantization-aware training approaches. Python 3.11-3.12 is the sweet spot—PyTorch 2.9 requires 3.10 minimum, and 3.13 support is still maturing.

**Core technologies:**
- **PyTorch 2.9+**: Quantization APIs, tensor operations — built-in support for INT8/FP16, native quantization workflows
- **Python 3.11-3.12**: Runtime language — best compatibility with PyTorch 2.9+
- **setuptools**: Build backend — mature, PyPA-endorsed, works with `pyproject.toml`

**Development tools (recommended but not blocking):**
- **ruff**: Linter & formatter — 130x faster than Black, replaces Flake8+Black+isort
- **pytest 9.0+**: Test framework — modern with subtests support
- **uv**: Package installer — 10-100x faster than pip, optional

**Intentionally avoided:**
- HuggingFace Transformers/Optimum (violates model-agnostic requirement)
- AutoGPTQ/GPTQ-for-LLaMA (LLM-specific, not general-purpose)
- BitsAndBytes (tightly coupled to HF accelerate)
- Poetry 2.0 (backward compatibility break)

### Expected Features

Quantization tools have well-established table stakes features. Users expect INT8 quantization as baseline (industry standard for ~4x compression), dynamic quantization (no calibration needed), per-channel granularity (per-tensor insufficient for accuracy), and save/export capabilities. Without these, the product feels incomplete.

**Must have (table stakes):**
- **INT8 Quantization** — industry standard, users expect 8-bit baseline with ~4x compression
- **Dynamic Quantization** — easiest method, no calibration data required
- **Per-Channel Quantization** — required for acceptable accuracy (per-tensor inadequate)
- **Python API** — `quantize(model, bits=8)` style interface for automation
- **Save/Export Quantized Weights** — persist quantized models for deployment
- **Model-Agnostic Input** — accept any PyTorch state_dict or model (key differentiator)

**Should have (competitive):**
- **INT4 Quantization** — more aggressive compression, increasingly standard for LLMs
- **CLI Interface** — build-phase tool needs command-line access for CI/CD
- **Group-Wise Quantization** — finer granularity than per-channel, better INT4 accuracy
- **Layer Skipping / Selective Quantization** — skip sensitive layers (e.g., lm_head, embeddings)
- **Calibration Dataset API** — for static quantization, users need easy calibration data provision
- **Safetensors Format** — modern standard, safer than pickle

**Defer (v2+):**
- **Quantization-Aware Training (QAT)** — requires full training pipeline, conflicts with build-phase scope
- **Mixed Precision** — different bits per layer, requires sensitivity analysis
- **Outlier Detection** — LLM.int8() style, valuable for LLMs but complex
- **HuggingFace Integration** — adds heavy dependency, conflicts with model-agnostic design
- **Runtime Inference Engine** — outside scope, quantization is build-time

### Architecture Approach

Lightweight quantization tools follow a layered architecture with four main layers: User Interface (CLI/Python API), Quantization Orchestrator (coordinates workflow), Core Quantization (observers, mappers, quantizers, validators), and PyTorch Integration (module replacement, state_dict utils). The recommended project structure separates core quantization logic (pure math, no PyTorch deps where possible) from PyTorch-specific module replacements.

**Major components:**
1. **Core Quantization (`core/`)** — observers (MinMax, Histogram), mappers (scale/zero-point calculation), quantizers (transformations), schemes (affine, symmetric)
2. **Calibration (`calibration/`)** — calibration data collection via forward passes, separated as it's distinct from pure quantization
3. **Modules (`modules/`)** — QuantizedLinear, QuantizedConv2d replacements for PyTorch modules
4. **Python API (`api/`)** — `@quantize` decorator and `quantize()` function interface
5. **CLI (`cli/`)** — thin wrapper around Python API using click/typer
6. **Serialization (`serialization/`)** — save/load quantized models with metadata preservation
7. **Validation (`validation/`)** — accuracy/size comparison metrics

**Key patterns:**
- **Observer Pattern** — collect tensor stats during calibration without modifying behavior
- **Module Replacement Pattern** — swap `nn.Linear`/`nn.Conv2d` with quantized variants
- **Phase-based Workflow** — Quantize -> Calibrate -> Freeze for static quantization
- **Decorator API Pattern** — `@quantize` decorator for simple one-shot quantization

**Build order (component dependencies):**
1. Core quantization math → 2. Observers → 3. Quantizers → 4. Quantized Modules → 5. Calibration → 6. Serialization → 7. Python API → 8. CLI → 9. Validation

### Critical Pitfalls

Research revealed several high-severity pitfalls that cause real-world quantization failures. The most critical is blind quantization—quantizing all layers indiscriminately causes 10-35% accuracy drops because embeddings, layer norms, and some operations are extremely sensitive to precision loss. Serialization issues are also rampant; PyTorch's quantized model serialization is fragile, and losing scale/zero-point metadata breaks the model entirely.

1. **Blind Quantization Without Selectivity** — implement selective quantization from day one: only quantize appropriate layer types (Linear, attention), exclude sensitive layers (norm, embeddings), provide per-layer exclusion options
2. **Serialization/Deserialization Breaking Quantization State** — use state_dict-based serialization (explicitly documented method), store quantization params alongside weights, validate round-trip preserves outputs in CI
3. **Insufficient Validation on Target Hardware** — quantized models validated on dev GPUs often fail on CPU servers or different architectures; add hardware detection, test on actual deployment hardware, document requirements clearly
4. **Calibration Data Mismatch** — non-representative calibration data causes poor scale/zero-point selection; document calibration requirements (100-500 samples), implement warnings for insufficient data, use percentile-based clipping for outliers
5. **Format Incompatibility Hell** — different quantization formats (GPTQ, AWQ, NF4, GGUF) are incompatible; use standard PyTorch formats, document compatibility with major inference frameworks, avoid proprietary formats

**"Looks Done But Isn't" checklist:**
- INT8 quantization often missing scale/zero-point validation
- Save/load often missing serialization of quantization metadata
- Model-agnostic claims often fail on non-Transformer architectures
- Calibration often missing guidance on data requirements

## Implications for Roadmap

Based on combined research, the roadmap should follow the component dependency order while addressing critical pitfalls early. The architecture research clearly defines the build sequence (core → observers → quantizers → modules → calibration → serialization → API → CLI). Feature research shows MVP requires INT8 dynamic/static, per-channel, Python API, and save/export. Pitfall research emphasizes that selective quantization and robust serialization must be in Phase 1, not bolted on later.

### Phase 1: Core Quantization Foundation

**Rationale:** This phase establishes the foundation that everything else depends on. Architecture research shows core quantization math has no dependencies on other components and can be unit tested with pure tensors. Pitfall research identifies blind quantization and serialization issues as critical—these must be addressed in the foundation, not retrofitted.

**Delivers:**
- Core quantization logic (schemes, mappers, quantizers)
- INT8 dynamic and static quantization
- Per-channel quantization (required for accuracy)
- Selective/layer-aware quantization (avoids blind quantization pitfall)
- Robust save/load with round-trip validation (avoids serialization pitfall)
- QuantizedLinear module replacement
- Python API (`quantize()` function and `@quantize` decorator)

**Addresses:** Table stakes features from FEATURES.md (INT8, dynamic quantization, per-channel, Python API, save/export, model-agnostic input)

**Avoids:** Blind quantization pitfall (selective quantization), serialization pitfall (round-trip validation), format incompatibility (standard PyTorch formats)

**Stack:** Python 3.11-3.12, PyTorch 2.9+, setuptools

**Architecture:** core/schemes.py, core/mappers.py, core/observers.py, core/quantizers.py, modules/linear.py, serialization/state_dict.py, api/functions.py, api/decorators.py

### Phase 2: Calibration, Validation, and CLI

**Rationale:** Once core quantization works, users need calibration infrastructure for static quantization and validation to verify accuracy. The CLI is critical for a build-phase tool (CI/CD integration). This phase addresses the hardware validation and calibration data pitfalls by adding quality checks and hardware detection.

**Delivers:**
- Calibration runner with MinMaxObserver
- Calibration dataset API (accept tensors or data loader)
- Validation metrics (accuracy comparison, size comparison, SQNR)
- CLI interface (click-based, wraps Python API)
- Hardware detection with helpful error messages
- Calibration quality checks and warnings
- Progress reporting for large models

**Uses:** click/rich for CLI, pytest for testing

**Implements:** calibration/runner.py, validation/metrics.py, cli/main.py

**Addresses:** FEATURES.md should-haves (CLI, calibration API, basic metrics), PITFALLS.md hardware validation and calibration mismatch

**Avoids:** Hardware validation pitfall (detection + testing), calibration mismatch pitfall (quality checks)

### Phase 3: Advanced Features (INT4, Group-Wise, Layer Control)

**Rationale:** After MVP is validated, add competitive differentiators. INT4 and group-wise quantization are increasingly standard for LLMs. Layer skipping/control builds on the selective quantization foundation from Phase 1.

**Delivers:**
- INT4 quantization with packing
- Group-wise quantization (finer granularity than per-channel)
- Per-layer configuration override
- Layer skipping (exclude sensitive modules)
- Safetensors format support
- Additional observer types (Histogram, MovingAverage)

**Implements:** INT4 quantizer implementation, group-wise mappers, per-layer config system, safetensors serialization

**Addresses:** FEATURES.md differentiators (INT4, group-wise, layer skipping, safetensors)

### Phase 4: Polish and Ecosystem Integration

**Rationale:** Final phase focuses on usability and optional integrations. Mixed precision and outlier detection are complex but valuable for LLM accuracy. These are deliberately deferred after core product-market fit validation.

**Delivers:**
- Mixed precision (different bits per layer)
- Outlier detection (LLM.int8() style)
- Layer sensitivity analysis
- Module fusion (Conv-BN-ReLU)
- Advanced calibration strategies
- Documentation improvements and examples

**Implements:** Mixed precision quantization, outlier detection, sensitivity analysis, module fusion

**Addresses:** FEATURES.md v2+ features (mixed precision, outlier detection, advanced observers)

**Optional research:** May need `/gsd:research-phase` for outlier detection (LLM.int8() paper reference patterns)

### Phase Ordering Rationale

- **Foundation first:** Phases 1-2 establish the complete quantization workflow (quantize → calibrate → validate → deploy). This is a usable product.
- **Dependencies respected:** The architecture research clearly shows component dependencies (core → observers → quantizers → modules → calibration → API → CLI). Roadmap follows this order.
- **Pitfalls addressed early:** Critical pitfalls (blind quantization, serialization, hardware validation, calibration) are addressed in Phases 1-2, not deferred.
- **MVP then differentiation:** Phase 1 delivers table stakes. Phase 2 makes it practical (CLI, validation). Phases 3-4 add competitive advantages.
- **Risk mitigation:** Selective quantization (prevents accuracy loss) and robust serialization (prevents data loss) are in Phase 1, not afterthoughts.

### Research Flags

**Phases likely needing deeper research during planning:**
- **Phase 3 (INT4, Group-Wise):** INT4 packing implementation details, group-wise scale calculation patterns—research shows these are complex with limited PyTorch native support. May need `/gsd:research-phase` to study torchao reference patterns.
- **Phase 4 (Outlier Detection):** LLM.int8() paper describes outlier detection but implementation details are sparse. Should use `/gsd:research-phase` to study LLM.int8() patterns if pursuing this feature.

**Phases with standard patterns (skip research-phase):**
- **Phase 1 (Core Foundation):** PyTorch native quantization APIs are well-documented, observer patterns are standard. No additional research needed.
- **Phase 2 (Calibration/CLI):** Calibration workflow is standard (PyTorch quantization in practice), CLI patterns with click/typer are well-established. No additional research needed.

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | HIGH | Verified with PyTorch 2.9+ official documentation, PyPI releases, Python packaging standards |
| Features | MEDIUM | Table stakes features from official PyTorch docs, but differentiators based on fragmented ecosystem analysis (bitsandbytes, AutoGPTQ, GGUF) |
| Architecture | HIGH | Based on PyTorch quantization documentation, torchao, optimum-quanto patterns—standard layered architecture is well-established |
| Pitfalls | MEDIUM | Critical pitfalls from official docs (PyTorch quantization), but some from community sources (forums, GitHub issues) |

**Overall confidence:** HIGH

All critical decisions are supported by high-confidence sources (PyTorch official documentation, quantization research papers). Areas with MEDIUM confidence (feature comparisons, some pitfalls) are non-blocking— roadmap can proceed safely, and implementation will validate assumptions.

### Gaps to Address

**Minor gaps (not blocking):**
- **INT4 implementation details:** PyTorch native INT4 support is partial (no native int4 dtype). May need to study torchao reference patterns during Phase 3 planning.
- **Group-wise quantization granularity:** Standard group sizes vary (128 vs 256 vs configurable). Should validate optimal defaults during implementation.
- **Safetensors format specifics:** Format is well-documented but quantization metadata conventions vary. Should verify during Phase 3 implementation.

**How to handle:**
- Gaps are implementation details, not architectural decisions
- Can be resolved during phase planning with targeted research
- Do not block roadmap creation—foundation is solid

## Sources

### Primary (HIGH confidence)
- PyTorch 2.9+ and 2.10 Release Notes — Python 3.10 minimum, quantization API changes
- PyTorch Quantization Documentation — `torch.ao.quantization` APIs, observer patterns, calibration workflow
- PyTorch Quantization in Practice (Blog) — Observer patterns, calibration workflow, QConfig
- LLM.int8() Paper (NeurIPS 2022) — Outlier detection, 8-bit matrix multiplication
- Quantizing Deep Convolutional Networks (Krishnamoorthi, 2018) — Per-channel vs per-tensor
- NVIDIA Developer Blog - Model Quantization — Quantization algorithms, granularity, affine vs symmetric
- optimum-quanto GitHub — Design overview, quantization workflow
- bitsandbytes GitHub — Lightweight design, minimal dependencies pattern

### Secondary (MEDIUM confidence)
- Hugging Face Optimum Documentation — Calibration techniques, quantization schemes
- ShadeCoder - Dynamic Quantization Guide — Common mistakes (1-6)
- PyTorch Discuss Forums — Serialization issues, quantization edge cases
- AutoGPTQ GitHub — GPTQ-based quantization features (project marked unmaintained)
- vLLM Forums — Hardware compatibility challenges
- A Survey of Quantization Methods (Gholami et al., 2021) — Comprehensive quantization techniques

### Tertiary (LOW confidence)
- GitHub issues (bitsandbytes macOS) — Platform limitations, single-source
- Medium articles on LLM quantization formats — Format confusion issues, community perspectives
- Tool comparison blogs (QLoRA vs AWQ vs GPTQ, GGUF vs GPTQ vs AWQ) — Competitor feature analysis

---
*Research completed: 2026-02-03*
*Ready for roadmap: yes*
