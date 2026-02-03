# Mono Quant

## What This Is

An ultra-lightweight, model-agnostic quantization package for PyTorch models. Users load models however they want (HuggingFace, PyTorch, anywhere), pass the weights, get quantized weights back. Used in build phase (CI/CD pipelines and local development), not included in deployment artifacts.

## Core Value

Simple, reliable model quantization with minimal dependencies - just torch, no bloat.

## Requirements

### Validated

- ✓ INT8 quantization — v1.0 - 8-bit integer quantization with per-channel scaling
- ✓ INT4 quantization — v1.0 - 4-bit integer with group-wise scaling
- ✓ FP16 quantization — v1.0 - Float16 for memory reduction
- ✓ Dynamic quantization — v1.0 - Per-tensor dynamic quantization without calibration
- ✓ Static quantization — v1.0 - Calibration with 3 observer types (MinMax, MovingAverage, Histogram)
- ✓ CLI interface — v1.0 - Command-line tool with git-style subcommands
- ✓ Python API — v1.0 - Unified `quantize()` function with `QuantizationResult`
- ✓ Model-agnostic — v1.0 - Works with any PyTorch model (nn.Module, state_dict, file path)
- ✓ CI/CD compatible — v1.0 - Integrates into build pipelines with progress bars
- ✓ Serialization — v1.0 - PyTorch and Safetensors format support
- ✓ Validation — v1.0 - SQNR metrics, size comparison, load testing
- ✓ Layer skipping — v1.0 - Default skip list for INT4 protects sensitive layers

### Active

(None - all v1 requirements validated. See REQUIREMENTS.md for future work.)

### Out of Scope

- HuggingFace transformers dependency - Users load models themselves, we only quantize
- Tokenizers and pipelines - Not a model serving/running tool
- Model loading - User provides loaded model or weights file path
- Fancy UI/dashboard - Logging is practical, not visual
- Inference runtime - Quantization happens during build, not serving
- Model architecture awareness - Weights-only, no structural assumptions

## Context

**Target users:** ML engineers in enterprise environments who need simple, reliable quantization without ecosystem baggage.

**Project nature:** Open source, learning/fun project but production-ready for enterprise use.

**Workflow:** Used during build phase - CI/CD pipelines quantize before deployment, developers quantize locally for testing. Quantized models are deployment artifacts, but mono-quant itself is not deployed.

**Current ecosystem:** Most quantization tools are tied to specific frameworks (HuggingFace, TFLite) or require heavy dependencies. Mono-quant fills the niche of "just quantize the weights, nothing else."

## Constraints

- **Tech stack**: Python, PyTorch - Standard ML stack, minimal extras
- **Dependencies**: torch required, everything else optional or std lib - Keep package lightweight
- **Compatibility**: Must work with any PyTorch model weights - Model-agnostic design
- **Use case**: Build-phase only - Not designed for runtime deployment

## Context

**Target users:** ML engineers in enterprise environments who need simple, reliable quantization without ecosystem baggage.

**Project nature:** Open source, learning/fun project but production-ready for enterprise use.

**Workflow:** Used during build phase - CI/CD pipelines quantize before deployment, developers quantize locally for testing. Quantized models are deployment artifacts, but mono-quant itself is not deployed.

**Current ecosystem:** Most quantization tools are tied to specific frameworks (HuggingFace, TFLite) or require heavy dependencies. Mono-quant fills the niche of "just quantize the weights, nothing else."

**Current State (v1.0 Shipped):**
- 5,228 lines of Python across 26 files
- 30 requirements delivered (100% of v1 scope)
- 4 phases, 13 plans, 66 must-haves verified
- All 8 cross-phase integration points working
- All 8 end-to-end flows complete
- No technical debt identified

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Model-agnostic design | Users have varied model sources, don't want to lock into HF ecosystem | ✓ Good - Verified to work with custom, HF-like, and pretrained models |
| Weights-only approach | Simpler API, fewer dependencies, focused responsibility | ✓ Good - Clean API with _prepare_model() handling both nn.Module and state_dict |
| CLI + Python API | CI/CD needs automation, humans need convenience | ✓ Good - Unified `quantize()` function and CLI `monoquant` command both working |
| Build-phase only | Separation of concerns - quantization is build-time optimization | ✓ Good - No runtime dependencies, quantized models deploy without mono-quant |
| Local imports pattern | Avoid circular dependencies between modules | ✓ Good - Fixed circular import between core.quantizers and modules.linear |
| Group-wise INT4 scaling | Balance compression ratio and accuracy | ✓ Good - 2x compression vs INT8, skip list protects sensitive layers |

---

*Last updated: 2026-02-03 after v1.0 milestone*

---
*Last updated: 2025-02-03 after initialization*
