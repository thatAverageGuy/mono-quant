# Mono Quant

## What This Is

An ultra-lightweight, model-agnostic quantization package for PyTorch models. Users load models however they want (HuggingFace, PyTorch, anywhere), pass the weights, get quantized weights back. Used in build phase (CI/CD pipelines and local development), not included in deployment artifacts.

## Core Value

Simple, reliable model quantization with minimal dependencies - just torch, no bloat.

## Requirements

### Validated

(None yet - ship to validate)

### Active

- [ ] INT8 quantization - 8-bit integer quantization for model compression
- [ ] INT4 quantization - 4-bit integer for maximum compression
- [ ] FP16 quantization - Float16 for memory reduction without precision loss
- [ ] Dynamic quantization - Per-tensor dynamic quantization
- [ ] CLI interface - Command-line tool for quick quantization
- [ ] Python API - Programmatic interface for automation
- [ ] Model-agnostic - Works with any PyTorch model weights
- [ ] CI/CD compatible - Integrates into build pipelines
- [ ] Local development workflow - Works for developer machines

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

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Model-agnostic design | Users have varied model sources, don't want to lock into HF ecosystem | — Pending |
| Weights-only approach | Simpler API, fewer dependencies, focused responsibility | — Pending |
| CLI + Python API | CI/CD needs automation, humans need convenience | — Pending |
| Build-phase only | Separation of concerns - quantization is build-time optimization | — Pending |

---
*Last updated: 2025-02-03 after initialization*
