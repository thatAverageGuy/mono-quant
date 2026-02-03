# Stack Research

**Domain:** Model Quantization Package for PyTorch
**Researched:** 2026-02-03
**Confidence:** HIGH

## Recommended Stack

### Core Framework

| Technology | Version | Purpose | Why Recommended |
|------------|---------|---------|-----------------|
| **Python** | 3.11 - 3.12 | Runtime language | Sweet spot for PyTorch stability: 3.10 is minimum, 3.13 support still maturing. 3.11-3.12 offer best compatibility with latest PyTorch releases. |
| **PyTorch** | 2.9+ | Tensor computation, quantization APIs | Built-in quantization support (`torch.quantization`, `torch.ao.quantization`). Python 3.10+ minimum required as of 2.9. Native PTQ, QAT, dynamic quantization APIs. |
| **NumPy** | 1.26+ (or 2.0+) | Numerical operations | Optional but useful for tensor operations during calibration/scale computation. PyTorch tensors can often replace, but NumPy is standard ecosystem dependency. |

### Supporting Libraries

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| **typing-extensions** | 4.12+ | Type hints compatibility | When using modern type hints (e.g., `Never`, `TypeIs`) that need backport to Python 3.11. Recommended for robust type annotations. |
| **click** | 8.1+ | CLI interface | When building CLI commands. Recommended for `mono-quant` CLI - mature, well-maintained, excellent help text generation. |
| **rich** | 13.7+ | Terminal output formatting | Optional, but highly recommended for CLI. Beautiful progress bars, tables, and colored output make quantization feedback much clearer. |

### Development Tools

| Tool | Purpose | Notes |
|------|---------|-------|
| **setuptools** | Build backend | Mature, stable, widely supported. Use with `pyproject.toml` (not `setup.py`). PyPA-endorsed backend. |
| **build** | Build frontend | Standard tool for building packages (`python -m build`). Required for creating distributable wheels. |
| **uv** | Package installer | 10-100x faster than pip. Optional but recommended for development. Replaces pip, pip-tools, virtualenv. |
| **ruff** | Linter & formatter | 130x faster than Black. Replaces Flake8, Black, isort. Single tool for linting, formatting, import sorting. |
| **pytest** | 9.0+ | Test framework | Modern pytest with subtests support (9.0+). Standard for Python testing. |
| **pytest-cov** | Latest | Coverage reporting | Use with pytest for coverage reports during CI/CD. |
| **mypy** | 1.11+ | Static type checking | Recommended for codebase quality. PyTorch has stubs; type hints catch quantization bugs. |

## Installation

```bash
# Core dependencies (required)
pip install torch>=2.9.0

# Optional but recommended
pip install numpy click rich

# Development dependencies
pip install -e ".[dev]"  # See pyproject.toml for extras
# Or individually:
pip install setuptools build
pip install ruff pytest pytest-cov mypy
```

## Alternatives Considered

| Recommended | Alternative | When to Use Alternative |
|-------------|-------------|-------------------------|
| **setuptools** | Hatchling | Hatchling is simpler but newer. Setuptools has broader ecosystem compatibility and more mature plugin support. |
| **setuptools** | Flit | Flit is minimal but less flexible for complex packages. Setuptools handles entry points and extras more conventionally. |
| **setuptools** | PDM | PDM is great but introduces a new dependency management paradigm. Setuptools + uv keeps it simpler. |
| **click** | Typer | Typer is more "magical" with type hints but adds more dependencies. Click is more explicit and lighter. |
| **ruff** | Black + Flake8 + isort | Ruff is faster (130x) and unified. Use the trio only if you need specific plugins not in Ruff. |
| **pytest** | unittest | Unittest is built-in but verbose. pytest's fixtures and parametrize are essential for quantization testing. |
| **uv** | pip | uv is 10-100x faster. Use pip only if uv has compatibility issues (rare). |

## What NOT to Use

| Avoid | Why | Use Instead |
|-------|-----|-------------|
| **HuggingFace Transformers** | Ties package to HF ecosystem, massive dependencies (datasets, tokenizers, safetensors), violates "model-agnostic" requirement | Use raw PyTorch - your package should work with ANY `nn.Module`, not just HF models |
| **HuggingFace Optimum** | Built around HF models, brings in HF bloat, not model-agnostic | Use PyTorch native `torch.ao.quantization` APIs |
| **AutoGPTQ / GPTQ-for-LLaMA** | LLM-specific, optimized for transformer attention patterns, not model-agnostic | Build general quantization that works on CNNs, RNNs, etc. |
| **BitsAndBytes** | Tightly coupled to HF `accelerate`, 8-bit optimizer focus not aligned with build-phase quantization goal | Use torchao patterns or custom INT4/INT8 implementations |
| **torchao** (as dependency) | Still evolving rapidly, integrates with HF Transformers, has training-phase focus | Study torchao for reference patterns, but don't depend on it - use native torch |
| **OpenVINO** | Intel-specific, ONNX dependency, not pure PyTorch | Keep it simple with torch only |
| **TensorFlow / TFLite** | Different framework entirely | Not applicable - PyTorch only |
| **ONNX** | Unnecessary conversion layer, adds complexity | Direct PyTorch quantization is simpler and more direct |
| **Poetry (for dependency mgmt)** | Poetry 2.0 just broke backward compatibility, mixing build+dependency mgmt adds complexity | Use setuptools + uv (build/install separation) |
| **Conda** | Anaconda channel deprecated by PyTorch as of 2.6, overkill for a single-package project | pip/uv with virtualenv or uv venv |
| **setup.py** | Legacy format, deprecated | `pyproject.toml` is the modern standard |

## Stack Patterns by Variant

**If building only core quantization (no CLI):**
- Omit `click` and `rich`
- Keep package purely library-focused
- Users can import and call quantize directly

**If including CLI (recommended for "install, import, quantize, deploy" workflow):**
- Add `click` for command parsing
- Add `rich` for user-friendly output
- Create entry point via `pyproject.toml` `[project.scripts]`

**If supporting development on Windows:**
- Python 3.11-3.12 fully supported by PyTorch on Windows
- Ruff works cross-platform
- All recommended tools have Windows wheels

**If supporting Linux GPU servers:**
- Consider CUDA-specific PyTorch wheels
- Build-phase quantization typically runs on CPU (calibration), but may need GPU for validation

## Version Compatibility

| Package A | Compatible With | Notes |
|-----------|-----------------|-------|
| **torch 2.9+** | Python 3.10, 3.11, 3.12, 3.13 (nightly) | Python 3.10 is minimum. 3.13 support in nightly. |
| **torch 2.9+** | NumPy 1.x, 2.x | PyTorch is compatible with both NumPy generations |
| **ruff** | Python 3.8+ | Supports all target Python versions |
| **pytest 9.0+** | Python 3.9+ | Supports all target Python versions |
| **mypy 1.11+** | Python 3.8+ | Supports all target Python versions |

## Quantization-Specific Technical Notes

### PyTorch Native Quantization APIs (2.9+)

PyTorch provides three main quantization approaches in `torch.ao.quantization`:

1. **Post-Training Dynamic Quantization** (`quantize_dynamic`)
   - Weights quantized, activations computed in float
   - Fastest, no calibration data needed
   - Best for LSTM/Transformer models

2. **Post-Training Static Quantization** (`quantize`)
   - Both weights and activations quantized
   - Requires calibration dataset
   - Best for CNN/inference-heavy models

3. **Quantization-Aware Training** (`quantize_qat`)
   - Simulates quantization during training
   - Best accuracy for low-bit (INT4) quantization

### Bit Width Support

| Bit Width | PyTorch Native Status | Implementation Notes |
|-----------|----------------------|---------------------|
| **FP16** | Native (torch.float16) | Trivial conversion, CUDA accelerated |
| **INT8** | Native (torch.qint8) | Fully supported, mature kernels |
| **INT4** | Partial | No native int4 dtype; need to implement packing or study torchao patterns |
| **Dynamic** | Native | `quantize_dynamic` API |

### Minimal Dependency Principle

The entire quantization package can be built with **only torch as required dependency**:

- Quantization APIs are in `torch.ao.quantization`
- All tensor operations available in core torch
- No external ML libraries needed for basic quantization

Optional dependencies (`click`, `rich`, `numpy`) are for:
- CLI interface convenience
- Numerical operations during scale calculation
- Better user experience

## Sources

- PyTorch 2.10.0 Release Notes (HIGH) - https://github.com/pytorch/pytorch/releases
- PyTorch 2.9.0 Release Notes (HIGH) - Confirms Python 3.10 minimum requirement, 3.14 preview
- PyTorch on PyPI (HIGH) - https://pypi.org/project/torch/ - Latest version 2.10.0
- TorchAO on PyPI (HIGH) - https://pypi.org/project/torchao/ - For reference patterns, not dependency
- PyTorch 2.5 Release Notes (MEDIUM) - CuDNN backend, quantization features
- Python Packaging Standards (MEDIUM) - https://roel.rs/doc/python3-setuptools/html/userguide/pyproject_config.html
- PyPA Build Backends (MEDIUM) - Setuptools, Hatchling, Flit, PDM endorsed options
- UV Package Manager (LOW) - 10-100x faster than pip, Rust-based
- Ruff Linter (LOW) - 130x faster than Black, replaces Flake8+Black+isort
- pytest 9.0 Release (MEDIUM) - Subtests support, latest stable version
- Quantization Explained (LOW) - INT8, INT4, GPTQ for production deployment
- LLM Quantization Resources (LOW) - https://github.com/pprp/Awesome-LLM-Quantization
- Neural Network Quantization in PyTorch (LOW) - https://arikpoz.github.io/posts/2025-04-16-neural-network-quantization-in-pytorch/

---
*Stack research for: Mono Quant - Ultra-lightweight, model-agnostic quantization package for PyTorch models*
*Researched: 2026-02-03*
