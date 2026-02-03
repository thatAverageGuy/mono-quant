# Mono Quant

**Ultra-lightweight, model-agnostic quantization for PyTorch**

[![PyPI Version](https://img.shields.io/pypi/v/mono-quant)](https://pypi.org/project/mono-quant/)
[![Python Version](https://img.shields.io/pypi/pyversions/mono-quant)](https://pypi.org/project/mono-quant/)
[![License](https://img.shields.io/github/license/thatAverageGuy/mono-quant)](https://github.com/thatAverageGuy/mono-quant/blob/main/LICENSE)
[![Documentation](https://img.shields.io/badge/docs-mkdocs-blue)](https://thataverageguy.github.io/mono-quant)
[![CI/CD](https://github.com/thatAverageGuy/mono-quant/actions/workflows/ci.yml/badge.svg)](https://github.com/thatAverageGuy/mono-quant/actions/workflows/ci.yml)

## What is Mono Quant?

Mono Quant is a simple, reliable model quantization package for PyTorch with minimal dependencies. Just `torch` and `numpy`, no bloat.

### Key Features

- **Model-Agnostic** - Works with any PyTorch model: HuggingFace, local, or custom
- **Multiple Modes** - INT8, INT4, and FP16 quantization
- **Flexible Calibration** - Dynamic (no data) or static (with calibration data)
- **Robust Validation** - SQNR metrics, size comparison, and accuracy warnings
- **Dual Interface** - Python API for automation, CLI for CI/CD
- **Build-Phase Only** - Quantize during build, deploy lightweight models

## Installation

```bash
pip install mono-quant
```

### Requirements

- Python 3.11 or higher
- PyTorch 2.0 or higher
- NumPy 1.24 or higher

## Quick Start

### Python API

```python
from mono_quant import quantize

# Dynamic INT8 quantization (no calibration data needed)
result = quantize(model, bits=8, dynamic=True)

# Save the quantized model
result.save("model_quantized.pt")

# Check metrics
print(f"Compression: {result.info.compression_ratio:.2f}x")
print(f"SQNR: {result.info.sqnr_db:.2f} dB")
```

### CLI

```bash
# Dynamic quantization
monoquant quantize --model model.pt --bits 8 --dynamic

# With custom output path
monoquant quantize --model model.pt --bits 8 --output model_quantized.pt
```

## Quantization Modes

### Dynamic Quantization (Fastest, No Data)

```python
result = quantize(model, bits=8, dynamic=True)
```

### Static Quantization (Best Accuracy, Requires Data)

```python
calibration_data = [torch.randn(1, 3, 224, 224) for _ in range(150)]

result = quantize(
    model,
    bits=8,
    dynamic=False,
    calibration_data=calibration_data
)
```

### INT4 Quantization (Maximum Compression)

```python
result = quantize(
    model,
    bits=4,
    dynamic=False,
    calibration_data=calibration_data,
    group_size=128  # Default
)
```

## Documentation

Full documentation available at **[https://thataverageguy.github.io/mono-quant](https://thataverageguy.github.io/mono-quant)**

- [Installation Guide](https://thataverageguy.github.io/mono-quant/getting-started/installation/)
- [Quick Start](https://thataverageguy.github.io/mono-quant/getting-started/quickstart/)
- [User Guide](https://thataverageguy.github.io/mono-quant/user-guide/modes/)
- [CLI Reference](https://thataverageguy.github.io/mono-quant/cli/)
- [API Reference](https://thataverageguy.github.io/mono-quant/api/)

## Why Mono Quant?

Most quantization tools are tied to specific frameworks (HuggingFace, TFLite) or require heavy dependencies. Mono Quant fills the niche of **"just quantize the weights, nothing else."**

### Design Philosophy

| Aspect | Approach |
|--------|----------|
| **Model Loading** | You load the model, we quantize it |
| **Dependencies** | Only torch and numpy required |
| **Use Case** | Build-phase (CI/CD, local development) |
| **Scope** | Quantization only, no runtime or serving |

## License

MIT License - see [LICENSE](https://github.com/thatAverageGuy/mono-quant/blob/main/LICENSE) for details.

## Contributing

Contributions welcome! Please see [CONTRIBUTING.md](https://thataverageguy.github.io/mono-quant/about/contributing/) for guidelines.

## Links

- **GitHub:** https://github.com/thatAverageGuy/mono-quant
- **PyPI:** https://pypi.org/project/mono-quant/
- **Documentation:** https://thataverageguy.github.io/mono-quant
- **Issues:** https://github.com/thatAverageGuy/mono-quant/issues
