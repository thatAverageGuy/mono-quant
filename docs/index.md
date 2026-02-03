# Mono Quant

**Ultra-lightweight, model-agnostic quantization for PyTorch**

[![PyPI Version](https://img.shields.io/pypi/v/mono-quant)](https://pypi.org/project/mono-quant/)
[![Python Version](https://img.shields.io/pypi/pyversions/mono-quant)](https://pypi.org/project/mono-quant/)
[![License](https://img.shields.io/github/license/thatAverageGuy/mono-quant)](https://github.com/thatAverageGuy/mono-quant/blob/main/LICENSE)

---

## What is Mono Quant?

Mono Quant is a simple, reliable model quantization package for PyTorch with minimal dependencies. Just `torch`, no bloat.

### Key Features

- **Model-Agnostic** - Works with any PyTorch model: HuggingFace, local, or custom
- **Multiple Modes** - INT8, INT4, and FP16 quantization
- **Flexible Calibration** - Dynamic (no data) or static (with calibration data)
- **Robust Validation** - SQNR metrics, size comparison, and accuracy warnings
- **Dual Interface** - Python API for automation, CLI for CI/CD
- **Build-Phase Only** - Quantize during build, deploy lightweight models

### Installation

```bash
pip install mono-quant
```

### Quick Start

```python
from mono_quant import quantize

# Quantize a model to INT8
result = quantize(model, bits=8, dynamic=True)

# Save the quantized model
result.save("model_quantized.pt")

# Check metrics
print(f"Compression: {result.info.compression_ratio:.2f}x")
print(f"SQNR: {result.info.sqnr_db:.2f} dB")
```

Or use the CLI:

```bash
monoquant quantize --model model.pt --bits 8 --dynamic
```

---

## Why Mono Quant?

Most quantization tools are tied to specific frameworks (HuggingFace, TFLite) or require heavy dependencies. Mono Quant fills the niche of **"just quantize the weights, nothing else."**

### Design Philosophy

| Aspect | Approach |
|--------|----------|
| **Model Loading** | You load the model, we quantize it |
| **Dependencies** | Only `torch` required |
| **Use Case** | Build-phase (CI/CD, local development) |
| **Scope** | Quantization only, no runtime or serving |

---

## Quantization Modes

### Dynamic Quantization (No Calibration)

Fastest option, no data required. Good for inference speedup.

```python
result = quantize(model, bits=8, dynamic=True)
```

### Static Quantization (With Calibration)

Best accuracy, requires representative data.

```python
result = quantize(
    model,
    bits=8,
    dynamic=False,
    calibration_data=calibration_tensors
)
```

### INT4 Quantization

Maximum compression with group-wise scaling.

```python
result = quantize(
    model,
    bits=4,
    dynamic=False,
    calibration_data=calibration_tensors,
    group_size=128  # Default
)
```

---

## What's Next?

- [**Installation Guide**](getting-started/installation.md) - Set up Mono Quant
- [**Quick Start**](getting-started/quickstart.md) - Step-by-step tutorial
- [**User Guide**](user-guide/modes.md) - Deep dive into features
- [**CLI Reference**](cli/commands.md) - Command-line usage
- [**API Reference**](api/quantize.md) - Python API details
- [**Examples**](examples/dynamic-int8.md) - Real-world code samples

---

## License

MIT License - see [LICENSE](https://github.com/thatAverageGuy/mono-quant/blob/main/LICENSE) for details.
