# Mono Quant v1.0 - Initial Release

Ultra-lightweight, model-agnostic quantization package for PyTorch models.

## 🎯 What is Mono Quant?

Mono Quant is a simple, reliable model quantization package for PyTorch with minimal dependencies. Just `torch`, no bloat.

## ✨ Key Features

### Core Quantization
- ✅ **INT8 quantization** with per-channel scaling
- ✅ **INT4 quantization** with group-wise scaling (2x compression vs INT8)
- ✅ **FP16 quantization** for memory reduction
- ✅ **Dynamic quantization** (no calibration data required)
- ✅ **Static quantization** with calibration data

### Calibration
- ✅ **MinMaxObserver** (default, fast)
- ✅ **MovingAverageMinMaxObserver** (robust, EMA smoothing)
- ✅ **HistogramObserver** (outlier-aware, KL divergence)
- ✅ Calibration data from tensors or DataLoader

### User Interface
- ✅ Unified `quantize()` Python API
- ✅ `QuantizationResult` with `.save()` and `.validate()` methods
- ✅ CLI with git-style subcommands (`monoquant`)
- ✅ Progress bars with CI/TTY auto-detection

### Serialization
- ✅ PyTorch format (.pt/.pth) support
- ✅ Safetensors format support
- ✅ Metadata preservation (bits, scheme, scales, zero-points)
- ✅ Model dequantization back to FP32

### Validation
- ✅ SQNR (signal-to-quantization-noise ratio) computation
- ✅ Model size comparison
- ✅ Load testing (round-trip validation)
- ✅ Accuracy warnings for aggressive quantization

### Advanced Features
- ✅ Model-agnostic design (any PyTorch model)
- ✅ Layer skipping for INT4 (protects sensitive layers)
- ✅ Symmetric and asymmetric quantization schemes
- ✅ Custom exception hierarchy with actionable suggestions

## 📊 Statistics

- **Requirements delivered:** 30/30 (100%)
- **Integration points:** 8/8 verified
- **E2E flows:** 8/8 working
- **Lines of code:** 5,228 Python
- **Files:** 26 source files
- **Technical debt:** None identified

## 📦 Installation

```bash
pip install mono-quant
```

## 🚀 Quick Start

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

# With custom output
monoquant quantize --model model.pt --bits 8 --output model_quantized.pt
```

## 💡 Use Cases

- **CI/CD Pipelines** - Automate quantization during build
- **Local Development** - Test quantized models before deployment
- **Model Compression** - Reduce model size by 4-8x
- **Inference Speedup** - Faster inference with quantized models

## 🔧 Requirements

- **Python:** 3.8 or higher
- **PyTorch:** 2.0 or higher

### Optional Dependencies

- `safetensors>=0.4` - For Safetensors format support
- `click>=8.1` - For CLI
- `tqdm>=4.66` - For progress bars

## 📚 Documentation

Full documentation available at: **https://thataverageguy.github.io/mono-quant**

- Installation guide
- Quick start tutorial
- User guide (modes, calibration, INT4, layer skipping)
- CLI reference
- API documentation
- Examples and tutorials

## 🎁 What's Included

- Model-agnostic quantization (works with HuggingFace, local, or custom models)
- Dynamic and static quantization modes
- INT8, INT4, and FP16 support
- Robust calibration with 3 observer types
- Layer skipping to protect sensitive components
- Serialization to PyTorch and Safetensors formats
- Validation with SQNR metrics and accuracy warnings
- Python API and CLI for automation

## 🚧 Known Limitations

- CLI does not support loading calibration data from files (use Python API)
- INT4 quantization requires calibration data (no dynamic INT4)
- No quantization-aware training (QAT) - build-phase only
- No ONNX/TFLite export (use dedicated conversion tools)

## 🗺️ Roadmap

### v2 (Future)

- Genetic optimization for quantization parameters
- Experiment tracking and logging
- Mixed precision (different bits per layer)
- LLM.int8() style outlier detection
- Automatic layer sensitivity analysis

## 📄 License

MIT License - see [LICENSE](https://github.com/thatAverageGuy/mono-quant/blob/main/LICENSE) for details.

## 🙏 Contributing

Contributions welcome! Please see [CONTRIBUTING.md](https://thataverageguy.github.io/mono-quant/about/contributing/) for guidelines.

## 📞 Support

- **Issues:** https://github.com/thatAverageGuy/mono-quant/issues
- **Documentation:** https://thataverageguy.github.io/mono-quant
- **PyPI:** https://pypi.org/project/mono-quant/

---

**Full Changelog:** https://thataverageguy.github.io/mono-quant/about/changelog/
