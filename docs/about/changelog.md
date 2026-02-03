# Release Notes

## v1.0 - Mono Quant Initial Release (2026-02-03)

### Features

#### Core Quantization
- ✅ INT8 quantization with per-channel scaling
- ✅ INT4 quantization with group-wise scaling (group_size=128)
- ✅ FP16 quantization for memory reduction
- ✅ Dynamic quantization (no calibration data required)
- ✅ Static quantization with calibration

#### Calibration
- ✅ MinMaxObserver (default, fast)
- ✅ MovingAverageMinMaxObserver (robust, EMA smoothing)
- ✅ HistogramObserver (outlier-aware, KL divergence)
- ✅ Calibration data from tensors or DataLoader

#### User Interface
- ✅ Unified `quantize()` Python API
- ✅ `QuantizationResult` with `.save()` and `.validate()` methods
- ✅ CLI with git-style subcommands (`monoquant`)
- ✅ Progress bars with CI/TTY auto-detection

#### Serialization
- ✅ PyTorch format (.pt/.pth) support
- ✅ Safetensors format support
- ✅ Metadata preservation (bits, scheme, scales, zero-points)
- ✅ Model dequantization back to FP32

#### Validation
- ✅ SQNR (signal-to-quantization-noise ratio) computation
- ✅ Model size comparison
- ✅ Load testing (round-trip validation)
- ✅ Accuracy warnings for aggressive quantization

#### Advanced Features
- ✅ Model-agnostic design (any PyTorch model)
- ✅ Layer skipping for INT4 (protects sensitive layers)
- ✅ Symmetric and asymmetric quantization schemes
- ✅ Custom exception hierarchy with actionable suggestions
- ✅ Zero-point clamping to prevent runtime errors

### Statistics

- **Requirements delivered:** 30/30 (100%)
- **Integration points:** 8/8 verified
- **E2E flows:** 8/8 working
- **Lines of code:** 5,228 Python
- **Files:** 26 source files
- **Technical debt:** None identified

### Dependencies

**Required:**
- Python >= 3.11
- torch >= 2.0
- numpy >= 1.24

**Included:**
- safetensors >= 0.4 (Safetensors format)
- click >= 8.1 (CLI)
- tqdm >= 4.66 (progress bars)

**Optional:**
- mkdocs >= 1.6.0 (documentation)
- mkdocs-material >= 9.7.0 (documentation theme)
- mkdocstrings[python] >= 1.0.0 (API documentation)

### Documentation

- Installation guide
- Quick start tutorial
- Basic usage examples
- CLI reference
- API documentation
- User guide
- Examples (dynamic INT8, static INT4, custom observer, CI/CD)

### Known Limitations

- CLI does not support loading calibration data from files (use Python API)
- INT4 quantization requires calibration data (no dynamic INT4)
- No quantization-aware training (QAT) - build-phase only
- No ONNX/TFLite export (use dedicated conversion tools)

### Future Enhancements (v2)

- Genetic optimization for quantization parameters
- Experiment tracking and logging
- Mixed precision (different bits per layer)
- LLM.int8() style outlier detection
- Automatic layer sensitivity analysis

---

## Older Versions

No older versions. v1.0 is the initial release.
