# Technical Specification: Mono Quant

**Version:** v2.0 (in progress)
**Last Updated:** 2026-02-24

---

## Core Value

Universal quantization gateway — quantize models with mono-quant, deploy anywhere with
existing tooling. No runtime mono-quant dependency required.

---

## System Constraints

| Constraint | Value | Rationale |
|-----------|-------|-----------|
| Python version | >= 3.11 | Type hints, match statements |
| PyTorch version | >= 2.0 | Modern nn.Module API |
| Required deps | `torch`, `numpy`, `safetensors`, `click`, `tqdm` | Minimal footprint |
| Optional deps | `onnx>=1.16`, `onnxruntime>=1.18` | Install via `[onnx]` extra |
| Use case | Build-phase only | Not for runtime serving |
| Model scope | Any PyTorch nn.Module | Model-agnostic |

---

## v1.0 Requirements (Shipped 2026-02-03)

All 30 requirements satisfied. See `docs/dev/tasks/TASKS.md` for task mapping.

### Model Agnostic

| ID | Requirement | Status |
|----|-------------|--------|
| AGN-01 | Accept any PyTorch `nn.Module` | ✅ Done |
| AGN-02 | Accept PyTorch `state_dict` | ✅ Done |
| AGN-03 | Work with models from any source (HF, local, custom) | ✅ Done |
| AGN-04 | Require only `torch` as core dependency | ✅ Done |

### Core Quantization

| ID | Requirement | Status |
|----|-------------|--------|
| QCORE-01 | INT8 quantization with per-channel scaling | ✅ Done |
| QCORE-02 | INT4 quantization with group-wise scaling | ✅ Done |
| QCORE-03 | FP16 quantization for memory reduction | ✅ Done |
| QCORE-04 | Dynamic quantization (no calibration data) | ✅ Done |
| QCORE-05 | Static quantization with calibration data | ✅ Done |
| QCORE-06 | User selects which layer types to quantize | ✅ Done |
| QCORE-07 | Symmetric and asymmetric quantization schemes | ✅ Done |

### Calibration

| ID | Requirement | Status |
|----|-------------|--------|
| CAL-01 | MinMaxObserver for scale/zero-point computation | ✅ Done |
| CAL-02 | MovingAverageMinMaxObserver | ✅ Done |
| CAL-03 | HistogramObserver with KL divergence | ✅ Done |
| CAL-04 | Layer skipping to protect sensitive modules | ✅ Done |
| CAL-05 | Accept calibration data as tensor list or DataLoader | ✅ Done |

### I/O

| ID | Requirement | Status |
|----|-------------|--------|
| IO-01 | Save to PyTorch format (`.pt` / `.pth`) | ✅ Done |
| IO-02 | Save to Safetensors format (`.safetensors`) | ✅ Done |
| IO-03 | Save quantization config with model | ✅ Done |
| IO-04 | Load quantized model from disk | ✅ Done |
| IO-05 | Dequantize model back to FP32 | ✅ Done |

### Validation

| ID | Requirement | Status |
|----|-------------|--------|
| VAL-01 | Display model size comparison | ✅ Done |
| VAL-02 | Compute SQNR (signal-to-quantization-noise ratio) | ✅ Done |
| VAL-03 | Validate quantized model can be loaded and run | ✅ Done |
| VAL-04 | Warn about potential accuracy issues | ✅ Done |

### User Interfaces

| ID | Requirement | Status |
|----|-------------|--------|
| UI-01 | Python API: `quantize(model, bits=8, dynamic=False)` | ✅ Done |
| UI-02 | CLI: `monoquant quantize --model model.pt --bits 8` | ✅ Done |
| UI-03 | Specify quantization parameters via API and CLI | ✅ Done |
| UI-04 | Progress bar for large model quantization | ✅ Done |

---

## v1.1 Requirements (Shipped 2026-02-04)

Enhancements shipped as patch/minor release on top of v1.0.

| Feature | Status |
|---------|--------|
| `QuantizedConv2d` with real INT8 weight storage | ✅ Done |
| Dynamic quantization exclusion parameters matching static API | ✅ Done |
| `convert_to_pytorch_native()` for zero-dependency deployment | ✅ Done |
| `state_dict` serialization for quantization metadata | ✅ Done |
| `QuantizedEmbedding` support (INT8/FP16 only) | ✅ Done |
| `revert_to_standard_modules()` for ONNX/ecosystem compat | ✅ Done |

---

## v2.0 Requirements (In Progress)

### ONNX Export (Phase 5 — Complete)

| ID | Requirement | Status |
|----|-------------|--------|
| ONNX-01 | Export quantized models to ONNX with QDQ nodes | ✅ Done |
| ONNX-02 | Support opset >= 13 for QDQ compatibility | ✅ Done |
| ONNX-03 | Preserve scale, zero-point in ONNX graph | ✅ Done |
| ONNX-04 | Support INT8 in ONNX format | ✅ Done |
| ONNX-05 | Support INT4 in ONNX format (with fallback to INT8 opset < 21) | ✅ Done |
| ONNX-06 | Validate exported ONNX with ONNX Runtime | ✅ Done |

### GPTQ Export (Phase 6 — Not started)

| ID | Requirement | Status |
|----|-------------|--------|
| GPTQ-01 | Export to GPTQ checkpoint format with packed INT4 weights | ☐ TODO |
| GPTQ-02 | Include per-group scales in GPTQ checkpoint | ☐ TODO |
| GPTQ-03 | Include zero-points in GPTQ checkpoint | ☐ TODO |
| GPTQ-04 | Include `quantization_config.json` for vLLM/SGLang compatibility | ☐ TODO |
| GPTQ-05 | Validate GPTQ exports with vLLM | ☐ TODO |

### AWQ Export (Phase 6 — Not started)

| ID | Requirement | Status |
|----|-------------|--------|
| AWQ-01 | Export to AWQ checkpoint format | ☐ TODO |
| AWQ-02 | Include activation-aware weight information | ☐ TODO |
| AWQ-03 | Include `quantization_config.json` for vLLM/SGLang compatibility | ☐ TODO |
| AWQ-04 | Validate AWQ exports with vLLM / SGLang | ☐ TODO |

### GGUF Export (Phase 7 — Not started)

| ID | Requirement | Status |
|----|-------------|--------|
| GGUF-01 | Export to GGUF binary format (header + metadata KV + tensor data) | ☐ TODO |
| GGUF-02 | Support Q4_K_M quantization type | ☐ TODO |
| GGUF-03 | Support Q4_K_S quantization type | ☐ TODO |
| GGUF-04 | Include required GGUF metadata (arch, quant type, tensor info) | ☐ TODO |
| GGUF-05 | Handle `architecture` parameter for tensor naming conventions | ☐ TODO |
| GGUF-06 | Validate GGUF exports with llama.cpp | ☐ TODO |

### Unified Export API (Phase 8 — Not started)

| ID | Requirement | Status |
|----|-------------|--------|
| API-01 | Python: `QuantizationResult.export(format, path, **kwargs)` | ☐ TODO |
| API-02 | CLI: `monoquant export <format> <input> <output> [options]` | ☐ TODO |
| API-03 | Export validation with runtime compatibility checks | ☐ TODO |
| API-04 | Progress reporting for large model exports | ☐ TODO |
| CONV-01 | Convert between formats without re-quantizing | ☐ TODO |
| CONV-02 | Re-quantization fallback when direct conversion is impossible | ☐ TODO |

---

## v3.0 Requirements (Deferred)

Not in current roadmap. Documented to prevent scope creep.

- Quantization-aware training (QAT) support
- Mixed precision per-layer configuration
- Outlier detection (LLM.int8() style)
- Advanced calibration strategies (GPTQ algorithm)
- Additional GGUF types (Q5_K_M, Q8_0)
- MoE model support for AWQ
- Batch export workflows

---

## Out of Scope (Explicitly Excluded)

| Feature | Reason |
|---------|--------|
| Runtime quantization | Build-phase only |
| Model loading / serving | Users own their loading stack |
| HuggingFace transformers dependency | Model-agnostic — users load models themselves |
| Tokenizers and pipelines | Not a serving tool |
| GUI or web interface | CLI + Python API only |
| Inference runtime | Quantization happens at build time |
| Bi-directional format conversion | Focus: mono-quant → ecosystem formats |
| ONNX Training API extensions | Inference only |

---

## Performance Characteristics

| Metric | v1.0 Verified |
|--------|--------------|
| INT8 compression ratio | ~4x vs FP32 |
| INT4 compression ratio | ~8x vs FP32 (~2x vs INT8) |
| SQNR thresholds | >30 dB: good, 20-30: warning, <10: critical |
| Calibration data | 100-500 samples recommended |
| Layer protection | 512-param threshold (default INT4 skip) |
| Save/load round-trip | Verified: 37.32x compression achieved in testing |

---

## SQNR Accuracy Warnings

```python
# core/quantizers.py
SQNR_THRESHOLDS = {
    'critical': 10.0,   # <10 dB — likely significant accuracy loss
    'warning':  20.0,   # <20 dB — notable accuracy degradation
    'good':     30.0,   # >30 dB — acceptable quantization noise
}
```

---

## Default INT4 Skip List

Sensitive layers automatically excluded from INT4 quantization:

```python
DEFAULT_INT4_SKIP = (
    nn.LayerNorm,
    nn.BatchNorm1d,
    nn.BatchNorm2d,
    nn.Embedding,        # INT8/FP16 via QuantizedEmbedding instead
    nn.MultiheadAttention,
)
# Also: layers with < 512 parameters are skipped automatically
```

---

*Spec document for: Mono Quant*
*Created: 2026-02-24 during docs/dev/ bootstrap*
