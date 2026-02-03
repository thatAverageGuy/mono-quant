# Requirements: Mono Quant

**Defined:** 2026-02-03
**Core Value:** Simple, reliable model quantization with minimal dependencies - just torch, no bloat

## v1 Requirements

Requirements for initial release. Each maps to roadmap phases.

### Quantization Core

- [ ] **QCORE-01**: User can quantize model to INT8 with per-channel scaling
- [ ] **QCORE-02**: User can quantize model to INT4 with group-wise scaling
- [ ] **QCORE-03**: User can quantize model to FP16
- [ ] **QCORE-04**: User can apply dynamic quantization (no calibration data)
- [ ] **QCORE-05**: User can apply static quantization (with calibration data)
- [ ] **QCORE-06**: User can select which layer types to quantize (Linear, Conv2d, LSTM)
- [ ] **QCORE-07**: User can choose symmetric or asymmetric quantization scheme

### Calibration

- [ ] **CAL-01**: System computes scale and zero-point using MinMaxObserver
- [ ] **CAL-02**: System supports MovingAverageMinMaxObserver for robustness
- [ ] **CAL-03**: System supports Histogram observer for outlier-aware calibration
- [ ] **CAL-04**: System provides layer skipping to skip sensitive modules
- [ ] **CAL-05**: System accepts calibration data as list of tensors or dataloader

### User Interface

- [ ] **UI-01**: User can quantize via Python API: `quantize(model, bits=8, dynamic=False)`
- [ ] **UI-02**: User can quantize via CLI: `monoquant quantize --model model.pt --bits 8`
- [ ] **UI-03**: User can specify quantization parameters (bits, scheme, observer types)
- [ ] **UI-04**: CLI shows progress for large model quantization

### Serialization

- [ ] **IO-01**: User can save quantized model to PyTorch format (.pt/.pth)
- [ ] **IO-02**: User can save quantized model to Safetensors format
- [ ] **IO-03**: System saves quantization config with model (bits, scheme, scale/zp)
- [ ] **IO-04**: User can load quantized model from disk
- [ ] **IO-05**: User can dequantize model back to FP32

### Validation

- [ ] **VAL-01**: System displays model size comparison (before/after quantization)
- [ ] **VAL-02**: System computes SQNR (signal-to-quantization-noise ratio)
- [ ] **VAL-03**: System validates quantized model can be loaded and run
- [ ] **VAL-04**: System warns about potential accuracy issues (e.g., all layers quantized)

### Model Agnostic

- [ ] **AGN-01**: System accepts any PyTorch nn.Module
- [ ] **AGN-02**: System accepts PyTorch state_dict
- [ ] **AGN-03**: System works with models from any source (HF, local, custom)
- [ ] **AGN-04**: System requires only torch as dependency

## v2 Requirements

Deferred to future release. Tracked but not in current roadmap.

### Genetic Optimization

- **GENO-01**: System evolves quantization parameters to maximize accuracy/speed tradeoff
- **GENO-02**: System supports architecture search (prune + quantize)
- **GENO-03**: System supports benchmark optimization (A/B test quantization methods)

### Experiment Tracking

- **EXP-01**: System logs quantization runs with parameters and results
- **EXP-02**: System compares multiple quantization configurations
- **EXP-03**: System reproduces quantization results from logged config

### Advanced Features

- **ADV-01**: System supports mixed precision (different bits per layer)
- **ADV-02**: System implements LLM.int8() style outlier detection
- **ADV-03**: System performs automatic layer sensitivity analysis
- **ADV-04**: System supports module fusion (Conv-BN-ReLU)

## Out of Scope

Explicitly excluded. Documented to prevent scope creep.

| Feature | Reason |
|---------|--------|
| Quantization-Aware Training (QAT) | Build-phase only tool; QAT requires full training pipeline |
| HuggingFace Transformers integration | Violates model-agnostic principle; adds heavy dependency |
| Runtime inference engine | Quantization is build-time; inference serving is separate problem |
| Automatic model loading from Hub | User loads model however they want; we quantize what's given |
| Custom CUDA kernels | Breaks "minimal dependencies" principle; use PyTorch native ops |
| Model architecture modification | Weights-only approach; assume user knows their model |
| Fancy UI/Dashboard | Build-phase tool doesn't need GUI; CLI + API sufficient |
| Non-PyTorch framework support | Focus on PyTorch; point to framework-specific tools for others |
| ONNX/TFLite export | Adds heavy dependencies; use dedicated conversion tools |
| Model zoo integration | Requires storage/bandwidth/maintenance; not core value |

## Traceability

Which phases cover which requirements. Updated during roadmap creation.

| Requirement | Phase | Status |
|-------------|-------|--------|
| AGN-01 | Phase 1 | Pending |
| AGN-02 | Phase 1 | Pending |
| AGN-03 | Phase 1 | Pending |
| AGN-04 | Phase 1 | Pending |
| QCORE-01 | Phase 1 | Pending |
| QCORE-03 | Phase 1 | Pending |
| QCORE-04 | Phase 1 | Pending |
| QCORE-07 | Phase 1 | Pending |
| CAL-01 | Phase 2 | Pending |
| CAL-05 | Phase 2 | Pending |
| IO-01 | Phase 2 | Pending |
| IO-02 | Phase 2 | Pending |
| IO-03 | Phase 2 | Pending |
| IO-04 | Phase 2 | Pending |
| IO-05 | Phase 2 | Pending |
| QCORE-05 | Phase 2 | Pending |
| QCORE-06 | Phase 2 | Pending |
| VAL-01 | Phase 2 | Pending |
| VAL-02 | Phase 2 | Pending |
| VAL-03 | Phase 2 | Pending |
| CAL-02 | Phase 3 | Pending |
| CAL-03 | Phase 3 | Pending |
| CAL-04 | Phase 3 | Pending |
| QCORE-02 | Phase 3 | Pending |
| VAL-04 | Phase 3 | Pending |
| UI-01 | Phase 4 | Pending |
| UI-02 | Phase 4 | Pending |
| UI-03 | Phase 4 | Pending |
| UI-04 | Phase 4 | Pending |

**Coverage:**
- v1 requirements: 30 total
- Mapped to phases: 30
- Unmapped: 0

**By Phase:**
- Phase 1: 8 requirements (model-agnostic foundation, INT8/FP16, dynamic quantization)
- Phase 2: 12 requirements (calibration, serialization, static quantization, validation)
- Phase 3: 5 requirements (INT4, advanced observers, layer skipping, warnings)
- Phase 4: 4 requirements (Python API, CLI, parameters, progress)

---
*Requirements defined: 2026-02-03*
*Last updated: 2026-02-03 after roadmap creation*
