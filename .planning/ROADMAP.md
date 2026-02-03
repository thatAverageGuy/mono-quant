# Roadmap: Mono Quant

## Overview

Mono Quant delivers simple, reliable model quantization with minimal dependencies. The roadmap follows a dependency-driven approach: establish model-agnostic foundation, add static quantization with robust serialization, introduce advanced calibration and INT4 support, then polish user interfaces. Each phase delivers a verifiable capability that unblocks the next phase.

## Phases

**Phase Numbering:**
- Integer phases (1, 2, 3, 4): Planned milestone work
- Decimal phases (2.1, 2.2): Urgent insertions (marked with INSERTED)

Decimal phases appear between their surrounding integers in numeric order.

- [ ] **Phase 1: Core Quantization Foundation** - Model-agnostic quantization with INT8/FP16 support
- [ ] **Phase 2: Static Quantization & I/O** - Calibration, serialization, and validation
- [ ] **Phase 3: Advanced Calibration & INT4** - Group-wise scaling, advanced observers, INT4 support
- [ ] **Phase 4: User Interfaces** - Python API and CLI for automation and CI/CD

## Phase Details

### Phase 1: Core Quantization Foundation

**Goal**: Users can quantize any PyTorch model to INT8 or FP16 using dynamic quantization

**Depends on**: Nothing (first phase)

**Requirements**: AGN-01, AGN-02, AGN-03, AGN-04, QCORE-01, QCORE-03, QCORE-04, QCORE-07

**Success Criteria** (what must be TRUE):
1. User can pass any PyTorch nn.Module or state_dict and receive a quantized version
2. User can quantize a model to INT8 with per-channel scaling using symmetric or asymmetric schemes
3. User can quantize a model to FP16 for memory reduction
4. User can apply dynamic quantization without providing calibration data
5. Package has only torch as a required dependency (all other deps are optional)

**Plans**: 4 plans

Plans:
- [ ] 01-01-PLAN.md — Project setup, config system, and model-agnostic input handling
- [ ] 01-02-PLAN.md — Core quantization math (symmetric/asymmetric schemes, scale/zp mappers)
- [ ] 01-03-PLAN.md — Quantization transformations (INT8, FP16) and QuantizedLinear module
- [ ] 01-04-PLAN.md — End-to-end dynamic_quantize() function with public API exports

### Phase 2: Static Quantization & I/O

**Goal**: Users can apply static quantization with calibration and save/load quantized models

**Depends on**: Phase 1

**Requirements**: CAL-01, CAL-05, IO-01, IO-02, IO-03, IO-04, IO-05, QCORE-05, QCORE-06, VAL-01, VAL-02, VAL-03

**Success Criteria** (what must be TRUE):
1. User can apply static quantization using calibration data provided as tensors or dataloader
2. User can select which layer types to quantize (Linear, Conv2d, LSTM)
3. User can save quantized model to PyTorch format (.pt/.pth) and Safetensors format
4. User can load a quantized model from disk and run inference
5. User can see model size comparison and SQNR metrics after quantization
6. System validates quantized model can be loaded and run before returning

**Plans**: 4 plans

Plans:
- [ ] 02-01: Calibration infrastructure (MinMaxObserver, calibration data API)
- [ ] 02-02: Static quantization with layer type selection
- [ ] 02-03: Serialization (PyTorch and Safetensors formats with metadata)
- [ ] 02-04: Validation metrics and model verification

### Phase 3: Advanced Calibration & INT4

**Goal**: Users can apply INT4 quantization with advanced observers and layer skipping

**Depends on**: Phase 2

**Requirements**: CAL-02, CAL-03, CAL-04, QCORE-02, VAL-04

**Success Criteria** (what must be TRUE):
1. User can quantize a model to INT4 with group-wise scaling
2. User can choose MovingAverageMinMaxObserver or Histogram observer for robust calibration
3. User can skip sensitive layers during quantization (e.g., embeddings, layer norm)
4. System warns about potential accuracy issues when aggressive quantization is applied

**Plans**: 3 plans

Plans:
- [ ] 03-01: INT4 quantization with group-wise scaling
- [ ] 03-02: Advanced observers (MovingAverageMinMax, Histogram)
- [ ] 03-03: Layer skipping and accuracy warnings

### Phase 4: User Interfaces

**Goal**: Users can quantize models via simple Python API or CLI for CI/CD integration

**Depends on**: Phase 3

**Requirements**: UI-01, UI-02, UI-03, UI-04

**Success Criteria** (what must be TRUE):
1. User can quantize via Python API: `quantize(model, bits=8, dynamic=False)`
2. User can quantize via CLI: `monoquant quantize --model model.pt --bits 8`
3. User can specify quantization parameters (bits, scheme, observer types) via API and CLI
4. CLI shows progress bar for large model quantization

**Plans**: 2 plans

Plans:
- [ ] 04-01: Python API (quantize function, parameter handling)
- [ ] 04-02: CLI interface (command parsing, progress reporting)

## Progress

**Execution Order:**
Phases execute in numeric order: 1 -> 2 -> 3 -> 4

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Core Quantization Foundation | 2/4 | In progress | 2026-02-03 |
| 2. Static Quantization & I/O | 0/4 | Not started | - |
| 3. Advanced Calibration & INT4 | 0/3 | Not started | - |
| 4. User Interfaces | 0/2 | Not started | - |
