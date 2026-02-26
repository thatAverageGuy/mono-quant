# Architecture: Mono Quant

**Version:** 1.1 (current) / v2.0 in progress
**Last Updated:** 2026-02-26

## System Overview

Mono Quant is a build-phase PyTorch quantization library. Users load models themselves,
pass them in, receive quantized models back. No runtime dependency — quantized models
deploy without mono-quant installed.

### Core Principle

```
User model (FP32) → mono-quant → Quantized model artifact → deploy anywhere
```

The user owns model loading and serving. mono-quant owns quantization only.

---

## Layer Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      User Interface Layer                        │
│  ┌───────────────┐  ┌───────────────┐  ┌────────────────────┐  │
│  │  Python API   │  │      CLI      │  │  QuantizationConfig│  │
│  │  quantize()   │  │  monoquant/mq │  │  (dataclass)       │  │
│  └───────┬───────┘  └───────┬───────┘  └──────────┬─────────┘  │
└──────────┼──────────────────┼────────────────────── ┼───────────┘
           └──────────────────┼────────────────────────┘
                              │
┌─────────────────────────────▼───────────────────────────────────┐
│                       API Orchestrator                           │
│  api/quantize.py — dispatch, input handling, result packaging    │
│  api/result.py   — QuantizationResult with .save()/.validate()  │
└──────────────────────────────┬──────────────────────────────────┘
                               │
         ┌─────────────────────┼──────────────────┐
         │                     │                  │
┌────────▼──────┐  ┌───────────▼─────┐  ┌────────▼──────────────┐
│ Core Quant.   │  │  Calibration    │  │    I/O Layer           │
│ core/         │  │  calibration/   │  │    io/                 │
│ ├ schemes.py  │  │  ├ runner.py    │  │    ├ formats.py        │
│ ├ mappers.py  │  │  └ data.py      │  │    ├ handlers.py       │
│ ├ quantizers  │  └─────────────────┘  │    └ validation.py     │
│ └ observers   │                       └────────────────────────┘
└────────┬──────┘
         │
┌────────▼──────────────────────────────────────────────────────┐
│                  PyTorch Integration Layer                      │
│  modules/                                                       │
│  ├ linear.py    — QuantizedLinear, QuantizedLinearInt4,        │
│  │                QuantizedConv2d                               │
│  └ embedding.py — QuantizedEmbedding                           │
└────────────────────────────────────────────────────────────────┘
         │ (v2.0 addition)
┌────────▼──────────────────────────────────────────────────────┐
│                      Export Layer (Phase 8)                    │
│  export/                                                        │
│  ├ __init__.py      — re-exports all format functions + orch.  │
│  ├ orchestrator.py  — export_model(), list_formats(),          │
│  │                    _detect_format() — unified dispatch       │
│  ├ base.py          — BaseExporter abstract class              │
│  ├ onnx.py          — ONNXExporter (Phase 5)                   │
│  ├ onnx_impl.py     — thin wrapper                             │
│  ├ gptq.py          — GPTQExporter (Phase 6)                   │
│  ├ gptq_impl.py     — thin wrapper                             │
│  ├ gguf.py          — re-export of gguf/exporter.py            │
│  ├ gguf_impl.py     — thin wrapper                             │
│  ├ gguf/            — GGUFExporter, GGUFWriter, arch_maps      │
│  └ common/                                                      │
│      ├ qdq_inserter.py  — QDQ node insertion utilities         │
│      └ validators.py    — ExportWarning, validate_export_pre,  │
│                           validate_export_post,                 │
│                           validate_onnx_model,                  │
│                           validate_gptq_checkpoint_structure,   │
│                           validate_gguf_checkpoint              │
└────────────────────────────────────────────────────────────────┘
│  (Phase 8 additions to api/result.py)                          │
│  result.export(path, format, **kwargs) — delegates to orch.   │
│  result.convert(bits, **kwargs)        — dynamic re-quantize  │
└────────────────────────────────────────────────────────────────┘
```

---

## Module Responsibilities

| Module | Responsibility | Key Classes/Functions |
|--------|---------------|----------------------|
| `config/` | Quantization parameters | `QuantizationConfig` dataclass |
| `core/schemes.py` | Symmetric/asymmetric math | `SymmetricScheme`, `AsymmetricScheme` |
| `core/mappers.py` | Scale/zero-point calculation | per-tensor, per-channel mappers |
| `core/quantizers.py` | Module replacement orchestration | `dynamic_quantize()`, `static_quantize()` |
| `core/observers.py` | Calibration statistics collection | `MinMaxObserver`, `MovingAverageMinMaxObserver`, `HistogramObserver` |
| `modules/linear.py` | Quantized nn.Linear/Conv2d replacement | `QuantizedLinear`, `QuantizedLinearInt4`, `QuantizedConv2d` |
| `modules/embedding.py` | Quantized nn.Embedding replacement | `QuantizedEmbedding` |
| `calibration/runner.py` | Forward pass orchestration | calibration loop over calibration data |
| `calibration/data.py` | Input normalization | tensor/dataloader normalization |
| `io/handlers.py` | Model input loading | `_prepare_model()` — nn.Module / state_dict / file path |
| `io/formats.py` | Save/load | `save_model()`, `load_model()` |
| `io/validation.py` | Post-quantization metrics | `ValidationResult`, `validate_quantization()`, SQNR |
| `api/quantize.py` | Unified entry point | `quantize()` — dispatches to dynamic or static |
| `api/result.py` | Result packaging | `QuantizationResult` with `.save()`, `.validate()`, `.export()` |
| `api/exceptions.py` | Error hierarchy | `QuantizationError`, `ONNXValidationError` |
| `cli/main.py` | CLI entry point | `cli` group (click), registered as `monoquant` / `mq` |
| `cli/commands.py` | CLI subcommands | `quantize_cmd`, `export_cmd`, `validate_cmd`, `info_cmd` |
| `cli/progress.py` | Progress reporting | `should_show_progress()`, tqdm wrappers |
| `export/base.py` | Export abstraction | `BaseExporter` abstract class |
| `export/onnx.py` | ONNX exporter | `ONNXExporter`, `export_to_onnx_impl()` |
| `export/common/qdq_inserter.py` | QDQ node construction | `insert_qdq_for_linear()`, `insert_qdq_for_conv2d()` |
| `export/common/validators.py` | Export validation | `validate_onnx_load()`, `validate_onnx_inference()` |

---

## Data Flow: Quantization

### Dynamic Quantization (no calibration)

```
User: quantize(model, bits=8, dynamic=True)
  │
  ▼ api/quantize.py
  _prepare_model(model)           # handles nn.Module / state_dict / path
  ↓
  core/quantizers.dynamic_quantize(model, config)
  ├── for each nn.Linear/Conv2d:
  │     QuantizedLinear.from_float(module)     # replace in-place
  │     compute scale/zero_point from weights
  │     store INT8 weights
  └── QuantizationInfo collected
  ↓
  io/validation.validate_quantization(original, quantized)
  ↓
  QuantizationResult(model, info)  ← returned to user
```

### Static Quantization (with calibration)

```
User: quantize(model, bits=8, calibration_data=data)
  │
  ▼ api/quantize.py
  _prepare_model(model)
  ↓
  calibration/runner.run_calibration(model, data, observers)
  ├── attach observers to target layers
  ├── forward pass with each calibration batch
  ├── observers collect min/max or histogram stats
  └── observers → mappers → scale/zero_point per layer
  ↓
  core/quantizers.static_quantize(model, qparams_map)
  ├── replace nn.Linear/Conv2d with QuantizedLinear etc.
  ├── store per-channel INT8 weights
  └── for INT4: pack into QuantizedLinearInt4 with group_size
  ↓
  check_accuracy_warnings(info)   # SQNR thresholds
  ↓
  QuantizationResult(model, info)
```

### Export Flow (v2.0)

```
result.export("model.onnx", format="onnx", opset=14)
  │
  ▼ export/__init__.py
  export_to_onnx(model, path, info, opset, validate)
  ├── revert_to_standard_modules(model, inplace=False)
  │     QuantizedLinear → nn.Linear (dequantized weights)
  ├── export/common/qdq_inserter.insert_qdq_nodes(std_model, qparams)
  │     insert QuantizeLinear / DequantizeLinear per layer
  ├── torch.onnx.export(qdq_model, path, opset_version=opset)
  └── if validate: export/common/validators.validate_onnx_load(path)
```

---

## Module Replacement Pattern

Module replacement is the core mechanism. Original modules are replaced recursively:

```python
# core/quantizers.py (simplified)
for name, module in model.named_children():
    if isinstance(module, nn.Linear):
        setattr(model, name, QuantizedLinear.from_float(module, config))
    else:
        _recurse(module, config)  # depth-first
```

**Replacement map:**

| Original | v1.0 Replacement | Notes |
|----------|-----------------|-------|
| `nn.Linear` | `QuantizedLinear` | INT8, asymmetric or symmetric |
| `nn.Linear` | `QuantizedLinearInt4` | INT4, group-wise packed |
| `nn.Conv2d` | `QuantizedConv2d` | INT8, real quantized storage |
| `nn.Embedding` | `QuantizedEmbedding` | INT8 or FP16 only |

**Reversion (for ONNX export):**

`revert_to_standard_modules()` converts back to standard nn.* with dequantized FP32 weights.
This is the bridge between mono-quant's internal representation and export.

---

## Key Architectural Decisions

| Decision | Rationale | See ADR |
|----------|-----------|---------|
| Model-agnostic design (no HF dependency) | Users have varied model sources | ADR-001 |
| Build-phase only | Quantization is build-time, not runtime | ADR-002 |
| CLI + Python API dual interface | CI/CD + interactive use | ADR-003 |
| Local imports to break circular deps | `core/quantizers` ↔ `modules/linear` circular | ADR-004 |
| QDQ format for ONNX (not direct quantized:: export) | PyTorch quantized:: ops unsupported in ONNX | ADR-005 |
| Optional ONNX deps (`pip install mono-quant[onnx]`) | Keep base package minimal | ADR-006 |
| INT4 → INT8 fallback for ONNX opset < 21 | ONNX INT4 support only in opset 21+ | ADR-007 |

---

## Dependency Graph

```
modules.linear
    ↑ depends on
core.schemes ──→ core.mappers ──→ core.quantizers ──→ modules.linear
                                        │
                              calibration.runner
                                        │
                                   api.quantize
                                        │
                          ┌─────────────┤
                     cli.commands    api.result
                                        │
                                   export.onnx
                                        │
                              export.common.*
```

**No circular imports.** The `core.quantizers` ↔ `modules.linear` circular was broken by
moving the `modules.linear` import to local function scope in `core/quantizers.py`.

---

## Quantization Modes

| Mode | Bits | Calibration | Module | Storage |
|------|------|-------------|--------|---------|
| Dynamic INT8 | 8 | None (from weights) | `QuantizedLinear` | INT8 weights |
| Static INT8 | 8 | Required | `QuantizedLinear` | INT8 weights + per-channel scale |
| Dynamic FP16 | 16 | None | In-place cast | FP16 weights |
| Static INT4 | 4 | Required | `QuantizedLinearInt4` | Packed INT8 (group-wise) |
| INT8 Conv2d | 8 | Dynamic | `QuantizedConv2d` | INT8 weights |
| INT8 Embedding | 8 | Dynamic | `QuantizedEmbedding` | INT8 weights |

---

## Serialization

Two formats supported:

| Format | Extension | Notes |
|--------|-----------|-------|
| Safetensors | `.safetensors` | Default; safe, fast, cross-platform |
| PyTorch | `.pt` / `.pth` | Full pickle format |

Both formats auto-convert to PyTorch-native modules on save (via `convert_to_pytorch_native()`),
meaning loaded models work without mono-quant installed. Quantization metadata is preserved
in the state_dict for inspection.

---

## v2.0 Export Layer Status

| Format | Phase | Requirements | Status |
|--------|-------|-------------|--------|
| ONNX (QDQ) | 5 | ONNX-01 to ONNX-06 | ✅ Complete |
| GPTQ checkpoint | 6 | GPTQ-01 to GPTQ-05 | Not started |
| AWQ checkpoint | 6 | AWQ-01 to AWQ-04 | Not started |
| GGUF binary | 7 | GGUF-01 to GGUF-06 | Not started |
| Unified export API | 8 | API-01 to API-04 | Not started |
| Format conversion | 8 | CONV-01 to CONV-02 | Not started |

---

## Test Structure

```
tests/
├── test_api.py          # End-to-end API tests (quantize, save, load, validate)
└── test_onnx_export.py  # ONNX export tests (23 tests, 483 lines)
```

Tests are integration-first. Unit coverage is achieved via integration paths.
Future tests for Phases 6-8 should follow the same pattern.

---

*Architecture document for: Mono Quant v1.1 / v2.0-in-progress*
*Created: 2026-02-24 during docs/dev/ bootstrap*
