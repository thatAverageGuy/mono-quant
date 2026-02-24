# State Machines: Mono Quant

**Last Updated:** 2026-02-24

---

## 1. Project-Level State Machine

Tracks what state a model is in as it moves through the mono-quant pipeline.

```
                    ┌─────────────┐
                    │   FP32      │  ← user's original model
                    │   MODEL     │
                    └──────┬──────┘
                           │ quantize() called
                           │
              ┌────────────▼─────────────┐
              │   QUANTIZING             │
              │   (module replacement,   │
              │    calibration if static)│
              └────────────┬─────────────┘
                           │ success
              ┌────────────▼─────────────┐
         ┌────│   QUANTIZED              │────┐
         │    │   (QuantizationResult)   │    │
         │    └────────────┬─────────────┘    │
         │                 │                  │
    save_model()     export_to_onnx()    validate()
         │           result.export()          │
         ▼                 │                  ▼
  ┌─────────────┐    ┌─────▼─────┐    ┌─────────────┐
  │  SERIALIZED │    │  ONNX /   │    │  VALIDATED  │
  │  (.pt /     │    │  GPTQ /   │    │  (metrics   │
  │  .safeten.) │    │  GGUF     │    │   checked)  │
  └─────────────┘    └───────────┘    └─────────────┘
        │
  load_model()
        │
        ▼
  ┌─────────────┐
  │  DEPLOYED   │  ← pure PyTorch, no mono-quant needed
  │  (FP32 load │
  │   from disk)│
  └─────────────┘
```

**Error Transitions (any state → ERROR):**
- Invalid model input → `QuantizationError`
- Calibration failure → `CalibrationError`
- Export failure → `ONNXValidationError` / `ExportError`
- Validation failure (if strict mode) → raises

---

## 2. Quantization Workflow State Machine

Internal states within a single `quantize()` call.

```
         ┌──────────┐
         │   IDLE   │ ← quantize() called
         └────┬─────┘
              │
              ▼
      ┌───────────────┐
      │  VALIDATING   │  check: model type, bits valid, config sane
      │   INPUT       │
      └───────┬───────┘
         ok / │ \ error
              │  └──────────────→ QuantizationError (raised)
              ▼
      ┌───────────────┐
      │   LOADING     │  _prepare_model()
      │   MODEL       │  handles: nn.Module | state_dict | file path
      └───────┬───────┘
              │
              ▼
         bits == 16?
        /            \
       yes            no
       │              │
       ▼              ▼
  ┌─────────┐   calibration_data?
  │  FP16   │   /              \
  │  CAST   │  yes              no
  └────┬────┘   │               │
       │        ▼               ▼
       │  ┌──────────┐   ┌──────────────┐
       │  │CALIBRATE │   │   DYNAMIC    │
       │  │(static)  │   │  QUANTIZE    │
       │  └────┬─────┘   └──────┬───────┘
       │       │                │
       │       ▼                │
       │  ┌──────────┐          │
       │  │  FREEZE  │          │
       │  │(apply    │          │
       │  │ qparams) │          │
       │  └────┬─────┘          │
       │       │                │
       └───────┴────────────────┘
                     │
                     ▼
              ┌─────────────┐
              │  VALIDATE   │  SQNR, size, load test
              └──────┬──────┘
                     │
                     ▼
              ┌─────────────┐
              │   RETURN    │  QuantizationResult
              └─────────────┘
```

---

## 3. Calibration State Machine

Internal states within `calibration/runner.py`.

```
   ┌──────────┐
   │  SETUP   │  attach observers to target layers
   └────┬─────┘
        │
        ▼
   ┌──────────────────────────────┐
   │  FOR each calibration batch  │◄─────────────────┐
   │  ┌────────────────────────┐  │                  │
   │  │ FORWARD_PASS           │  │                  │
   │  │ (model.eval(),         │  │                  │
   │  │  torch.no_grad())      │  │                  │
   │  └──────────┬─────────────┘  │                  │
   │             │                │                  │
   │             ▼                │                  │
   │  ┌────────────────────────┐  │                  │
   │  │ OBSERVER_UPDATE        │  │                  │
   │  │ min/max / histogram    │  │   more batches   │
   │  │ per-layer stats        │  │                  │
   │  └──────────┬─────────────┘  │                  │
   │             │ done            │                  │
   └─────────────┼────────────────┘                  │
                 │                            ────────┘
                 ▼
   ┌─────────────────────────────┐
   │  COMPUTE_QPARAMS            │
   │  observer → mapper          │
   │  → scale/zero_point per     │
   │    layer (per-channel)      │
   └──────────────┬──────────────┘
                  │
                  ▼
   ┌─────────────────────────────┐
   │  DETACH_OBSERVERS           │
   │  remove hooks from model    │
   └──────────────┬──────────────┘
                  │
                  ▼
   ┌─────────────────────────────┐
   │  RETURN qparams_map         │
   └─────────────────────────────┘
```

---

## 4. ONNX Export State Machine (Phase 5, complete)

```
   export_to_onnx(model, path, info, opset, validate)
            │
            ▼
   ┌─────────────────┐
   │  REVERT_MODULES │  revert_to_standard_modules()
   │  (copy, not     │  QuantizedLinear → nn.Linear (FP32)
   │   in-place)     │
   └────────┬────────┘
            │
            ▼
   ┌─────────────────┐
   │  INSERT_QDQ     │  qdq_inserter.insert_qdq_nodes()
   │  for each layer:│  - DequantizeLinear node
   │    scale tensor │  - scale initializer (FLOAT)
   │    zp tensor    │  - zero_point initializer (INT8/UINT8)
   └────────┬────────┘
            │
            ▼
   ┌─────────────────┐
   │  ADD_METADATA   │  quantization_bits, scheme,
   │                 │  layer_count, mono_quant_version
   └────────┬────────┘
            │
            ▼
   ┌─────────────────┐
   │  ONNX_EXPORT    │  torch.onnx.export()
   │  (opset >= 13)  │
   └────────┬────────┘
            │ fail → ONNXValidationError (enhanced 4-part message)
            │ ok
            ▼
   validate == 'none'?
   /                 \
  yes                no
  │                  │
  ▼            validate == 'load'?
  DONE          /              \
               yes              no (full)
               │                │
               ▼                ▼
     ┌──────────────┐   ┌──────────────────┐
     │ LOAD_TEST    │   │ LOAD_AND_INFER   │
     │ ONNX Runtime │   │ random input +   │
     │ InferenceSession│ │ NaN/Inf check   │
     └──────┬───────┘   └────────┬─────────┘
            │ ok                  │ ok
            ▼                     ▼
           DONE                  DONE
```

---

## 5. Future Export State Machine (Phases 6-8, planned)

Template for GPTQ/AWQ/GGUF exporters (same BaseExporter interface).

```
   BaseExporter.export(model, path, info, **options)
            │
            ▼
   ┌─────────────────┐
   │ VALIDATE_COMPAT │  validate_compatibility(model, info)
   │                 │  check: dtype match, format constraints
   └────────┬────────┘
            │ fail → ExportError
            │ ok
            ▼
   ┌─────────────────┐
   │ BUILD_METADATA  │  build_metadata(model, info)
   │                 │  format-specific metadata struct
   └────────┬────────┘
            │
            ▼
   ┌─────────────────┐
   │ PACK_WEIGHTS    │  format-specific weight packing
   │                 │  (INT4 packing, GGUF tensors, etc.)
   └────────┬────────┘
            │
            ▼
   ┌─────────────────┐
   │  WRITE_FILE     │  stream or in-memory write
   └────────┬────────┘
            │
            ▼
   ┌─────────────────┐
   │  VALIDATE       │  format-specific: load test in
   │                 │  target runtime (vLLM, llama.cpp)
   └─────────────────┘
```

---

## 6. CLI State Machine

```
   monoquant quantize --model X --bits 8 [--dynamic]
            │
            ▼
   ┌─────────────────┐
   │  PARSE_ARGS     │  Click validates types/required
   └────────┬────────┘
            │ invalid → usage error, exit 1
            │
            ▼
   ┌─────────────────┐
   │  LOAD_MODEL     │  from --model path
   └────────┬────────┘
            │ file not found → error, exit 2
            │
            ▼
   ┌─────────────────┐
   │  QUANTIZE       │  calls Python API quantize()
   │  (with tqdm     │  progress visible on terminal
   │   progress bar) │
   └────────┬────────┘
            │ error → exit 3
            │
            ▼
   ┌─────────────────┐
   │  SAVE_OUTPUT    │  auto-name or --output path
   └────────┬────────┘
            │
            ▼
   ┌─────────────────┐
   │  PRINT_SUMMARY  │  compression ratio, SQNR, size
   └────────┬────────┘
            │
            ▼
           EXIT 0
```

---

*State machines document for: Mono Quant*
*Created: 2026-02-24 during docs/dev/ bootstrap*
