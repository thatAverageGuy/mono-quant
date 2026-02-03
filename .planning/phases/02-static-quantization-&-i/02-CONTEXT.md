# Phase 2: Static Quantization & I/O - Context

**Gathered:** 2026-02-03
**Status:** Ready for planning

## Phase Boundary

Users can apply static quantization using calibration data and save/load quantized models. This phase adds calibration infrastructure (MinMaxObserver), layer type selection, serialization (PyTorch + Safetensors formats), and validation metrics. The focus is on backend engineering — users call APIs and run commands, but don't see visual interfaces.

## Implementation Decisions

### Calibration data interface
- **Input formats:** Support both tensors (List[torch.Tensor]) and DataLoader for flexibility
- **Sample count:** Claude's discretion — determine best approach for typical ML workflows
- **Forward passes:** Claude's discretion — determine based on calibration accuracy needs
- **Progress reporting:** Auto-show progress for large datasets (auto-detect "large")

### Layer selection approach
- **API design:** Support both include and exclude parameters for flexibility
  - `layer_types=[nn.Linear, nn.Conv2d]` to specify which to quantize
  - `skip_layers=[nn.LayerNorm, nn.Embedding]` to exclude specific types
- **Selection method:** Support both auto-detect by type AND manual name selection
  - Auto-detect from model layers (user provides module types, we find instances)
  - User provides exact layer names (e.g., 'model.encoder.0.weight')
- **Skipped layer reporting:** Claude's discretion — determine best approach

### Safetensors metadata
- **Metadata to store:** All four categories
  - Quantization parameters: dtype (qint8, float16), scheme (symmetric/asymmetric), axis
  - Model architecture: Original model architecture info (layer types, shapes)
  - Version compatibility: Package version, PyTorch version used for quantization
  - Metrics & statistics: Original model size, quantized size, compression ratio, SQNR
- **Structure:** Global + per-layer metadata
  - Single global JSON object in metadata field
  - Per-layer metadata for individual tensor details
- **Write timing:** Auto-save with model (quantized_model.save(path) writes metadata automatically)

### Validation behavior
- **When to run:** Always validate after quantization to ensure model integrity
- **Validation checks:** All four checks
  - SQNR metric — Calculate and report Signal-to-Quantization-Noise Ratio
  - Model size comparison — Compare original vs quantized sizes (MB, compression ratio)
  - Load and run test — Test that model can be loaded from disk and run inference
  - Weight range check — Verify quantized weights are in valid range
- **Failure handling:** Configurable behavior
  - `on_failure='error'` — raise exception (quantization fails)
  - `on_failure='warn'` — warning + return model anyway
  - `on_failure='ignore'` — silent, return model

### Claude's Discretion
- Calibration sample count — determine based on typical ML workflow needs
- Calibration forward passes — single vs multiple based on accuracy requirements
- Skipped layer reporting — how users are informed about skipped layers
- Large dataset threshold — auto-detect "large" for progress reporting

## Specific Ideas

No specific requirements — open to standard approaches for quantization libraries and ML tools.

## Deferred Ideas

None — discussion stayed within phase scope.

---

*Phase: 02-static-quantization-&-i/o*
*Context gathered: 2026-02-03*
