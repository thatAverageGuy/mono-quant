# T-008: Validation Metrics — SQNR, Size, Load Test

## Status
DONE

## Phase
02-04 — Phase 2: Static Quantization & I/O

## Requirements
- VAL-01: Display model size comparison
- VAL-02: Compute SQNR
- VAL-03: Validate quantized model can be loaded and run

## Decisions
- ValidationResult dataclass with sqnr_db, compression_ratio, size_original_mb, size_quantized_mb
- SQNR computed over all quantizable weight tensors
- validate_quantization() runs inference on random input to verify model works

## Files
- `src/mono_quant/io/validation.py` — ValidationResult, validate_quantization()
