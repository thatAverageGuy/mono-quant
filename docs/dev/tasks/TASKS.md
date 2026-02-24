# Tasks: Mono Quant

**Index only. Full details in each task directory.**

---

## Active

*No active tasks. Phase 5 complete. Phase 6 planning not started.*

---

## Pending (v2.0 Phase 6 — GPTQ/AWQ Export)

| ID | Summary | Status | Depends On | Detail |
|----|---------|--------|------------|--------|
| T-018 | 4-bit packing format matching AutoGPTQ | TODO | T-017 | docs/dev/tasks/T-018/ |
| T-019 | GPTQ checkpoint export with quantization_config.json | TODO | T-018 | docs/dev/tasks/T-019/ |
| T-020 | AWQ checkpoint export with activation-aware weights | TODO | T-018 | docs/dev/tasks/T-020/ |
| T-021 | vLLM/SGLang validation and accuracy benchmarking | TODO | T-019, T-020 | docs/dev/tasks/T-021/ |

## Pending (v2.0 Phase 7 — GGUF Export)

| ID | Summary | Status | Depends On | Detail |
|----|---------|--------|------------|--------|
| T-022 | GGUF binary format writer (header + KV metadata) | TODO | T-021 | docs/dev/tasks/T-022/ |
| T-023 | Q4_K_M and Q4_K_S quantization type support | TODO | T-022 | docs/dev/tasks/T-023/ |
| T-024 | Architecture-specific tensor naming conventions | TODO | T-022 | docs/dev/tasks/T-024/ |
| T-025 | llama.cpp validation testing | TODO | T-023, T-024 | docs/dev/tasks/T-025/ |

## Pending (v2.0 Phase 8 — Unified Export API)

| ID | Summary | Status | Depends On | Detail |
|----|---------|--------|------------|--------|
| T-026 | Python export API on QuantizationResult | TODO | T-021, T-025 | docs/dev/tasks/T-026/ |
| T-027 | CLI unified export command | TODO | T-026 | docs/dev/tasks/T-027/ |
| T-028 | Export validation and runtime compatibility checks | TODO | T-026 | docs/dev/tasks/T-028/ |
| T-029 | Format conversion between quantization types | TODO | T-026 | docs/dev/tasks/T-029/ |

---

## Completed

| ID | Summary | Completed | Milestone | Detail |
|----|---------|-----------|-----------|--------|
| T-001 | Project setup, config system, model-agnostic input handling | 2026-02-03 | v1.0 | docs/dev/tasks/T-001/ |
| T-002 | Core quantization math (symmetric/asymmetric schemes, scale/zp mappers) | 2026-02-03 | v1.0 | docs/dev/tasks/T-002/ |
| T-003 | Quantization transformations, QuantizedLinear module | 2026-02-03 | v1.0 | docs/dev/tasks/T-003/ |
| T-004 | dynamic_quantize() function and public API exports | 2026-02-03 | v1.0 | docs/dev/tasks/T-004/ |
| T-005 | Calibration infrastructure (MinMaxObserver, runner, data normalization) | 2026-02-03 | v1.0 | docs/dev/tasks/T-005/ |
| T-006 | Layer selection API and static_quantize() with calibration | 2026-02-03 | v1.0 | docs/dev/tasks/T-006/ |
| T-007 | Serialization (PyTorch and Safetensors formats with metadata) | 2026-02-03 | v1.0 | docs/dev/tasks/T-007/ |
| T-008 | Validation metrics (SQNR, size, load test) and public API integration | 2026-02-03 | v1.0 | docs/dev/tasks/T-008/ |
| T-009 | INT4 quantization with group-wise scaling, QuantizedLinearInt4 | 2026-02-03 | v1.0 | docs/dev/tasks/T-009/ |
| T-010 | Advanced observers (MovingAverageMinMax, Histogram with KL divergence) | 2026-02-03 | v1.0 | docs/dev/tasks/T-010/ |
| T-011 | Layer skipping (default INT4 skip list) and accuracy warnings | 2026-02-03 | v1.0 | docs/dev/tasks/T-011/ |
| T-012 | Python API — unified quantize(), QuantizationResult | 2026-02-03 | v1.0 | docs/dev/tasks/T-012/ |
| T-013 | CLI interface — Click subcommands, progress bars, entry points | 2026-02-03 | v1.0 | docs/dev/tasks/T-013/ |
| BF-001 | v1.1 — QuantizedConv2d INT8, QuantizedEmbedding, PyTorch-native deploy, revert | 2026-02-04 | v1.1 | docs/dev/tasks/BF-001/ |
| T-014 | Export infrastructure — BaseExporter, lazy imports, validation framework | 2026-02-04 | v2.0/P5 | docs/dev/tasks/T-014/ |
| T-015 | ONNX QDQ node insertion utilities | 2026-02-04 | v2.0/P5 | docs/dev/tasks/T-015/ |
| T-016 | Opset version support and quantization parameter preservation | 2026-02-04 | v2.0/P5 | docs/dev/tasks/T-016/ |
| T-017 | CLI export command, error handling, validation testing | 2026-02-04 | v2.0/P5 | docs/dev/tasks/T-017/ |

---

## Blocked

*None.*

---

## Phase → Task Mapping

| Phase | Milestone | Tasks | Status |
|-------|-----------|-------|--------|
| 1 — Core Quantization Foundation | v1.0 | T-001 to T-004 | ✅ Done |
| 2 — Static Quantization & I/O | v1.0 | T-005 to T-008 | ✅ Done |
| 3 — Advanced Calibration & INT4 | v1.0 | T-009 to T-011 | ✅ Done |
| 4 — User Interfaces | v1.0 | T-012 to T-013 | ✅ Done |
| v1.1 bugfixes & features | v1.1 | BF-001 | ✅ Done |
| 5 — ONNX Export | v2.0 | T-014 to T-017 | ✅ Done |
| 6 — GPTQ/AWQ Export | v2.0 | T-018 to T-021 | ☐ TODO |
| 7 — GGUF Binary Export | v2.0 | T-022 to T-025 | ☐ TODO |
| 8 — Unified Export API | v2.0 | T-026 to T-029 | ☐ TODO |

---

*Status values: `TODO` → `IN_PROGRESS` → `DONE` | `BLOCKED`*
*Updated: 2026-02-24*
