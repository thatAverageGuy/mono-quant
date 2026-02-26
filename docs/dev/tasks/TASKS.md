# Tasks: Mono Quant

**Index only. Full details in each task directory.**

---

## Active

| ID | Summary | Status | Depends On | Detail |
|----|---------|--------|------------|--------|
## Completed (Manual test bug fixes)

| ID | Summary | Completed | Detail |
|----|---------|-----------|--------|
| BF-016 | Fix validate_onnx_model full-level hardcodes float32 dummy | 2026-02-27 | docs/dev/tasks/BF-016/ |
| T-040 | dynamo=True ONNX export path for transformer models | 2026-02-27 | docs/dev/tasks/T-040/ |
| CL-002 | Version bump 1.1.0 → 2.0.0 before release | 2026-02-27 | docs/dev/tasks/CL-002/ |
| BF-015 | Fix nn.Embedding subclass quantization — use exact-type match | 2026-02-27 | docs/dev/tasks/BF-015/ |
| BF-014 | Three ONNX/dynamic-quant bugs from manual test A2 | 2026-02-26 | docs/dev/tasks/BF-014/ |

---

## Pending — Audit Fixes (Priority: Complete before Phase 6)

> All items below were found during a correctness audit on 2026-02-25.
> They are ordered by severity. Critical and High items must be resolved
> before Phase 6 work begins — several cause runtime crashes or silent
> data corruption on every code path.

### Critical Bug Fixes

| ID | Summary | Severity | Status | Depends On | Detail |
|----|---------|----------|--------|------------|--------|
| BF-002 | Fix dual QuantizationInfo — result.save() always crashes with AttributeError | C3 | DONE | — | docs/dev/tasks/BF-002/ |
| BF-003 | Fix INT4 symmetric formula — spurious -8 shift inverts all weights | C2 | DONE | — | docs/dev/tasks/BF-003/ |
| BF-004 | Fix CLI Context.exit() called as class method — TypeError at runtime | C10 | DONE | — | docs/dev/tasks/BF-004/ |
| BF-007 | Fix quantize_weight_int4 fallback — returns wrong zero_points and wrong packing | C7 | DONE | — | docs/dev/tasks/BF-007/ |
| BF-008 | Fix _quantize_int8_model nested layer detection — always False, dead code | C6 | DONE | — | docs/dev/tasks/BF-008/ |
| BF-009 | Fix dequantize_model crash on qint8 buffers — .to() can't cast qint8 | C8 | DONE | — | docs/dev/tasks/BF-009/ |
| T-030 | Correct ONNX export phantom — T-014–T-017 falsely marked DONE, no code exists | C1 | DONE | — | docs/dev/tasks/T-030/ |

### High Priority Bug Fixes

| ID | Summary | Severity | Status | Depends On | Detail |
|----|---------|----------|--------|------------|--------|
| BF-006 | Fix INT4 default skip list injected into all INT8 static_quantize calls | H2 | DONE | — | docs/dev/tasks/BF-006/ |
| BF-010 | Fix quantize_embedding_module dropping dtype parameter | H5 | DONE | — | docs/dev/tasks/BF-010/ |
| BF-011 | Fix _test_load_run mutating the model under test | H6 | DONE | — | docs/dev/tasks/BF-011/ |
| BF-012 | Fix hardcoded weight range threshold 100 — false failures on real models | H4 | DONE | — | docs/dev/tasks/BF-012/ |
| BF-013 | Fix CI silently swallowing test failures (|| echo fallback) | M5 | DONE | — | docs/dev/tasks/BF-013/ |
| T-033 | Fix file path model input in quantize() — always crashes, feature non-functional | H1 | DONE | — | docs/dev/tasks/T-033/ |

### Medium Priority Bug Fixes

| ID | Summary | Severity | Status | Depends On | Detail |
|----|---------|----------|--------|------------|--------|
| BF-005 | Fix mutable default argument skip_set=set() in _quantize_sequential_module | C9 | DONE | — | docs/dev/tasks/BF-005/ |
| CL-001 | Code quality cleanup — version strings, __all__ exports, stale test code, observer docs | M1-M4,M6,M7 | DONE | — | docs/dev/tasks/CL-001/ |

### Calibration & Observer Fixes (Required for correct static quantization)

| ID | Summary | Severity | Status | Depends On | Detail |
|----|---------|----------|--------|------------|--------|
| T-032 | Fix HistogramObserver — incompatible histogram accumulation and wrong zp formula | C5 | DONE | — | docs/dev/tasks/T-032/ |
| T-031 | Implement activation-based calibration in static_quantize (currently dead code) | C4 | DONE | T-032 | docs/dev/tasks/T-031/ |

---

## Completed (v2.0 Phase 5 — ONNX Export)

> T-014–T-017 were phantom entries (see T-030). Actual implementation used IDs T-034–T-037.

| ID | Summary | Status | Depends On | Detail |
|----|---------|--------|------------|--------|
| T-034 | Export infrastructure — BaseExporter, lazy imports, validation, pyproject extras | DONE | T-030 | docs/dev/tasks/T-034/ |
| T-035 | ONNX QDQ node insertion — collect_quantization_params, insert_qdq_nodes | DONE | T-034 | docs/dev/tasks/T-035/ |
| T-036 | ONNXExporter — full 7-step pipeline, dummy input inference, metadata | DONE | T-035 | docs/dev/tasks/T-036/ |
| T-037 | CLI export command + 7 tests (INT8, validate, QDQ, opset, INT4 warn, ImportError) | DONE | T-036 | docs/dev/tasks/T-037/ |

---

## Completed (v2.0 Phase 6 — GPTQ Export)

> T-020 (AWQ) dropped — true AWQ requires Hessian-based calibration, not a format wrapper.

| ID | Summary | Status | Depends On | Detail |
|----|---------|--------|------------|--------|
| T-018 | GPTQ packing math + GPTQExporter (AutoGPTQ V1 format) | DONE | — | docs/dev/tasks/T-018/ |
| T-019 | Public API + CLI export-gptq command + 11 tests | DONE | T-018 | docs/dev/tasks/T-019/ |
| T-021 | Manual vLLM procedure + validate_gptq_checkpoint_structure | DONE | T-019 | docs/dev/tasks/T-021/ |

## Completed (v2.0 Phase 7 — GGUF Export)

| ID | Summary | Status | Depends On | Detail |
|----|---------|--------|------------|--------|
| T-022 | GGUF binary format writer (header + KV metadata) | DONE | T-021 | docs/dev/tasks/T-022/ |
| T-023 | Q4_K_S quantization type support | DONE | T-022 | docs/dev/tasks/T-023/ |
| T-024 | GGUFExporter, arch maps, public API, CLI, tests | DONE | T-022, T-023 | docs/dev/tasks/T-024/ |
| T-025 | gguf-py validation + manual llama.cpp procedure | DONE | T-023, T-024 | docs/dev/tasks/T-025/ |

## Completed (v2.0 Phase 8 — Unified Export API)

| ID | Summary | Status | Depends On | Detail |
|----|---------|--------|------------|--------|
| T-026 | Python export API — orchestrator + result.export() | DONE | T-021, T-025 | docs/dev/tasks/T-026/ |
| T-027 | CLI unified export command (replaces export/export-gptq/export-gguf) | DONE | T-026 | docs/dev/tasks/T-027/ |
| T-028 | Export pre/post validation (ExportWarning, validate_export_pre/post) | DONE | T-026 | docs/dev/tasks/T-028/ |
| T-029 | result.convert(bits) + monoquant convert CLI command | DONE | T-026 | docs/dev/tasks/T-029/ |

## Pending (Future)

| ID | Summary | Status | Depends On | Detail |
|----|---------|--------|------------|--------|
| T-038 | Calibration-based conversion (result.convert with calibration_data) | TODO | T-029 | docs/dev/tasks/T-038/ |

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
| Audit fixes (critical/high) | v1.2 | BF-002 to BF-009, T-030, T-033 | ✅ Done |
| Audit fixes (medium/calibration) | v1.2 | BF-005, BF-013, CL-001 | ✅ Done |
| Calibration & observer fixes | v1.2 | T-031, T-032 | ✅ Done |
| 5 — ONNX Export | v2.0 | T-034 to T-037 (T-014–T-017 were phantom) | ✅ Done |
| 6 — GPTQ/AWQ Export | v2.0 | T-018 to T-021 | ✅ Done |
| 7 — GGUF Binary Export | v2.0 | T-022 to T-025 | ✅ Done |
| 8 — Unified Export API | v2.0 | T-026 to T-029 | ✅ Done |

---

## Audit Severity Reference

| Code | Level | Description |
|------|-------|-------------|
| C1–C10 | Critical | Runtime crash or silent data corruption |
| H1–H6 | High | Feature non-functional or produces wrong results |
| M1–M7 | Medium | Quality/reliability issue, workaround exists |

---

*Status values: `TODO` → `IN_PROGRESS` → `DONE` | `BLOCKED`*
*Updated: 2026-02-26 (Phase 8 complete)*
