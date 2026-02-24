# Developer Documentation Index

Internal development documentation for mono-quant.

## Start Here

- **[ARCHITECTURE.md](ARCHITECTURE.md)** — System architecture, layer diagram, module map, data flows
- **[STATE_MACHINES.md](STATE_MACHINES.md)** — State diagrams for quantization, export, calibration, CLI
- **[SPEC.md](SPEC.md)** — All requirements (v1.0 done, v2.0 in progress, v3.0 deferred)
- **[CONTRIBUTING.md](CONTRIBUTING.md)** — Branch model, commit format, code conventions, testing rules

## Task Tracking

- **[tasks/TASKS.md](tasks/TASKS.md)** — Task index (completed + pending)

### Pending Work (v2.0)

| Phase | Tasks | Status |
|-------|-------|--------|
| Phase 6: GPTQ/AWQ Export | T-018 to T-021 | TODO |
| Phase 7: GGUF Binary Export | T-022 to T-025 | TODO |
| Phase 8: Unified Export API | T-026 to T-029 | TODO |

## Architecture Decisions

| ADR | Decision |
|-----|---------|
| [ADR-001](adr/ADR-001.md) | Model-agnostic design, no HuggingFace dependency |
| [ADR-002](adr/ADR-002.md) | Build-phase only, no runtime quantization |
| [ADR-003](adr/ADR-003.md) | Dual interface: CLI + Python API |
| [ADR-004](adr/ADR-004.md) | Local imports to break circular dependency |
| [ADR-005](adr/ADR-005.md) | QDQ format for ONNX export |
| [ADR-006](adr/ADR-006.md) | Optional ONNX deps via extras group |
| [ADR-007](adr/ADR-007.md) | INT4 fallback to INT8 for opset < 21 |

## Session Context

- Current: see [CONTEXT.md](../../CONTEXT.md) at project root
- Archive: [session_context/](session_context/)
