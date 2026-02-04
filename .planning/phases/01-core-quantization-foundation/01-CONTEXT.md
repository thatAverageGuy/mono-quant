# Phase 1: Core Quantization Foundation - Context

**Gathered:** 2026-02-03
**Status:** Ready for planning

## Phase Boundary

Users can quantize any PyTorch model to INT8 or FP16 using dynamic quantization. Core quantization engine with model-agnostic input handling, configurable schemes, and torch-only dependency.

## Implementation Decisions

### Model Input Handling
- Accept both nn.Module and state_dict — detect format and handle appropriately
- Lazy validation — try quantizing, fail if unsupported (don't pre-validate all layers)
- Always copy — never modify the original model the user passed in
- Flexible return — return both nn.Module and state_dict, or let user choose format

### Quantization Defaults
- Symmetric vs asymmetric is layer-dependent (e.g., activations symmetric, weights asymmetric)
- Per-channel vs per-tensor is layer-dependent based on what works best for each layer type
- Parameters override config files when both present
- Configuration priority: function parameters > global config file > defaults

### Error Behavior
- Unsupported layers: Partial quantization — quantize what's possible, return list of skipped layers
- Invalid input: Layered approach — attempt recovery first, then descriptive error with suggestions
- Mid-process failure: Clean failure — raise exception, preserve original model state
- Non-critical warnings: Log to console and continue

### Internal Structure
- Highly modular — schemes, mappers, quantizers as separate modules with clear boundaries
- Plugin architecture for extensibility — users can add custom schemes/observers in future phases
- Observers fully separate from quantization logic (even though Phase 1 is dynamic-only)
- Internal API granularity is at thatAverageGuy's discretion

### thatAverageGuy's Discretion
- FP16 quantization approach (simple cast vs full quantization pipeline)
- Internal API granularity — choose what best fits the architecture
- Exact component boundaries within the modular structure
- Observer design patterns (even though not used in Phase 1 dynamic quantization)

## Specific Ideas

No specific requirements — open to standard PyTorch quantization patterns

## Deferred Ideas

None — discussion stayed within phase scope

---

*Phase: 01-core-quantization-foundation*
*Context gathered: 2026-02-03*
