# Phase 4: User Interfaces - Context

**Gathered:** 2026-02-03
**Status:** Ready for planning

## Phase Boundary

Simplify quantization workflows through two interfaces — a Python function API for programmatic use, and a command-line tool for automation/CI-CD. This phase wraps existing quantization functionality (from Phases 1-3) in convenient interfaces, not new quantization capabilities.

**In scope:**
- Python `quantize()` function that unifies `dynamic_quantize()` and `static_quantize()` behind simple API
- CLI `monoquant` command with subcommands (quantize, validate, info, compare, calibrate)
- Progress bars for long-running operations in both CLI and Python API
- Error handling with actionable messages and configurable behavior

**Out of scope:**
- New quantization algorithms or observers (those are in Phase 1-3)
- Model training or fine-tuning capabilities
- GUI or web interface (CLI + Python API only)

## Implementation Decisions

### Python API Design
- **Unified function approach**: Single `quantize()` function that dispatches to `dynamic_quantize()` or `static_quantize()` based on parameters (not split functions or class-based)
- **Parameter handling**: Simplified surface with common options exposed (bits, scheme, dynamic), advanced options via `**kwargs` or config dict (not full exposure or config object)
- **Return value**: `QuantizationResult` object with `.model`, `.info` attributes and convenience methods (`.save()`, `.validate()`) — not just tuple or model-only
- **Input flexibility**: Accept all formats — nn.Module, state_dict, or file path (API handles loading/inspection internally)

### CLI Command Structure
- **Subcommand pattern**: `monoquant quantize --model model.pt --bits 8` (git-style with subcommands, not flag-based or direct commands)
- **Flag naming**: Both short and long flags available (`-b/--bits`, `-m/--model`, `-s/--scheme`) — not short-only or long-only
- **Output naming**: Default auto-naming scheme (`model_quantized.pt`) with `--output` flag to explicitly name (not explicit-only or in-place)
- **Command set**: Full toolkit — `quantize`, `validate`, `info`, `compare`, `calibrate` commands (not just quantize or minimal core commands)

### Progress Reporting
- **Progress library**: You decide (tqdm, rich, or custom minimal)
- **When to show**: Always show progress bar for calibration and quantization (no threshold or time-based hiding)
- **What to display**: You decide (minimal, standard, or rich info)
- **Python API progress**: You decide (silent, always show, or callback pattern)

### Error Handling
- **Python API errors**: Hybrid approach — exceptions for direct use, but Result object also has `.success` flag and `.errors` list
- **CLI error behavior**: Configurable via `--strict` flag (default: warn on recoverable, `--strict`: exit immediately)
- **Validation timing**: Two-stage validation — quick upfront validation of obvious issues, deep validation during quantization
- **Error messages**: Both technical error + actionable suggestion (e.g., "group_size=128 too large for layer with 64 channels. Use --group-size 64 or smaller.")

### Claude's Discretion
**Python API:**
- Exact function signature and parameter names
- Whether to use `*args`/`**kwargs` or explicit parameters
- Result object implementation details and method signatures
- Progress bar library choice (tqdm, rich, or custom)
- Python API progress display approach (silent, bar, or callback)

**CLI:**
- Exact command names and hierarchy (monoquant vs. mq)
- Flag ordering and grouping in help text
- Output format for info/compare commands
- Progress bar styling and position
- Help text examples and usage patterns

**Error Handling:**
- Specific exception types and hierarchy
- Error code values for CLI exit
- Validation rules and their timing
- Message formatting and wording

## Specific Ideas

- CLI should feel familiar to users of git, pytest, or other dev tools
- Python API should be simple: `from mono_quant import quantize; result = quantize(model, bits=8, dynamic=False)`
- Error messages should guide users toward fixes, not just report problems
- Progress bars should work in CI/CD environments (no fancy Unicode if it breaks terminals)

## Deferred Ideas

None — discussion stayed within phase scope.

---

*Phase: 04-user-interfaces*
*Context gathered: 2026-02-03*
