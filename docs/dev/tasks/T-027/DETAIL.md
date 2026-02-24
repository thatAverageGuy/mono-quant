# T-027: CLI Unified Export Command

## Status
TODO

## Phase
08-02 — Phase 8: Unified Export API and Format Conversion

## Requirements
- API-02: CLI: monoquant export <format> <input> <output> [options]

## Proposed CLI Interface

```bash
# Format as positional arg
monoquant export onnx model.pt model.onnx --opset 14 --validate load
monoquant export gptq model.pt model_gptq/ --group-size 128
monoquant export awq model.pt model_awq/
monoquant export gguf model.pt model.gguf --architecture llama

# Format auto-detected
monoquant export model.pt model.onnx
monoquant export model.pt model.gguf --architecture llama

# List available formats
monoquant export --list-formats
```

## Format-Specific Options

| Format | Options |
|--------|---------|
| onnx | --opset, --validate [none/load/full], --mode [qdq/fp32] |
| gptq | --group-size [default: 128], --sym/--no-sym |
| awq | --group-size, --version [GEMM/GEMV] |
| gguf | --architecture, --quant-type [q4_k_m/q4_k_s], --validate-runtime |

## Success Criteria
- [ ] monoquant export works for all 4 formats
- [ ] --list-formats shows available formats and their status
- [ ] Format-specific options correctly passed through
- [ ] Progress bar during export for large models
- [ ] Helpful error if format-specific deps not installed

## Dependencies
- T-026 (Python API unified first)

## Implementation Guidance

1. Refactor current `export_cmd` in cli/commands.py to use orchestrator
2. Add format-specific option groups (Click's @click.option with help grouping)
3. Add --list-formats subcommand
4. Wire progress callback from orchestrator to tqdm
