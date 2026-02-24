# T-013: CLI Interface — Click Subcommands, Progress Bars, Entry Points

## Status
DONE

## Phase
04-02 — Phase 4: User Interfaces

## Requirements
- UI-02: CLI: monoquant quantize --model model.pt --bits 8
- UI-03: Specify quantization parameters via CLI
- UI-04: Progress bar for large model quantization

## Decisions
- Click (not typer) for CLI framework — already in deps
- Git-style subcommands: quantize, validate, info, compare, calibrate
- tqdm for progress bars (works in CI/CD environments)
- Two entry points: `monoquant` and `mq` (alias)
- CLI calls Python API — no quantization logic in CLI itself

## CLI Commands Implemented
- `monoquant quantize` — main quantization command
- `monoquant validate` — validate a quantized model
- `monoquant info` — model info and stats
- `monoquant compare` — compare two models
- `monoquant export` — export to formats (added in Phase 5)

## Files
- `src/mono_quant/cli/main.py` — CLI entry point and group
- `src/mono_quant/cli/commands.py` — all subcommand implementations
- `src/mono_quant/cli/progress.py` — progress bar utilities
