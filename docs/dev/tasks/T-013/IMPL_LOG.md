# Implementation Log: T-013

## Summary
CLI interface using Click with git-style subcommands. CLI is a thin wrapper
around Python API — no quantization logic lives in CLI code.

## Status
DONE — 2026-02-03 | Milestone: v1.0 | End of Phase 4.

## Files Changed
| File | Change | What |
|------|--------|------|
| src/mono_quant/cli/main.py | Created | CLI entry point |
| src/mono_quant/cli/commands.py | Created | Subcommand implementations |
| src/mono_quant/cli/progress.py | Created | Progress bar utilities |
| pyproject.toml | Modified | Added monoquant and mq entry points |
