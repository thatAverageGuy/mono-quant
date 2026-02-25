# Implementation Log: BF-004

## Summary
Fixed all `_click.Context.exit(N)` calls in `cli/commands.py`. These were called
as class methods, passing `N` as `self`, causing `TypeError` at runtime on every
CLI error path. Replaced with `raise SystemExit(N)`.

## What Was Done
- Replaced 11 occurrences of `_click.Context.exit(N)` with `raise SystemExit(N)`
  across all exit code values (1, 2, 3, 4, 5) in `commands.py`.
- Added two CLI tests verifying exit codes are returned correctly (not TypeError).

## How It Was Done
`replace_all=True` Edit calls for each exit code value. One additional fix for
the `calibrate_cmd` occurrence which was at 4-space indent (function body level)
rather than 8-space (nested block level).

## Files Changed

| File | Change Type | What Changed |
|------|-------------|--------------|
| src/mono_quant/cli/commands.py | Modified | 11× `_click.Context.exit(N)` → `raise SystemExit(N)` |
| tests/test_bugfixes.py | Modified | CLI exit code tests added |

## Detailed Changes Per File
### src/mono_quant/cli/commands.py (Modified)
Occurrences fixed:
- Line 132: exit(2) — missing calibration for static quantize
- Line 144: exit(2) — CLI calibration loading not implemented
- Line 151: exit(3) — quantization failed + strict mode
- Line 176: exit(3) — warnings + strict mode
- Line 185: exit(2) — MonoQuantError handler
- Line 192: exit(1) — general Exception handler
- Line 247: exit(4) — validate_cmd strict mode
- Line 258: exit(4) — validate_cmd exception handler
- Line 350: exit(5) — info_cmd exception handler
- Line 424: exit(1) — compare_cmd exception handler
- Line 459: exit(1) — calibrate_cmd stub

## Testing Results
- Unit: `test_cli_quantize_no_calibration_exits_code_2` — PASS
- Unit: `test_cli_calibrate_exits_code_1` — PASS
- All 15 tests passing

## Final State
**Status**: DONE | **Date**: 2026-02-25
