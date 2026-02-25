# BF-004: Fix CLI Context.exit() called as class method — crashes at runtime

## Status
TODO

## Audit Reference
C10 (High)

## Problem
In `cli/commands.py`, `_click.Context.exit(code)` is called as a class method in
multiple places (lines 132, 144, 151, 176, 185, etc.):

```python
_click.Context.exit(2)
```

`Context.exit()` is an **instance method**, not a class method. When called on the class,
`2` is passed as `self`. This raises `TypeError` at runtime whenever these code paths
are hit (e.g., missing calibration data, validation failures, strict mode).

## Requirements
1. Replace all `_click.Context.exit(N)` with the correct pattern.
2. Error exit codes must be preserved exactly as-is.
3. No change to CLI behavior, only fix the crash.

## Decisions
- **Decision:** Replace with `raise SystemExit(N)`.
  Reason: Cleanest pattern inside a Click command body. `sys.exit(N)` also works but
  requires an import. `ctx.exit(N)` works but requires `@click.pass_context` decorator,
  which changes the function signature and requires more refactoring.
  Alternatives: `raise click.exceptions.Exit(N)` — also valid Click idiom.

## Success Criteria
- [ ] `monoquant quantize --model m.pt --static` (no calibration) exits with code 2
  without TypeError
- [ ] `monoquant quantize --model m.pt --dynamic --strict` with warnings exits with code 3
- [ ] All `_click.Context.exit` occurrences replaced
- [ ] Existing tests still pass

## Dependencies
- None

## Implementation Guidance

In `src/mono_quant/cli/commands.py`, find all occurrences of `_click.Context.exit(N)`
and replace with `raise SystemExit(N)`:

```python
# BEFORE
_click.Context.exit(2)

# AFTER
raise SystemExit(2)
```

Occurrences to fix (search for `Context.exit`):
- `quantize_cmd`: lines ~132, ~144, ~151, ~176, ~185
- `validate_cmd`: lines ~247
- `info_cmd`: line ~350
- `calibrate_cmd`: line ~459

## Testing Requirements
- Unit: Mock the model file path; call `quantize_cmd` without `--calibration`;
  verify `SystemExit(2)` is raised (not TypeError)
- Unit: Call `calibrate_cmd`; verify `SystemExit(1)` raised
- Coverage target: All exit code paths

## Open Questions
<!-- MUST be empty before implementation begins -->
