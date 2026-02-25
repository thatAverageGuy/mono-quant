# T-030: Correct ONNX export phantom — tasks T-014–T-017 falsely marked DONE

## Status
TODO

## Audit Reference
C1 (Critical — documentation integrity)

## Problem
Tasks T-014 through T-017 (Phase 5: ONNX export) are marked `DONE` in TASKS.md.
The commit history shows `wip: phase 05 paused at task 4/4 (complete)`.

However, `src/mono_quant/export/` does **not exist**. No ONNX export code
exists anywhere in the codebase. `export_to_onnx()` is listed in `__init__.py`
docstring as a public API function but is never defined or imported.

The public `__init__.py` does NOT export `export_to_onnx` (it's in a docstring
comment only), so there is no import error at runtime. But the function is
completely absent.

This is a documentation/state integrity problem: the project believes Phase 5
is done when it isn't.

## Requirements
1. Mark T-014–T-017 as `TODO` (not DONE) in TASKS.md.
2. Update IMPL_LOG.md files for T-014–T-017 to reflect actual state (not
   implemented).
3. Remove the `export_to_onnx()` mention from `__init__.py` docstring OR add
   a clear `NotImplementedError` stub so callers get an actionable error.
4. Plan the actual Phase 5 ONNX implementation as a new set of tasks.

## Decisions
- **Decision:** Add a `NotImplementedError` stub for `export_to_onnx()` in
  `__init__.py` so callers get a clear error rather than `ImportError` or
  `AttributeError`.
  Reason: The function is documented in the public API; removing the mention
  entirely would be a breaking documentation change. A stub is safer.

- **Decision:** Create new sub-tasks T-034 through T-037 to formally plan the
  ONNX implementation (renaming T-014–T-017 is confusing since IMPL_LOGs exist;
  prefer new IDs).
  Reason: T-014–T-017 IMPL_LOGs may have planning content that shouldn't be
  discarded. New IDs keep history clean.

## Success Criteria
- [ ] T-014–T-017 status in TASKS.md is corrected
- [ ] `export_to_onnx()` raises `NotImplementedError` with a helpful message
- [ ] No test or user code silently depends on a non-existent export function
- [ ] New task IDs T-034–T-037 are created for the actual ONNX implementation

## Dependencies
- None (this is a documentation and stub task only)

## Implementation Guidance

### Step 1: Add stub in `src/mono_quant/__init__.py`
```python
def export_to_onnx(model, path, **kwargs):
    raise NotImplementedError(
        "ONNX export is not yet implemented. "
        "It is planned for a future release. "
        "Track progress at: docs/dev/tasks/T-034/"
    )
```

### Step 2: Update TASKS.md
- Change T-014, T-015, T-016, T-017 status from `DONE` to `TODO`
- Add note in each that the original DONE marking was incorrect

### Step 3: Update IMPL_LOG.md for T-014–T-017
- Add an "Audit Note" section to each existing IMPL_LOG.md stating:
  "Marked DONE in original planning but implementation was not completed.
   Status corrected by T-030 on [date]. See T-034+ for actual implementation."

### Step 4: Create T-034–T-037 DETAIL.md files
Mirror the original T-014–T-017 planning content but with updated, accurate
implementation guidance.

## Testing Requirements
- Unit: `import mono_quant; mono_quant.export_to_onnx(model, "x.onnx")`
  raises `NotImplementedError` (not `AttributeError`)
- Coverage target: stub function

## Open Questions
<!-- MUST be empty before implementation begins -->
