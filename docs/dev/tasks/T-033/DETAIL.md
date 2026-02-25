# T-033: Fix file path model input in `quantize()`

## Status
TODO

## Audit Reference
H1 (High)

## Problem
`api/quantize.py`'s `quantize()` accepts a file path string in addition to
an `nn.Module`. The file path branch:

```python
if isinstance(model, (str, Path)):
    state_dict = torch.load(model)
    model = _prepare_model(state_dict)
```

`_prepare_model(state_dict)` requires an `architecture` argument to reconstruct
the model from the state dict. Called with only `state_dict`, it raises
`ValueError: architecture required to reconstruct model from state dict`.

Every file path call to `quantize()` crashes. The feature is fully non-functional.

## Requirements
1. Fix the file path input path so it works end-to-end.
2. Two sub-options for resolution (see Decisions).
3. If the feature is kept, it must work without silent failure.
4. Public API signature of `quantize()` must not break existing callers.

## Decisions
- **Option A — Remove the file path branch.**
  Reason: `quantize()` is fundamentally a model transformation, not a model
  loader. Loading from file conflates concerns. Users can load their own model
  and pass the `nn.Module`. The feature adds complexity for minimal value.
  This is a L1 removal.

- **Option B — Fix the file path branch by requiring `architecture` kwarg.**
  ```python
  def quantize(model, ..., architecture=None):
      if isinstance(model, (str, Path)):
          if architecture is None:
              raise ValueError("architecture kwarg required when model is a file path")
          state_dict = torch.load(model)
          model = _prepare_model(state_dict, architecture=architecture)
  ```
  Reason: Preserves the feature; makes the requirement explicit.

- **Decision:** Prefer **Option A**. The file path branch has never been
  usable (it always raises), so no caller depends on it. Removing it is
  strictly safer than fixing it. Add a clear `TypeError` if a string is
  passed, guiding users to load the model themselves.

## Success Criteria
- [ ] Passing a file path string to `quantize()` raises `TypeError` with a
      helpful message (not crashes with `ValueError` from `_prepare_model`)
- [ ] Passing an `nn.Module` continues to work
- [ ] No test or documented example uses the file path path
- [ ] `_prepare_model` private helper is either removed or kept internal

## Dependencies
- None

## Implementation Guidance

In `src/mono_quant/api/quantize.py`:

```python
# BEFORE (approximate)
def quantize(model, ...):
    if isinstance(model, (str, Path)):
        state_dict = torch.load(model)
        model = _prepare_model(state_dict)   # always crashes
    ...

# AFTER — remove the file path branch entirely
def quantize(model, ...):
    if not isinstance(model, nn.Module):
        raise TypeError(
            f"quantize() expects an nn.Module, got {type(model).__name__}. "
            "To quantize from a file, load the model first: "
            "model = torch.load('path.pt'); quantize(model, ...)"
        )
    ...
```

Also remove `_prepare_model` if it has no other callers (search the codebase).

## Testing Requirements
- Unit: `quantize("path.pt")` raises `TypeError` with helpful message
- Unit: `quantize(nn.Linear(4, 4))` still works (regression check)
- Coverage target: type-check branch

## Open Questions
<!-- MUST be empty before implementation begins -->
