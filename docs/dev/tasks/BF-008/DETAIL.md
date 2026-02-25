# BF-008: Fix `_quantize_int8_model` nested layer detection always False

## Status
TODO

## Audit Reference
C6 (Critical)

## Problem
In `core/quantizers.py`, `_quantize_int8_model` uses this check to detect
nested quantizable layers:

```python
if not isinstance(module, type(model_copy.get_submodule(name))):
    # recurse into nested module
```

`model_copy.get_submodule(name)` returns the same `module` object (it retrieves
the module by the same path name). So `isinstance(module, type(module))` is
always `True`, making `not isinstance(...)` always `False`. The recursion branch
is dead code — nested modules are **never** traversed.

This means models with nested quantizable layers (e.g. transformer blocks that
contain `nn.Linear` layers two levels deep) silently skip those inner layers.
Only the top-level modules directly accessible via `named_children()` get
quantized.

## Requirements
1. Replace the broken check with a correct recursive traversal strategy.
2. All quantizable leaf modules at any nesting depth must be reached.
3. Non-quantizable container modules (Sequential, ModuleList) must be
   recursed into without being quantized themselves.

## Decisions
- **Decision:** Replace the broken condition with a recursive `named_modules()`
  walk instead of `named_children()`.
  Reason: `named_modules()` yields ALL descendant modules recursively, not
  just direct children. Combined with a check against the target types
  (`nn.Linear`, `nn.Conv2d`, etc.), this correctly handles any nesting depth.
  The broken `get_submodule` check can be removed entirely.

## Success Criteria
- [ ] `nn.Linear` nested two or more levels deep is quantized
- [ ] Container modules (Sequential, ModuleList, custom nn.Module wrappers)
      are recursed into correctly
- [ ] The broken `isinstance(module, type(model_copy.get_submodule(name)))`
      check is removed
- [ ] Existing tests pass

## Dependencies
- None

## Implementation Guidance

In `src/mono_quant/core/quantizers.py`, in `_quantize_int8_model`:

```python
# BEFORE (approximate)
for name, module in model_copy.named_children():
    if not isinstance(module, type(model_copy.get_submodule(name))):
        _quantize_int8_model(module, ...)   # never reached
    elif isinstance(module, (nn.Linear, nn.Conv2d, ...)):
        # quantize directly

# AFTER — walk ALL descendants at once
for name, module in model_copy.named_modules():
    if isinstance(module, (nn.Linear, nn.Conv2d, nn.Embedding)):
        if _should_skip(name, module, skip_layers):
            continue
        # quantize module in place using setattr on the parent
        parent_name, _, child_name = name.rpartition(".")
        parent = model_copy.get_submodule(parent_name) if parent_name else model_copy
        setattr(parent, child_name, _quantize_one_module(module, config))
```

Adjust to match the actual variable names found in the file.

## Testing Requirements
- Unit: Build a 3-level nested model (`Outer(Middle(nn.Linear))`); call
  `static_quantize`; verify the inner `nn.Linear` is quantized.
- Unit: Verify container module itself is not replaced with a QuantizedLinear.
- Coverage target: nested traversal path

## Open Questions
<!-- MUST be empty before implementation begins -->
