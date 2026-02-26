# T-040: dynamo=True ONNX export path

## Status
TODO

## Change Level
**LEVEL 2 — LOCAL**
New code path in 3 files. No public interface removal — dynamo is additive.
Existing TorchScript path remains default for backward compatibility.

---

## Background

`torch.onnx.export` with `dynamo=False` (TorchScript tracing) fails for transformer
models because it can't handle complex forward signatures (multiple inputs, optional
kwargs, custom nn.Embedding subclasses). BF-014 added graceful error handling, but
the export still fails.

`dynamo=True` uses `torch.export` (FX tracing) internally. It handles multi-input
models correctly, does not choke on LongTensor inputs or optional kwargs, and
produces an ONNX graph with identical initializer naming to the TorchScript path —
meaning the existing QDQ inserter works on dynamo-exported graphs without any changes.

Confirmed on PyTorch 2.10.0:
- Simple MLP (float input): works under dynamo
- Embedding + Linear (LongTensor input): works under dynamo
- Initializer names (`fc1.weight`, `fc2.weight`): identical to TorchScript output
- Requires `onnxscript` package (not currently in optional deps)

---

## Requirements

1. `ONNXExporter.export()` accepts `dynamo: bool = False` parameter.
2. When `dynamo=True`, the export uses `torch.onnx.export(..., dynamo=True)`.
3. The existing TorchScript path (`dynamo=False`) is unchanged — backward compatible.
4. QDQ insertion and metadata steps run on both paths identically (graph naming is
   the same — no changes to `insert_qdq_nodes` needed).
5. `_infer_dummy_input` is used on both paths (still needed for dynamo — the model
   still needs a concrete input to trace through).
6. `onnxscript` is added to `pyproject.toml` `[onnx]` optional dependency group.
7. The `dynamo` parameter is threaded through: `result.export()` → `export_model()`
   → `_export_onnx()` → `export_to_onnx_impl()` → `ONNXExporter.export()`.
8. Tests pass for: dynamo MLP export, dynamo Embedding export, existing TorchScript
   tests unaffected.

---

## Decisions

- **Default stays `dynamo=False`** — existing behaviour preserved. Users opt in.
  Reason: dynamo has different failure modes for exotic models; don't surprise users.
- **Dummy input still required** — dynamo still needs a concrete input for tracing
  via `torch.export.export`. The `_infer_dummy_input` heuristic still applies.
- **QDQ inserter unchanged** — confirmed both paths produce identical initializer
  naming. No graph-structure changes needed.
- **`opset_version` param** — with `dynamo=True`, PyTorch ignores `opset_version`
  in `torch.onnx.export` (it uses its own default). Pass it anyway for the
  TorchScript path; document the limitation for dynamo.
- **`dynamic_axes` param** — dynamo path has a different `dynamic_shapes` mechanism.
  For simplicity, skip `dynamic_axes` on the dynamo path. Users with shape
  requirements should use TorchScript path or pass inputs directly.

---

## State Machine

```
export() called
       |
   dynamo param?
   /          \
False          True
(TorchScript)  (FX/dynamo)
   |               |
_infer_dummy   _infer_dummy
   |               |
torch.onnx.    torch.onnx.
export(...)    export(...,
dynamo=False   dynamo=True)
   |               |
   +-------+-------+
           |
     load proto
           |
     insert QDQ     (same inserter, same initializer names)
           |
     attach metadata
           |
     save .onnx
           |
   validate (optional)
```

---

## Implementation Guidance

### Step 1 — `ONNXExporter.export()` (`src/mono_quant/export/onnx.py`)

Add `dynamo: bool = False` to the signature:

```python
def export(
    self,
    model: nn.Module,
    path: Union[str, Path],
    opset: int = 14,
    dummy_input: Optional[torch.Tensor] = None,
    validate: Union[str, ValidationLevel] = "none",
    dynamo: bool = False,
) -> None:
```

In the export step (currently step 5), branch on `dynamo`:

```python
if dynamo:
    # dynamo path: FX tracing, handles complex forward signatures
    try:
        torch.onnx.export(
            fp32_model,
            (dummy_input,),
            str(tmp_path),
            dynamo=True,
        )
    except (RuntimeError, TypeError) as e:
        raise RuntimeError(
            f"ONNX tracing failed (dynamo=True): {e}\n\n"
            "Hint: provide dummy_input matching your model's forward() signature."
        ) from e
else:
    # TorchScript path (existing, unchanged)
    try:
        torch.onnx.export(
            fp32_model,
            (dummy_input,),
            str(tmp_path),
            dynamo=False,
            opset_version=opset,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        )
    except (RuntimeError, TypeError) as e:
        raise RuntimeError(
            f"ONNX tracing failed: {e}\n\n"
            "Hint: this model requires a non-standard input format. "
            "Pass dummy_input explicitly or use dynamo=True for transformer models: "
            "result.export('out.onnx', dynamo=True)"
        ) from e
```

### Step 2 — `export_to_onnx_impl` (`src/mono_quant/export/onnx_impl.py`)

Thread `dynamo` through:

```python
def export_to_onnx_impl(
    model, path, opset=14, dummy_input=None, validate="none", dynamo=False, **kwargs
) -> None:
    exporter = ONNXExporter()
    exporter.export(model=model, path=path, opset=opset,
                    dummy_input=dummy_input, validate=validate, dynamo=dynamo)
```

### Step 3 — `_export_onnx` in orchestrator (`src/mono_quant/export/orchestrator.py`)

Thread `dynamo` through:

```python
def _export_onnx(model, path, **options):
    ...
    export_to_onnx_impl(
        model=model,
        path=path,
        opset=options.get("opset", 14),
        dummy_input=options.get("dummy_input", None),
        validate=options.get("validate", "none"),
        dynamo=options.get("dynamo", False),   # NEW
    )
```

### Step 4 — `pyproject.toml`

Add `onnxscript` to the `[onnx]` extra:

```toml
[project.optional-dependencies]
onnx = ["onnx>=1.14", "onnxruntime>=1.16", "onnxscript>=0.1"]
```

### Step 5 — Update the TorchScript error hint

Update the existing `except (RuntimeError, TypeError)` hint message in the
TorchScript branch to mention `dynamo=True` as the alternative. (Replace the
current message from BF-014 with the updated version from Step 1 above.)

### Step 6 — Tests (`tests/test_onnx_export.py`)

Add three new tests:

1. `test_export_onnx_dynamo_mlp_succeeds` — `dynamo=True` on a simple MLP produces
   a valid ONNX file
2. `test_export_onnx_dynamo_embedding_model_succeeds` — `dynamo=True` on an
   Embedding+Linear model with LongTensor dummy input produces a valid ONNX file
3. `test_export_onnx_dynamo_qdq_nodes_present` — QDQ nodes are present in the
   dynamo-exported graph (verifies QDQ inserter works on dynamo output)

Existing tests must remain passing (TorchScript path unchanged).

---

## Testing Requirements

- Unit: 3 new tests (see Step 6)
- Regression: all existing ONNX tests must still pass
- Manual: run `test_a_onnx_simple.py` with `dynamo=True` added — should still pass
- Coverage target: both branches of the dynamo/TorchScript split

---

## Open Questions

*None.*

---

## Dependencies

- BF-014 (merged — provides the graceful error infrastructure this builds on)
- BF-015 should ideally be done first or in parallel: if `nn.Embedding` subclasses
  are still being quantized, `embed_positions` will still be broken after revert,
  and dynamo will still fail for OPT-125m even with this fix. See gap analysis below.
