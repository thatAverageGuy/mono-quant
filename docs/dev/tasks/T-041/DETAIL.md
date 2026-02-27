# T-041: Fix QDQ node insertion for dynamo-exported ONNX graphs

## Status
TODO

## Change Level
**LEVEL 2 — LOCAL**
Changes to QDQ inserter + collector. No public interface changes.

---

## Background

The QDQ inserter (`insert_qdq_nodes` in `export/common/qdq_inserter.py`) matches
quantized layer names against ONNX graph initializer names.

For the TorchScript path, initializer names mirror the PyTorch module hierarchy
exactly: `fc1.weight`, `model.decoder.layers.0.self_attn.k_proj.weight`, etc.

For the dynamo path, the situation is more nuanced (investigated 2026-02-27 against
the actual `opt_int8.onnx` from manual test A2):

**ACTUAL ROOT CAUSE** (not naming mangling as originally hypothesised):

| Layer type | ONNX op | ONNX weight name | QDQ status |
|---|---|---|---|
| `fc1`, `fc2` | `Gemm(x, w, b, transB=1)` | `model.decoder.layers.N.fc1.weight` | ✓ Already works |
| `k_proj`, `q_proj`, `v_proj`, `out_proj` | `MatMul(x, w.T) + b` | `val_44`, `val_53`, … | ✗ Broken |

When dynamo lowers `F.linear(x, w, b)` as `MatMul(x, w.T)` instead of
`Gemm(x, w, b, transB=1)`, the transposed weight `w.T` is stored as a new
anonymous constant (`val_N`) and the original parameter name is lost.

**Empirical verification (OPT-125m, opt_int8.onnx)**:
- 256 total initializers; 51 have `.weight` suffix (all named correctly)
- 49 anonymous `val_N` initializers with weight-like shapes (768×768 or 768×50272)
- 48 of them = 12 layers × 4 attention projections (k, q, v, out) — each 768×768
- 1 extra = lm_head weight (768×50272)
- Confirmed: `val_44 == q_proj.weight.T` exactly (numpy allclose, atol=1e-5)
- Already working: 24 Gemm QDQ pairs for fc1/fc2 (QDQ DOES insert for those)

**Why the split?** All of fc1/fc2 go through `Gemm` (weight stored with name, transB=1),
while all attention projections go through `MatMul(x, w.T)` (transposed weight stored
anonymously). This is a difference in how the dynamo/onnxscript lowering handles
`F.linear` calls in different contexts.

---

## Requirements

1. After dynamo export, recover the mapping `{val_N → (param_name, is_transposed)}`
   by matching anonymous initializer values against model parameters.
2. Pass this mapping into `insert_qdq_nodes` so it can locate the right initializer.
3. When inserting QDQ for a transposed weight: use axis=1 (per-channel along the
   output-channel dimension, which moves from dim 0 to dim 1 under transpose).
4. QDQ nodes must be correctly inserted for both paths after the fix.
5. Existing TorchScript QDQ tests must remain passing.
6. The fix must be opt-in — only triggered on the dynamo path.

---

## Investigation Findings (2026-02-27)

### ONNX graph structure (OPT-125m, dynamo=True)
```
Op counts: Reshape:97, Add:85, MatMul:73, Transpose:60,
           LayerNormalization:25, QuantizeLinear:24, DequantizeLinear:24,
           Gemm:24, Mul:12, Softmax:12, Relu:12, Where:12, IsNaN:12, Gather:2
```
- 24 Gemm nodes = fc1 + fc2 for 12 layers → already have QDQ
- 48 MatMul nodes (of 73) = attention projections → no QDQ; 25 = QKV attention ops
- No Transpose node reads an initializer; val_N are pre-transposed constants

### Value matching approach
```python
# For each anonymous initializer in the ONNX graph:
for init in proto.graph.initializer:
    if '.' not in init.name:  # anonymous
        arr = numpy_helper.to_array(init)
        # Compare against all named parameters of same shape
        for param_name, param in model.named_parameters():
            if param.shape == arr.shape:
                if np.allclose(arr, param.detach().cpu().numpy(), atol=1e-5):
                    mapping[init.name] = (param_name, is_transposed=False)
            if param.T.shape == arr.shape:
                if np.allclose(arr, param.T.detach().cpu().numpy(), atol=1e-5):
                    mapping[init.name] = (param_name, is_transposed=True)
```

### QDQ axis adjustment
- Original weight [out, in], per-output-channel axis=0
- Transposed weight [in, out], channels now at axis=1

---

## Implementation Plan

### New function: `_build_dynamo_name_map` in `qdq_inserter.py`

```python
def _build_dynamo_name_map(
    model: nn.Module,
    model_proto: "onnx.ModelProto",
) -> Dict[str, Tuple[str, bool]]:
    """Match anonymous ONNX initializers to named model parameters.

    Returns dict: {onnx_init_name: (param_dotted_name, is_transposed)}
    """
    import numpy as np
    from onnx import numpy_helper

    # Only target anonymous (non-dotted), weight-like (>=2D, large) initializers
    anon_inits = {}
    for init in model_proto.graph.initializer:
        if '.' not in init.name and len(init.dims) >= 2:
            arr = numpy_helper.to_array(init)
            if max(arr.shape) > 64:  # weight-like, not a small constant
                anon_inits[init.name] = arr

    if not anon_inits:
        return {}

    # Build param lookup: shape → [(name, array)]
    from collections import defaultdict
    params_by_shape: Dict[tuple, list] = defaultdict(list)
    for name, param in model.named_parameters():
        arr = param.detach().cpu().float().numpy()
        params_by_shape[arr.shape].append((name, arr))

    name_map: Dict[str, Tuple[str, bool]] = {}
    for anon_name, anon_arr in anon_inits.items():
        shape = anon_arr.shape
        # Direct match
        for param_name, param_arr in params_by_shape.get(shape, []):
            if np.allclose(anon_arr, param_arr, atol=1e-5):
                name_map[anon_name] = (param_name, False)
                break
        else:
            # Transposed match (handles MatMul(x, w.T) lowering)
            t_shape = shape[::-1] if len(shape) == 2 else None
            if t_shape:
                for param_name, param_arr in params_by_shape.get(t_shape, []):
                    if np.allclose(anon_arr, param_arr.T, atol=1e-5):
                        name_map[anon_name] = (param_name, True)
                        break

    return name_map
```

### Update `insert_qdq_nodes` signature

```python
def insert_qdq_nodes(
    model_proto: "onnx.ModelProto",
    qparams: Dict[str, LayerQParams],
    dynamo_name_map: Optional[Dict[str, Tuple[str, bool]]] = None,
) -> "onnx.ModelProto":
```

Inside, after the existing fallback logic, add a second fallback using
`dynamo_name_map`: if `weight_name` not found in `init_names`, check
if any entry in `dynamo_name_map` maps to `{module_name}.weight`. If so,
use that ONNX initializer name and adjust axis by +1 if `is_transposed`.

### Update export pipeline

In `onnx.py` Step 6 (QDQ insertion), when `dynamo=True`:
```python
if dynamo:
    dynamo_name_map = _build_dynamo_name_map(fp32_model, proto)
else:
    dynamo_name_map = None
proto = insert_qdq_nodes(proto, qparams, dynamo_name_map=dynamo_name_map)
```

`fp32_model` is already in scope at this step (it's the reverted model used for export).

---

## Files to Change

| File | What Changes |
|---|---|
| `src/mono_quant/export/common/qdq_inserter.py` | Add `_build_dynamo_name_map`; update `insert_qdq_nodes` signature + logic |
| `src/mono_quant/export/onnx.py` | Call `_build_dynamo_name_map` when `dynamo=True`; pass map to `insert_qdq_nodes` |
| `tests/test_onnx_export.py` | Update `test_export_onnx_dynamo_qdq_nodes_present` to use a nested model (2-layer transformer with attention), not just MLP |
| `docs/dev/tasks/T-041/IMPL_LOG.md` | Create after implementation |

---

## Success Criteria

- [ ] OPT-125m dynamo export produces QDQ nodes for all 72 quantized linear layers
      (fc1×12 + fc2×12 + k/q/v/out_proj×12×4 = 72 total)
- [ ] `test_export_onnx_dynamo_qdq_nodes_present` passes for a nested model
      with attention projections (not just flat MLP)
- [ ] No regressions on TorchScript QDQ tests
- [ ] `_build_dynamo_name_map` handles models with no anonymous initializers
      without error (returns empty dict)

---

## Performance Notes

Value matching is O(A × P × W) where A = anonymous initializers, P = parameters of
same shape, W = weight size in elements. For OPT-125m: 49 anon × 72 quantized ×
768×768 ≈ 2·10⁹ comparisons worst case, but params_by_shape grouping reduces actual
comparisons to ~49 × 4 = 196 (only same-shape candidates). Each comparison: ~590K
float32 elements → ~2.3MB per comparison × 196 = ~450MB of reads. Acceptable for
a one-time export operation.

For very large models (LLaMA-70B, etc.), consider fingerprinting (first-row hash)
to reduce comparison cost before full allclose check.

---

## Dependencies

- T-040 (done — dynamo export path exists)
- BF-017 (done — OPT-125m exports without DynamicCache/encoding errors)

---

## Notes

Post-v2.0. Does not block release. The ONNX files without QDQ on attention projections
are valid FP32 models. The gap is: no INT8 inference benefit for attention projections
on TensorRT/ORT INT8 EP, and fc1/fc2 already get QDQ correctly.
