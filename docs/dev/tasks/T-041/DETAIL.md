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
The match works because TorchScript preserves `named_parameters()` naming.

For the dynamo path (`torch.export` → ONNX), initializer naming differs:
- On simple models (Sequential MLP), naming happens to be identical — this is
  what T-040 probing confirmed and why the unit tests pass.
- On complex nested models (OPT-125m, LLaMA, etc.), dynamo uses mangled or
  prefixed names (e.g. `p_model_decoder_layers_0_self_attn_k_proj_weight`
  or similar internal torch.export naming).

Evidence: manual test A2 (OPT-125m, dynamo=True) produced 74 "Could not find
ONNX initializer for '...weight'" warnings — one per quantized linear layer.
The ONNX file is valid and loadable (FP32 weights), but has no QDQ nodes,
meaning no INT8 inference benefit on supporting runtimes.

---

## Requirements

1. Inspect the actual initializer names in a dynamo-exported ONNX graph for a
   known nested model (e.g. the 6-layer transformer in test fixtures).
2. Understand the naming transformation dynamo applies to nested module params.
3. Update `collect_quantization_params` and/or `insert_qdq_nodes` to match
   dynamo initializer names in addition to TorchScript names.
4. QDQ nodes must be correctly inserted for both paths after the fix.
5. Existing TorchScript tests must remain passing.

---

## Investigation Starting Point

After `torch.onnx.export(..., dynamo=True)`, load the proto and inspect:
```python
import onnx
proto = onnx.load("model.onnx")
print([i.name for i in proto.graph.initializer][:20])
```
Compare against `list(model.named_parameters())` to find the naming pattern.

Common dynamo naming patterns observed in PyTorch 2.x:
- Dots replaced with underscores and prefixed: `p_model_decoder_layers_0_...`
- Or stored with the full dotted name but as a graph input rather than initializer
- Requires investigation against the actual exported graph.

---

## Success Criteria

- [ ] OPT-125m dynamo export produces QDQ nodes for all quantized linear layers
- [ ] Unit test: `test_export_onnx_dynamo_qdq_nodes_present` passes for a nested
      model (not just the flat MLP currently used)
- [ ] No regressions on TorchScript QDQ tests

---

## Dependencies

- T-040 (done — dynamo export path exists and exports valid ONNX)
- BF-017 (done — OPT-125m can be exported without DynamicCache/encoding errors)

---

## Notes

This is a quality improvement, not a correctness blocker. The ONNX files
exported without QDQ nodes are valid FP32 ONNX models that load and run correctly.
The missing QDQ nodes mean:
- No INT8 inference benefit on TensorRT/ORT INT8 EP
- File size remains FP32 (~4x larger than it could be with INT8 initializers)

Deferred to post-v2.0. Do not block the release on this.
