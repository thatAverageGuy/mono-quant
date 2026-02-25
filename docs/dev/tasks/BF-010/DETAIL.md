# BF-010: Fix `quantize_embedding_module` dropping the dtype parameter

## Status
TODO

## Audit Reference
H5 (High)

## Problem
In `modules/embedding.py`, `quantize_embedding_module` accepts a `dtype`
parameter but never passes it through:

```python
def quantize_embedding_module(module, dtype=torch.qint8, symmetric=True, ...):
    return QuantizedEmbedding.from_embedding(module, symmetric=symmetric, ...)
    #                                         ↑ dtype is NOT passed
```

`QuantizedEmbedding.from_embedding` (or its internal helper) hardcodes
`quantize_weight_int8` regardless of the `dtype` argument. This means:
- Passing `dtype=torch.float16` silently quantizes to INT8 anyway.
- The `dtype` parameter is a documentation lie — callers believe they control
  the dtype but they don't.

## Requirements
1. `dtype` must be forwarded from `quantize_embedding_module` to
   `QuantizedEmbedding.from_embedding`.
2. `from_embedding` must respect `dtype`: use `quantize_weight_int8` for INT8,
   cast for FP16, or raise for unsupported dtypes.
3. Public interface of `quantize_embedding_module` must not change.

## Decisions
- **Decision:** Thread `dtype` through the call chain.
  Reason: The parameter exists; it should do what it says.
  For FP16 embeddings: a simple `.half()` weight cast is sufficient since
  embeddings don't benefit from INT8 quantization beyond storage savings.

## Success Criteria
- [ ] `quantize_embedding_module(module, dtype=torch.float16)` returns a
      module whose weight is `float16`, not `qint8`
- [ ] `quantize_embedding_module(module, dtype=torch.qint8)` still works
- [ ] No silent dtype mismatch

## Dependencies
- None

## Implementation Guidance

In `src/mono_quant/modules/embedding.py`:

```python
# quantize_embedding_module — pass dtype through
def quantize_embedding_module(module, dtype=torch.qint8, symmetric=True, ...):
    return QuantizedEmbedding.from_embedding(
        module, dtype=dtype, symmetric=symmetric, ...   # add dtype here
    )

# QuantizedEmbedding.from_embedding — handle dtype
@classmethod
def from_embedding(cls, embedding, dtype=torch.qint8, symmetric=True):
    if dtype == torch.float16:
        weight = embedding.weight.data.half()
        # store as float16, no quantization
        ...
    else:
        # existing INT8 path
        quantized_weight, scale, zp = quantize_weight_int8(
            embedding.weight.data, symmetric=symmetric
        )
        ...
```

## Testing Requirements
- Unit: `quantize_embedding_module(emb, dtype=torch.float16)` → weight is fp16
- Unit: `quantize_embedding_module(emb, dtype=torch.qint8)` → weight is qint8
- Coverage target: both dtype branches

## Open Questions
<!-- MUST be empty before implementation begins -->
