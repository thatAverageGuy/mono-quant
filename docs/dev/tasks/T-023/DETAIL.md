# T-023: Q4_K_S Quantization Type Support in GGUF

## Status
TODO

## Phase
07-02 — Phase 7: GGUF Binary Format Export

## Requirements
- Implement Q4_K_S quantization: all layers use Q4_K blocks (no Q6_K mixed layers)
- Produce binary block data compatible with llama.cpp's dequantizer
- Re-quantize from FP32 (same approach as GPTQExporter: revert → FP32 → re-quantize)

## Decisions
- **Q4_K_S only** — Q4_K_M (which mixes Q6_K for attention layers) is deferred; Q4_K_S
  covers the common use case and is what most users mean by "4-bit GGUF"
- **Re-quantize from FP32** — mono-quant's internal INT4 format (group_size=128, GPTQ-style
  packing) is incompatible with Q4_K's block structure; always revert to FP32 first
- **Simple min-max quantizer** — matches llama.cpp's dequantization formula; not the
  iterative optimizer used in llama.cpp's encoder (acceptable quality loss)
- **Authoritative reference**: `ggml/src/ggml-quants.c` — `quantize_row_q4_K_ref()`
  and `dequantize_row_q4_K()`. Dequantization formula is what matters for correctness.

## Q4_K Block Structure

**GGML type code**: `GGML_TYPE_Q4_K = 12`
**Block covers**: 256 weights (QK_K = 256)
**Block size**: 144 bytes

```
Offset  Size   Field
──────  ─────  ───────────────────────────────────────────────
0       2      d      (FP16) — super-block scale for quantized scales
2       2      dmin   (FP16) — super-block scale for quantized mins
4       12     scales — 8 sub-block scales + 8 sub-block mins, 6 bits each (packed)
16      128    qs     — 256 × 4-bit nibbles (2 per byte, lo-nibble first)
```

Total: 2 + 2 + 12 + 128 = **144 bytes per 256 weights**.

For a weight matrix of shape `(out_features, in_features)`:
- in_features must be divisible by 256
- Number of blocks = `out_features × (in_features // 256)`
- Total bytes = `out_features × (in_features // 256) × 144`

## 6-Bit Scale Packing (scales[12])

Each super-block of 256 weights has 8 sub-blocks of 32 weights.
Each sub-block i has a 6-bit quantized scale `ls[i]` and 6-bit quantized min `lm[i]`.

Packing 16 × 6-bit values into 12 bytes:

```python
# ls[0..7]: sub-block scales as 6-bit ints (0..63)
# lm[0..7]: sub-block mins  as 6-bit ints (0..63)
scales = bytearray(12)
for j in range(4):
    scales[j]   = ls[j] & 0x3F          # low 6 bits of ls[j] (already ≤63)
    scales[j+4] = lm[j] & 0x3F          # low 6 bits of lm[j]
for j in range(4, 8):
    scales[j+4] = (ls[j] & 0xF) | ((lm[j] & 0xF) << 4)  # low 4 bits of each
    scales[j-4] |= (ls[j] >> 4) << 6   # high 2 bits of ls[j] → bits 6-7 of scales[j-4]
    scales[j]   |= (lm[j] >> 4) << 6   # high 2 bits of lm[j] → bits 6-7 of scales[j]
```

Decoding (for verification):
```python
ls = [0] * 8
lm = [0] * 8
for j in range(4):
    ls[j] = scales[j] & 0x3F
    lm[j] = scales[j+4] & 0x3F
for j in range(4, 8):
    ls[j] = (scales[j+4] & 0xF) | ((scales[j-4] >> 6) << 4)
    lm[j] = (scales[j+4] >> 4)  | ((scales[j]   >> 6) << 4)
```

## Quantization Algorithm

Reference dequantization from llama.cpp (this is what we must be compatible with):
```
w_dequant[j] = (d * ls[sub]) * q[j] - (dmin * lm[sub])
```
where:
- `d`, `dmin` are the FP16 super-block scale factors
- `ls[sub]`, `lm[sub]` are the 6-bit quantized scale/min for sub-block `sub`
- `q[j]` is the 4-bit quantized weight value (0..15)

Our encoding (min-max approach, compatible with above):

```
For each super-block of 256 weights (organized as 8 sub-blocks of 32):

  # Per sub-block i (32 weights):
  w_min[i] = min(weights[i*32 : (i+1)*32])
  w_max[i] = max(weights[i*32 : (i+1)*32])
  scale[i]  = (w_max[i] - w_min[i]) / 15.0   # [0, 15] quantization range
  if scale[i] < 1e-8: scale[i] = 1e-8         # avoid division by zero
  submin[i] = -w_min[i]                        # store as positive min offset

  # Quantize weights within sub-block:
  q[j] = round((w[j] - w_min[i]) / scale[i])  # in [0, 15]

  # Super-block scale factors:
  d_val    = max(scale[i] for i in 0..7) / 63.0  # scale for ls values
  dmin_val = max(submin[i] for i in 0..7) / 63.0  # scale for lm values

  # Quantize sub-block scales and mins:
  ls[i] = round(scale[i] / d_val)    clamped to [0, 63]
  lm[i] = round(submin[i] / dmin_val)  clamped to [0, 63]

  # Encode: d/dmin as FP16, ls/lm as packed 6-bit in scales[12], q as nibbles in qs[128]
```

## Nibble Packing (qs[128])

256 weights, 4 bits each → 128 bytes. Low nibble = even weight, high nibble = odd weight:
```python
for k in range(128):
    qs[k] = (q[2*k] & 0xF) | ((q[2*k+1] & 0xF) << 4)
```

## FP32 → FP16 Conversion

Use `struct.pack('<e', val)` for FP16 (Python 3.6+ supports `'e'` format).
Clamp to FP16 range before packing: `max(-65504, min(65504, val))`.

## API

**File**: `src/mono_quant/export/gguf/quant_types.py`

```python
GGML_TYPE_F32  = 0
GGML_TYPE_F16  = 1
GGML_TYPE_Q4_0 = 2   # not implemented, listed for reference
GGML_TYPE_Q4_K = 12
GGML_TYPE_Q6_K = 14  # not implemented, listed for reference

def quantize_to_q4_k_s(weight: torch.Tensor) -> bytes:
    """Quantize a 2D weight tensor to Q4_K_S block format.

    Args:
        weight: (out_features, in_features) float32 tensor.
                in_features must be divisible by 256.

    Returns:
        Raw bytes: out_features × (in_features // 256) × 144 bytes.

    Raises:
        ValueError: If in_features is not divisible by 256.
    """
```

Implementation notes:
- Work in Python + PyTorch (no need for Cython/C extension)
- Process one row at a time (outer loop over out_features) for clarity
- Use `torch.float32` throughout; convert to FP16 only at the final packing step
- The function returns `bytes` (not a tensor) — this is raw binary block data

## State Machine: quantize_to_q4_k_s() flow

```
   [weight: (out, in) float32]
              │
              ▼
   [validate: in % 256 == 0]──fail──→ [ValueError]
              │
         for each row r in 0..out:
              │
              ▼
      [split into 8 sub-blocks of 32]
              │
         for each sub-block i:
              ▼
      [compute scale[i], submin[i]]
      [quantize 32 weights → q[i][0..31]]
              │
              ▼
      [compute d_val = max(scale)/63]
      [compute dmin_val = max(submin)/63]
      [compute ls[i] = round(scale[i]/d_val)]
      [compute lm[i] = round(submin[i]/dmin_val)]
              │
              ▼
      [pack d, dmin as FP16 (4 bytes)]
      [pack ls/lm into scales[12]]
      [pack q into qs[128] nibbles]
      [append 144 bytes to output]
              │
   [all rows done]
              │
              ▼
   [return bytes (out × blocks × 144)]
```

## Success Criteria
- [ ] Block size: `len(quantize_to_q4_k_s(torch.zeros(256, 256))) == 256 * 144`
- [ ] Scale packing round-trip: encode → decode → ls/lm match original (±0)
- [ ] Nibble packing round-trip: pack → unpack → q values match original
- [ ] Reconstruction error: dequantized weights within 15% relative MAE of originals
- [ ] in_features not divisible by 256 raises ValueError

## Dependencies
- T-022 (GGUFWriter — quant_types.py is used by the writer, not the other way around)

## Testing Requirements
- `test_q4k_block_size` — verify byte count for known shape
- `test_q4k_scale_packing_round_trip` — encode ls/lm, decode, verify exact match
- `test_q4k_nibble_packing_round_trip` — pack q values, unpack, verify match
- `test_q4k_reconstruction_error` — dequantize and verify relative error < 15%
- `test_q4k_invalid_in_features` — non-multiple of 256 raises ValueError
Coverage target: all branches in quantize_to_q4_k_s

## Open Questions
None.
