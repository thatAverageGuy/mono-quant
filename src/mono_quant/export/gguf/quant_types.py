"""GGUF quantization type codes and block-format packing.

Implements Q4_K_S: 4-bit k-quantization (small variant — all layers use Q4_K blocks).

Block structure for Q4_K (GGML_TYPE_Q4_K = 12):
    Offset  Size   Field
    0       2      d     (FP16) — super-block scale for quantized scales
    2       2      dmin  (FP16) — super-block scale for quantized mins
    4       12     scales — 8 sub-block scales + 8 sub-block mins, 6 bits each
    16      128    qs    — 256 × 4-bit nibbles (lo-nibble = even weight)
    ─────────────────────
    Total: 144 bytes per 256-weight super-block.

References:
    ggml/src/ggml-quants.h   — block_q4_K struct
    ggml/src/ggml-quants.c   — quantize_row_q4_K_ref, dequantize_row_q4_K
"""

import struct

import torch

# ---------------------------------------------------------------------------
# GGML tensor type codes (subset — only what we use or reference)
# ---------------------------------------------------------------------------

GGML_TYPE_F32  = 0
GGML_TYPE_F16  = 1
GGML_TYPE_Q4_0 = 2   # not implemented — listed for reference
GGML_TYPE_Q4_K = 12
GGML_TYPE_Q6_K = 14  # not implemented — listed for reference

# Weights per Q4_K super-block
_QK_K = 256
# Sub-blocks per super-block
_N_SUB = 8
# Weights per sub-block
_SUB_SIZE = _QK_K // _N_SUB   # 32
# Block size in bytes
Q4_K_BLOCK_BYTES = 144


def quantize_to_q4_k_s(weight: torch.Tensor) -> bytes:
    """Quantize a 2D weight tensor to Q4_K_S block format.

    Produces raw bytes compatible with llama.cpp's dequantize_row_q4_K().

    Args:
        weight: (out_features, in_features) float32 tensor.
                in_features must be divisible by 256.

    Returns:
        Raw bytes: out_features × (in_features // 256) × 144 bytes.

    Raises:
        ValueError: If in_features is not divisible by 256.
    """
    if weight.dim() != 2:
        raise ValueError(f"Expected 2D weight tensor, got shape {tuple(weight.shape)}")

    out_features, in_features = weight.shape
    if in_features % _QK_K != 0:
        raise ValueError(
            f"in_features ({in_features}) must be divisible by {_QK_K} for Q4_K packing."
        )

    n_blocks_per_row = in_features // _QK_K
    w = weight.float()

    out = bytearray()

    for row in range(out_features):
        row_data = w[row]  # (in_features,)
        for blk in range(n_blocks_per_row):
            start = blk * _QK_K
            super_block = row_data[start : start + _QK_K]
            out += _pack_q4_k_block(super_block)

    return bytes(out)


# ---------------------------------------------------------------------------
# Internal: pack one 256-weight super-block into 144 bytes
# ---------------------------------------------------------------------------

def _pack_q4_k_block(super_block: torch.Tensor) -> bytes:
    """Pack a 256-weight 1D tensor into one 144-byte Q4_K block.

    Args:
        super_block: 1D float32 tensor of length 256.

    Returns:
        144 bytes in Q4_K wire format.
    """
    scales   = []   # per-sub-block float scale
    submins  = []   # per-sub-block float -min  (positive)
    q_values = []   # per-sub-block uint4 list, shape (32,)

    for i in range(_N_SUB):
        sub = super_block[i * _SUB_SIZE : (i + 1) * _SUB_SIZE]
        w_min = sub.min().item()
        w_max = sub.max().item()

        scale_i = (w_max - w_min) / 15.0
        if scale_i < 1e-8:
            scale_i = 1e-8

        submin_i = -w_min  # positive value

        q = ((sub - w_min) / scale_i).round().clamp(0, 15).to(torch.int32).tolist()

        scales.append(scale_i)
        submins.append(submin_i)
        q_values.append(q)

    # Super-block scale factors (max across sub-blocks)
    max_scale  = max(scales)
    max_submin = max(submins)

    d_val    = max_scale  / 63.0 if max_scale  > 1e-8 else 1e-8
    dmin_val = max_submin / 63.0 if max_submin > 1e-8 else 1e-8

    # 6-bit quantized scale and min per sub-block
    ls = [min(63, round(s / d_val))    for s in scales ]
    lm = [min(63, round(m / dmin_val)) for m in submins]

    # Pack d and dmin as FP16 (2 bytes each)
    d_fp16    = _f32_to_fp16_bytes(d_val)
    dmin_fp16 = _f32_to_fp16_bytes(dmin_val)

    # Pack 8 ls + 8 lm as 6-bit values into 12 bytes
    scales_bytes = _pack_6bit_scales(ls, lm)

    # Pack 256 nibbles (4-bit values) into 128 bytes
    flat_q = [v for sub in q_values for v in sub]  # 256 values
    qs = bytearray(128)
    for k in range(128):
        qs[k] = (flat_q[2 * k] & 0xF) | ((flat_q[2 * k + 1] & 0xF) << 4)

    return d_fp16 + dmin_fp16 + bytes(scales_bytes) + bytes(qs)


def _pack_6bit_scales(ls: list, lm: list) -> bytearray:
    """Pack 8 scales and 8 mins (each 0..63) into 12 bytes.

    Packing scheme (matches llama.cpp quantize_row_q4_K_ref):
        For j in 0..3:  scales[j]   = ls[j] & 0x3F
                        scales[j+4] = lm[j] & 0x3F
        For j in 4..7:  scales[j+4] = (ls[j] & 0xF) | ((lm[j] & 0xF) << 4)
                        scales[j-4] |= (ls[j] >> 4) << 6
                        scales[j]   |= (lm[j] >> 4) << 6

    Args:
        ls: List of 8 ints in [0, 63] — sub-block scales.
        lm: List of 8 ints in [0, 63] — sub-block mins.

    Returns:
        bytearray of length 12.
    """
    b = bytearray(12)
    for j in range(4):
        b[j]     = ls[j] & 0x3F
        b[j + 4] = lm[j] & 0x3F
    for j in range(4, 8):
        b[j + 4]  = (ls[j] & 0xF) | ((lm[j] & 0xF) << 4)
        b[j - 4] |= (ls[j] >> 4) << 6
        b[j]     |= (lm[j] >> 4) << 6
    return b


def unpack_6bit_scales(b: bytes) -> tuple:
    """Decode 12 bytes back into 8 ls and 8 lm values (for testing).

    Returns:
        (ls, lm) — two lists of 8 ints each, values in [0, 63].
    """
    b = bytearray(b)
    ls = [0] * 8
    lm = [0] * 8
    for j in range(4):
        ls[j] = b[j]     & 0x3F
        lm[j] = b[j + 4] & 0x3F
    for j in range(4, 8):
        ls[j] = (b[j + 4] & 0xF)  | ((b[j - 4] >> 6) << 4)
        lm[j] = (b[j + 4] >> 4)   | ((b[j]     >> 6) << 4)
    return ls, lm


def _f32_to_fp16_bytes(val: float) -> bytes:
    """Convert a float32 to 2 bytes of FP16 (little-endian).

    Clamps to the FP16 representable range before packing.
    """
    val = max(-65504.0, min(65504.0, val))
    return struct.pack("<e", val)
