"""GPTQ INT4 quantization and bit-packing utilities.

Implements the AutoGPTQ V1 format:
  - qweight shape: (in_features // 8, out_features), int32, LSB-first packing
  - qzeros  shape: (num_groups, out_features // 8), int32, stores (zero - 1)
  - scales  shape: (num_groups, out_features), float16
  - g_idx   shape: (in_features,), int32, value = i // group_size
"""

from typing import Tuple

import torch


def _pack_int4_rows_to_int32(
    w_uint4: torch.Tensor,
) -> torch.Tensor:
    """Pack 8 consecutive rows of uint4 values into one int32 per row.

    Args:
        w_uint4: (in_features, out_features) tensor with values in [0, 15].

    Returns:
        (in_features // 8, out_features) int32 tensor.
        Row k packs w_uint4[8k], w_uint4[8k+1], ..., w_uint4[8k+7]
        into bits 3:0, 7:4, ..., 31:28 of each output element.
    """
    in_features, out_features = w_uint4.shape
    n_rows = in_features // 8
    w = w_uint4.reshape(n_rows, 8, out_features).to(torch.int32)
    shifts = torch.arange(8, dtype=torch.int32, device=w.device) * 4  # [0,4,8,...,28]
    return (w << shifts.view(1, 8, 1)).sum(dim=1).to(torch.int32)


def _pack_int4_cols_to_int32(
    z_uint4: torch.Tensor,
) -> torch.Tensor:
    """Pack 8 consecutive columns of uint4 values into one int32 per column group.

    Args:
        z_uint4: (num_groups, out_features) tensor with values in [0, 15].

    Returns:
        (num_groups, out_features // 8) int32 tensor.
        Column group k packs z_uint4[:, 8k], ..., z_uint4[:, 8k+7]
        into bits 3:0, 7:4, ..., 31:28.
    """
    num_groups, out_features = z_uint4.shape
    n_cols = out_features // 8
    z = z_uint4.reshape(num_groups, n_cols, 8).to(torch.int32)
    shifts = torch.arange(8, dtype=torch.int32, device=z.device) * 4
    return (z << shifts.view(1, 1, 8)).sum(dim=2).to(torch.int32)


def unpack_gptq_weight(
    qweight: torch.Tensor,
    in_features: int,
) -> torch.Tensor:
    """Unpack a GPTQ qweight tensor back to uint4 values.

    Args:
        qweight: (in_features // 8, out_features) int32 tensor.
        in_features: Original in_features dimension.

    Returns:
        (in_features, out_features) int32 tensor with values in [0, 15].
    """
    out_features = qweight.shape[1]
    mask = torch.tensor(0xF, dtype=torch.int32, device=qweight.device)
    rows = []
    for bit_offset in range(8):
        rows.append((qweight >> (bit_offset * 4)) & mask)
    # rows[i] is (in_features // 8, out_features)
    # Stack along a new dim then reshape
    stacked = torch.stack(rows, dim=1)  # (in_features // 8, 8, out_features)
    return stacked.reshape(in_features, out_features)


def quantize_to_gptq_int4(
    weight: torch.Tensor,
    group_size: int = 128,
    sym: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize a weight tensor to GPTQ INT4 format.

    Groups are formed along the in_features axis (columns of the weight matrix).

    Args:
        weight: (out_features, in_features) float tensor.
        group_size: Number of input features per quantization group.
        sym: If True, use symmetric quantization (zero=8). If False, asymmetric.

    Returns:
        Tuple of (qweight, qzeros, scales, g_idx):
          - qweight: (in_features // 8, out_features) int32
          - qzeros:  (num_groups, out_features // 8) int32, V1 encoding (zero - 1)
          - scales:  (num_groups, out_features) float16
          - g_idx:   (in_features,) int32

    Raises:
        ValueError: If in_features is not divisible by group_size.
    """
    out_features, in_features = weight.shape
    if in_features % group_size != 0:
        raise ValueError(
            f"in_features ({in_features}) must be divisible by group_size ({group_size})"
        )
    if out_features % 8 != 0:
        raise ValueError(
            f"out_features ({out_features}) must be divisible by 8 for GPTQ packing"
        )

    num_groups = in_features // group_size

    # Reshape to (out_features, num_groups, group_size) for per-group stats
    w = weight.float().reshape(out_features, num_groups, group_size)
    w_max = w.amax(dim=-1)  # (out_features, num_groups)
    w_min = w.amin(dim=-1)  # (out_features, num_groups)

    if sym:
        absmax = torch.maximum(w_max.abs(), w_min.abs())
        scales = (absmax / 8.0).clamp(min=1e-8)        # (out_features, num_groups)
        zeros = torch.full_like(scales, 8.0)
    else:
        scales = ((w_max - w_min) / 15.0).clamp(min=1e-8)
        zeros = (-w_min / scales).round().clamp(0, 15)

    # Quantize to unsigned [0, 15]
    q = (w / scales.unsqueeze(-1) + zeros.unsqueeze(-1)).round().clamp(0, 15)
    # q: (out_features, num_groups, group_size) → flatten in_features
    q = q.reshape(out_features, in_features).to(torch.int32)

    # qweight: transpose → (in_features, out_features), then pack rows
    qweight = _pack_int4_rows_to_int32(q.T.contiguous())  # (in_features // 8, out_features)

    # scales: (num_groups, out_features), float16  [permute from (O, G) → (G, O)]
    scales_out = scales.permute(1, 0).to(torch.float16)

    # qzeros: (num_groups, out_features), V1 stores (zero - 1), then pack cols
    zeros_T = zeros.permute(1, 0).to(torch.int32)           # (num_groups, out_features)
    qzeros = _pack_int4_cols_to_int32((zeros_T - 1).clamp(0, 15))  # (G, out_features // 8)

    g_idx = torch.arange(in_features, dtype=torch.int32) // group_size

    return qweight, qzeros, scales_out, g_idx
