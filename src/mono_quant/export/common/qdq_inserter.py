"""QDQ node insertion utilities for ONNX export.

Collects per-channel quantization parameters from quantized modules
and rewires the ONNX graph with QuantizeLinear/DequantizeLinear nodes.
"""

import warnings
from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn


@dataclass
class LayerQParams:
    """Quantization parameters for a single layer."""

    int8_weight: Optional[torch.Tensor]  # int8 repr, None for INT4
    scales: torch.Tensor                  # per-channel float32
    zero_points: torch.Tensor             # per-channel int32
    axis: int                             # quantization axis (typically 0)
    is_int4: bool = False


def collect_quantization_params(
    model: nn.Module,
    opset: int = 14,
) -> Dict[str, LayerQParams]:
    """Walk model and extract quantization parameters from quantized modules.

    Args:
        model: Quantized nn.Module (before reverting to standard modules).
        opset: ONNX opset version. INT4 QDQ requires opset>=21; below that,
               INT4 layers are collected with is_int4=True so the caller can warn.

    Returns:
        Dict mapping module path (e.g. "0" or "encoder.fc") to LayerQParams.
    """
    from mono_quant.modules.linear import QuantizedConv2d, QuantizedLinear, QuantizedLinearInt4

    qparams: Dict[str, LayerQParams] = {}

    for name, module in model.named_modules():
        if isinstance(module, (QuantizedLinear, QuantizedConv2d)):
            if module._quantized_weight is None:
                continue
            qparams[name] = LayerQParams(
                int8_weight=module._quantized_weight.int_repr().cpu(),
                scales=module._quantized_weight.q_per_channel_scales().cpu(),
                zero_points=module._quantized_weight.q_per_channel_zero_points().cpu(),
                axis=module._quantized_weight.q_per_channel_axis(),
                is_int4=False,
            )
        elif isinstance(module, QuantizedLinearInt4):
            # Per-group INT4: QDQ requires opset>=21, defer to future work
            qparams[name] = LayerQParams(
                int8_weight=None,
                scales=module._scales.cpu(),
                zero_points=module._zero_points.cpu(),
                axis=0,
                is_int4=True,
            )

    return qparams


def insert_qdq_nodes(
    model_proto: "onnx.ModelProto",  # noqa: F821
    qparams: Dict[str, LayerQParams],
) -> "onnx.ModelProto":  # noqa: F821
    """Insert QuantizeLinear + DequantizeLinear nodes for each INT8 layer.

    For each entry in qparams (is_int4=False only), this function:
    1. Locates the ``{name}.weight`` initializer in the ONNX graph.
    2. Adds scale and zero_point initializers.
    3. Creates a QuantizeLinear → DequantizeLinear node pair.
    4. Rewires the Gemm/Conv node to use the dequantized weight.

    QDQ pattern inserted:
        weight_fp32 → QuantizeLinear(scale, zp) → weight_q
                    → DequantizeLinear(scale, zp) → weight_dq
                    → Gemm(activation, weight_dq, bias)

    Args:
        model_proto: ONNX model protobuf (mutated in place).
        qparams: Dict of module name → LayerQParams. INT4 entries are skipped.

    Returns:
        The modified model_proto.
    """
    import numpy as np
    import onnx
    from onnx import numpy_helper

    graph = model_proto.graph

    # Map from initializer name → True (fast lookup)
    init_names = {init.name for init in graph.initializer}

    # Collect (position, [nodes_to_insert_before]) keyed by original node index
    nodes_to_insert: Dict[int, list] = {}

    for module_name, params in qparams.items():
        if params.is_int4:
            continue  # INT4 QDQ deferred to opset>=21

        weight_name = f"{module_name}.weight"

        if weight_name not in init_names:
            # Fallback: try to match by suffix in case of name mangling
            candidates = [n for n in init_names if n.endswith(f"{module_name}.weight")]
            if candidates:
                weight_name = candidates[0]
            else:
                warnings.warn(
                    f"Could not find ONNX initializer for '{module_name}.weight'. "
                    "QDQ nodes will not be inserted for this layer.",
                    UserWarning,
                    stacklevel=3,
                )
                continue

        # Unique names for new tensors
        scale_name = f"__qdq_{module_name}_scale"
        zp_name = f"__qdq_{module_name}_zero_point"
        weight_q_name = f"__qdq_{module_name}_quantized"
        weight_dq_name = f"__qdq_{module_name}_dequantized"

        # Scales: float32 per-channel
        scales_np = params.scales.float().numpy()
        graph.initializer.append(numpy_helper.from_array(scales_np, name=scale_name))

        # Zero points: int8 per-channel (INT8 quantization)
        zp_np = params.zero_points.numpy().astype(np.int8)
        graph.initializer.append(numpy_helper.from_array(zp_np, name=zp_name))

        # QuantizeLinear: FP32 weight → INT8
        q_node = onnx.helper.make_node(
            "QuantizeLinear",
            inputs=[weight_name, scale_name, zp_name],
            outputs=[weight_q_name],
            axis=params.axis,
        )

        # DequantizeLinear: INT8 → FP32 (fake-quantized)
        dq_node = onnx.helper.make_node(
            "DequantizeLinear",
            inputs=[weight_q_name, scale_name, zp_name],
            outputs=[weight_dq_name],
            axis=params.axis,
        )

        # Rewire: find graph nodes that take weight_name as input, redirect to weight_dq_name
        for idx, node in enumerate(graph.node):
            rewired = False
            for j, inp in enumerate(node.input):
                if inp == weight_name:
                    node.input[j] = weight_dq_name
                    rewired = True
            if rewired:
                nodes_to_insert.setdefault(idx, []).extend([q_node, dq_node])

    if not nodes_to_insert:
        return model_proto

    # Rebuild node list, inserting QDQ pairs just before the nodes that use them
    old_nodes = list(graph.node)
    del graph.node[:]
    for i, node in enumerate(old_nodes):
        if i in nodes_to_insert:
            graph.node.extend(nodes_to_insert[i])
        graph.node.append(node)

    return model_proto
