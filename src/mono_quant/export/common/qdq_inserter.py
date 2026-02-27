"""QDQ node insertion utilities for ONNX export.

Collects per-channel quantization parameters from quantized modules
and rewires the ONNX graph with QuantizeLinear/DequantizeLinear nodes.
"""

import warnings
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

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


def _fingerprint(arr: "np.ndarray") -> bytes:
    """Byte fingerprint using the first 8 and last 8 float32 values of an array.

    Taking bytes from both ends of the flattened array means that two weight
    matrices with the same leading values but different tails (common in real
    trained models) will not collide.  For random float32 weights the collision
    probability per pair is ~1/(2^512) — negligible.

    Candidates that share a fingerprint are always verified with ``np.allclose``
    before being accepted, so false positives are impossible regardless.
    """
    import numpy as np

    flat = np.ascontiguousarray(arr).ravel().view(np.uint8)
    n = len(flat)
    head = flat[:min(32, n)].tobytes()   # 8 float32 = 32 bytes
    tail = flat[max(0, n - 32):].tobytes()
    return head + tail


def _build_dynamo_name_map(
    model: nn.Module,
    model_proto: "onnx.ModelProto",  # noqa: F821
) -> Dict[str, Tuple[str, bool]]:
    """Match anonymous ONNX initializers to named model parameters.

    When dynamo exports ``F.linear(x, w, b)`` as ``MatMul(x, w.T)``, the
    transposed weight is stored as an anonymous ``val_N`` constant and the
    original parameter name is lost.  This function recovers the mapping by
    fingerprint-indexed value comparison — O(P) to build the index, O(1) per
    lookup for typical trained models where all weights have unique values.

    Args:
        model: FP32 model (after ``revert_to_standard_modules``) used for export.
        model_proto: ONNX ModelProto from a dynamo export.

    Returns:
        Dict mapping ``{onnx_init_name: (param_dotted_name, is_transposed)}``.
        ``is_transposed=True`` means the ONNX initializer is ``param.weight.T``.
        Returns an empty dict when no anonymous weight-like initializers are found.
    """
    import numpy as np
    from onnx import numpy_helper

    # Collect anonymous (no dots), 2-D+ initializers.
    # 1-D vectors and scalars are excluded by the ndim guard.
    anon_inits: Dict[str, "np.ndarray"] = {}
    for init in model_proto.graph.initializer:
        if "." not in init.name and len(init.dims) >= 2:
            arr = numpy_helper.to_array(init)
            if arr.size > 0:
                anon_inits[init.name] = np.ascontiguousarray(arr.astype(np.float32))

    if not anon_inits:
        return {}

    # Build a fingerprint index over all model parameters — both direct and
    # transposed orientations.  This makes the dominant lookup O(1) per
    # anonymous initializer instead of O(P_same_shape × W).
    #
    # fp_index: fingerprint → [(param_name, contiguous_array, is_transposed)]
    fp_index: Dict[bytes, list] = defaultdict(list)
    for param_name, param in model.named_parameters():
        arr = np.ascontiguousarray(param.detach().cpu().float().numpy())
        fp_index[_fingerprint(arr)].append((param_name, arr, False))
        if arr.ndim == 2:
            arr_t = np.ascontiguousarray(arr.T)
            fp_index[_fingerprint(arr_t)].append((param_name, arr_t, True))

    name_map: Dict[str, Tuple[str, bool]] = {}
    for anon_name, anon_arr in anon_inits.items():
        candidates = fp_index.get(_fingerprint(anon_arr), [])
        for param_name, param_arr, is_transposed in candidates:
            # Shape check before allclose guards against fingerprint collisions.
            if anon_arr.shape == param_arr.shape and np.allclose(anon_arr, param_arr, atol=1e-5):
                name_map[anon_name] = (param_name, is_transposed)
                break

    return name_map


def insert_qdq_nodes(
    model_proto: "onnx.ModelProto",  # noqa: F821
    qparams: Dict[str, LayerQParams],
    dynamo_name_map: Optional[Dict[str, Tuple[str, bool]]] = None,
) -> "onnx.ModelProto":  # noqa: F821
    """Insert QuantizeLinear + DequantizeLinear nodes for each INT8 layer.

    For each entry in qparams (is_int4=False only), this function:
    1. Locates the ``{name}.weight`` initializer in the ONNX graph.
    2. Adds scale and zero_point initializers.
    3. Creates a QuantizeLinear → DequantizeLinear node pair.
    4. Rewires the Gemm/MatMul node to use the dequantized weight.

    QDQ pattern inserted:
        weight_fp32 → QuantizeLinear(scale, zp) → weight_q
                    → DequantizeLinear(scale, zp) → weight_dq
                    → Gemm(activation, weight_dq, bias)

    Args:
        model_proto: ONNX model protobuf (mutated in place).
        qparams: Dict of module name → LayerQParams. INT4 entries are skipped.
        dynamo_name_map: Optional mapping from ``_build_dynamo_name_map``.
            When provided, used as a third fallback to locate weights that
            dynamo stored as anonymous ``val_N`` constants (e.g. transposed
            attention projection weights stored as MatMul inputs).

    Returns:
        The modified model_proto.
    """
    import numpy as np
    import onnx
    from onnx import numpy_helper

    graph = model_proto.graph

    # Map from initializer name → True (fast lookup)
    init_names = {init.name for init in graph.initializer}

    # Build reverse lookup from dynamo_name_map for O(1) access.
    # dynamo_name_map: {onnx_name → (param_dotted_name, is_transposed)}
    # reverse:         {param_dotted_name → (onnx_name, is_transposed)}
    reverse_dynamo_map: Dict[str, Tuple[str, bool]] = {}
    if dynamo_name_map:
        for onnx_name, (param_name, is_transposed) in dynamo_name_map.items():
            reverse_dynamo_map[param_name] = (onnx_name, is_transposed)

    # Collect (position, [nodes_to_insert_before]) keyed by original node index
    nodes_to_insert: Dict[int, list] = {}

    for module_name, params in qparams.items():
        if params.is_int4:
            continue  # INT4 QDQ deferred to opset>=21

        is_weight_transposed = False
        weight_name = f"{module_name}.weight"

        if weight_name not in init_names:
            # Fallback 1: suffix match (handles name-prefix differences)
            candidates = [n for n in init_names if n.endswith(f"{module_name}.weight")]
            if candidates:
                weight_name = candidates[0]
            # Fallback 2: dynamo name map — weight stored as val_N (w.T pattern)
            elif weight_name in reverse_dynamo_map:
                weight_name, is_weight_transposed = reverse_dynamo_map[weight_name]
            else:
                warnings.warn(
                    f"Could not find ONNX initializer for '{module_name}.weight'. "
                    "QDQ nodes will not be inserted for this layer.",
                    UserWarning,
                    stacklevel=3,
                )
                continue

        # When dynamo stores the weight transposed (val_N = w.T), the per-channel
        # quantization axis shifts: original axis 0 on [out, in] becomes axis 1
        # on the transposed [in, out] tensor.
        qdq_axis = 1 if (is_weight_transposed and params.axis == 0) else params.axis

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
            axis=qdq_axis,
        )

        # DequantizeLinear: INT8 → FP32 (fake-quantized)
        dq_node = onnx.helper.make_node(
            "DequantizeLinear",
            inputs=[weight_q_name, scale_name, zp_name],
            outputs=[weight_dq_name],
            axis=qdq_axis,
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
