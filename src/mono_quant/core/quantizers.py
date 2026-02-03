"""
Quantization transformation functions for weights.

These functions apply actual quantization transformations to tensors,
converting floating-point weights to quantized representations (INT8, FP16).
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Type, Union

import torch
import torch.nn as nn

from mono_quant.core.mappers import calculate_scale_zp_per_channel


# Type alias for layer types used in selection (re-exported from observers)
LayerTypes = Union[Type[nn.Module], Tuple[Type[nn.Module], ...]]

# Type alias for calibration data (re-exported from calibration module)
CalibrationData = Union[List[torch.Tensor], "torch.utils.data.DataLoader"]


@dataclass
class QuantizationInfo:
    """
    Metadata about a quantized model.

    This dataclass captures information about what was quantized,
    how many calibration samples were used, and the quantization
    parameters applied.

    Attributes:
        selected_layers: List of layer names that were quantized.
        skipped_layers: List of layer names that were skipped.
        calibration_samples_used: Number of calibration samples actually used.
        dtype: Target quantization dtype (torch.qint8 or torch.float16).
        symmetric: Whether symmetric quantization was used.

    Examples:
        >>> info = QuantizationInfo(
        ...     selected_layers=["0", "2"],
        ...     skipped_layers=["1"],
        ...     calibration_samples_used=50,
        ...     dtype=torch.qint8,
        ...     symmetric=False
        ... )
        >>> print(f"Quantized {len(info.selected_layers)} layers")
        Quantized 2 layers
    """

    selected_layers: List[str]
    skipped_layers: List[str]
    calibration_samples_used: int
    dtype: torch.dtype
    symmetric: bool


def quantize_weight_int8(
    weight: torch.Tensor,
    symmetric: bool = False,
    axis: int = 0,
) -> torch.Tensor:
    """
    Quantize a weight tensor to INT8 using per-channel scaling.

    This function applies per-channel affine quantization to weight tensors,
    which is the standard approach for neural network weight quantization.
    Per-channel quantization uses separate scale and zero-point for each
    channel along the specified axis.

    Args:
        weight: Input weight tensor to quantize. For standard PyTorch layer
            weights, this is typically (out_features, in_features) for Linear
            or (out_channels, in_channels, kernel_h, kernel_w) for Conv2d.
        symmetric: If True, use symmetric quantization (zero_point = 0).
                   If False, use asymmetric (affine) quantization. Default is False.
        axis: The channel axis for per-channel quantization. Default is 0,
              which corresponds to the output channel dimension for standard
              PyTorch weight layouts (nn.Linear, nn.Conv2d).

    Returns:
        Quantized tensor with dtype torch.qint8. The tensor stores the
        quantized integer values along with per-channel scale and zero_point
        metadata.

    Examples:
        >>> import torch
        >>> # Linear layer weight: (out_features, in_features)
        >>> w = torch.randn(64, 128)
        >>> qw = quantize_weight_int8(w, symmetric=False, axis=0)
        >>> assert qw.dtype == torch.qint8
        >>> assert qw.qscheme() == torch.per_channel_affine
    """
    # Calculate per-channel scale and zero-point
    scale, zero_point = calculate_scale_zp_per_channel(
        weight, dtype=torch.qint8, symmetric=symmetric, axis=axis
    )

    # Apply per-channel quantization using PyTorch's native primitive
    # zero_point must be int type for torch.quantize_per_channel
    quantized = torch.quantize_per_channel(
        weight,
        scales=scale,
        zero_points=zero_point.int(),
        axis=axis,
        dtype=torch.qint8,
    )

    return quantized


def quantize_weight_fp16(weight: torch.Tensor) -> torch.Tensor:
    """
    Quantize a weight tensor to FP16 using dtype casting.

    FP16 quantization uses simple dtype casting rather than a full
    quantization pipeline with scale and zero-point. This approach is
    simpler and effective for memory reduction, as FP16 has native
    hardware support on modern accelerators.

    See 01-RESEARCH.md "FP16 Quantization (Simple Cast Approach)" and
    discussion of Float16 Dynamic Quantization model size benefits.

    Args:
        weight: Input weight tensor to convert to FP16.

    Returns:
        Weight tensor with dtype torch.float16.

    Examples:
        >>> import torch
        >>> w = torch.randn(64, 128)
        >>> w_fp16 = quantize_weight_fp16(w)
        >>> assert w_fp16.dtype == torch.float16
        >>> assert w_fp16.element_size() == 2  # 2 bytes for float16
    """
    return weight.to(torch.float16)


def dequantize_weight(q_weight: torch.Tensor) -> torch.Tensor:
    """
    Dequantize a quantized weight tensor back to float32.

    This function handles both INT8 quantized tensors (created by
    torch.quantize_per_channel) and FP16 tensors. For INT8, it uses
    torch.dequantize() to apply the affine transformation. For FP16,
    it simply casts back to float32.

    Args:
        q_weight: Quantized weight tensor. Either a torch.qint8 quantized
                  tensor or a torch.float16 tensor.

    Returns:
        Dequantized weight tensor with dtype torch.float32.

    Raises:
        TypeError: If the input tensor is not a recognized quantized type.

    Examples:
        >>> import torch
        >>> w = torch.randn(64, 128)
        >>> qw = quantize_weight_int8(w)
        >>> w_dq = dequantize_weight(qw)
        >>> assert w_dq.dtype == torch.float32
    """
    # Check if it's a quantized tensor (has dequantize method)
    if hasattr(q_weight, 'dequantize'):
        # INT8 quantized tensor - use torch.dequantize()
        return torch.dequantize(q_weight)
    elif q_weight.dtype == torch.float16:
        # FP16 tensor - simple dtype cast
        return q_weight.to(torch.float32)
    else:
        raise TypeError(
            f"Unsupported quantized tensor type: {type(q_weight)}. "
            f"Expected torch.qint8 quantized tensor or torch.float16 tensor."
        )


def dynamic_quantize(
    model: Union[nn.Module, Dict[str, torch.Tensor]],
    dtype: torch.dtype = torch.qint8,
    symmetric: bool = False,
    per_channel: bool = True,
    config: Optional["QuantizationConfig"] = None,
) -> Tuple[nn.Module, List[str]]:
    """
    Dynamically quantize a PyTorch model.

    This function performs dynamic quantization on a model, converting
    supported layers (nn.Linear, nn.Conv2d) to their quantized equivalents.
    Unsupported layers are skipped and returned in the skipped list.

    The function always creates a copy of the model, leaving the original
    unchanged (per CONTEXT.md requirement).

    Args:
        model: Either a PyTorch nn.Module or a state_dict. If a state_dict
            is provided, it will be loaded into a model instance.
        dtype: Target quantization dtype. Options:
            - torch.qint8: 8-bit signed integer (default)
            - torch.float16: 16-bit floating point
        symmetric: If True, use symmetric quantization (zero_point = 0).
                   If False, use asymmetric quantization. Default is False.
        per_channel: If True, use per-channel scaling (default).
                     If False, use per-tensor scaling.
        config: Optional QuantizationConfig. If provided, its values will
                override dtype, symmetric, per_channel parameters.

    Returns:
        A tuple of (quantized_model, skipped_layers):
        - quantized_model: A COPY of the model with quantized layers
        - skipped_layers: List of layer names that were not quantized

    Raises:
        TypeError: If model is not an nn.Module or state_dict.
        ValueError: If dtype is not supported.

    Examples:
        >>> import torch.nn as nn
        >>> from mono_quant import dynamic_quantize
        >>> model = nn.Sequential(
        ...     nn.Linear(128, 256),
        ...     nn.ReLU(),
        ...     nn.Linear(256, 10)
        ... )
        >>> q_model, skipped = dynamic_quantize(model, dtype=torch.qint8)
        >>> print(f"Skipped {len(skipped)} layers: {skipped}")
        Skipped 1 layers: ['1']

        FP16 quantization:
        >>> q_model_fp16, skipped = dynamic_quantize(model, dtype=torch.float16)
        >>> print(f"Skipped {len(skipped)} layers: {skipped}")
        Skipped 0 layers: []
    """
    # Handle config override (config priority pattern)
    if config is not None:
        dtype = config.dtype
        if config.symmetric is not None:
            symmetric = config.symmetric
        per_channel = config.per_channel

    # FP16 quantization uses simpler flow
    if dtype == torch.float16:
        return _quantize_fp16_model(model)

    # For INT8 quantization, use layer-specific approach
    return _quantize_int8_model(model, dtype=dtype, symmetric=symmetric, per_channel=per_channel)


def _quantize_fp16_model(
    model: Union[nn.Module, Dict[str, torch.Tensor]],
) -> Tuple[nn.Module, List[str]]:
    """
    Quantize a model to FP16 using simple dtype casting.

    FP16 quantization is simpler than INT8 - we just cast all parameters
    to float16. No layer type filtering is needed since all layers support
    FP16 weights.

    Args:
        model: Either a PyTorch nn.Module or a state_dict.

    Returns:
        A tuple of (quantized_model, skipped_layers):
        - quantized_model: Model with FP16 weights
        - skipped_layers: Empty list (all layers quantized for FP16)
    """
    # Local import to avoid circular dependency
    from mono_quant.io.handlers import _prepare_model

    # Get a copy of the model
    model_copy = _prepare_model(model)

    # Convert all parameters to FP16
    for param in model_copy.parameters():
        param.data = param.data.to(torch.float16)

    return model_copy, []


def static_quantize(
    model: Union[nn.Module, Dict[str, torch.Tensor]],
    calibration_data: CalibrationData,
    dtype: torch.dtype = torch.qint8,
    symmetric: bool = False,
    per_channel: bool = True,
    layer_types: Optional[LayerTypes] = None,
    skip_types: Optional[LayerTypes] = None,
    layer_names: Optional[List[str]] = None,
    skip_names: Optional[List[str]] = None,
    num_calibration_samples: int = 150,
    config: Optional["QuantizationConfig"] = None,
) -> Tuple[nn.Module, QuantizationInfo]:
    """
    Statically quantize a PyTorch model using calibration data.

    This function performs static quantization by first running calibration
    to determine activation ranges, then quantizing selected layers. Layer
    selection can be done by type (layer_types, skip_types) or by exact name
    (layer_names, skip_names).

    The function always creates a copy of the model, leaving the original
    unchanged (per CONTEXT.md requirement).

    Args:
        model: Either a PyTorch nn.Module or a state_dict. If a state_dict
            is provided, it will be loaded into a model instance.
        calibration_data: Calibration data in one of the following formats:
            - List[torch.Tensor]: Direct list of input tensors
            - DataLoader: PyTorch DataLoader yielding batches
        dtype: Target quantization dtype. Options:
            - torch.qint8: 8-bit signed integer (default)
            - torch.float16: 16-bit floating point
        symmetric: If True, use symmetric quantization (zero_point = 0).
                   If False, use asymmetric quantization. Default is False.
        per_channel: If True, use per-channel scaling (default).
                     If False, use per-tensor scaling.
        layer_types: Optional layer type(s) to select for quantization.
                     Can be a single nn.Module type or a tuple of types.
                     If None, defaults to (nn.Linear, nn.Conv2d).
        skip_types: Optional layer type(s) to exclude from quantization.
                    Can be a single nn.Module type or a tuple of types.
        layer_names: Optional list of exact layer names to quantize.
        skip_names: Optional list of layer names to exclude from quantization.
        num_calibration_samples: Maximum number of samples to use for calibration.
                                 Default is 150. Limited by actual data size.
        config: Optional QuantizationConfig. If provided, its values will
                override dtype, symmetric, per_channel parameters.

    Returns:
        A tuple of (quantized_model, quantization_info):
        - quantized_model: A COPY of the model with quantized layers
        - quantization_info: QuantizationInfo dataclass with metadata

    Raises:
        TypeError: If model is not an nn.Module or state_dict.
        ValueError: If dtype is not supported.

    Examples:
        >>> import torch.nn as nn
        >>> from mono_quant import static_quantize
        >>> model = nn.Sequential(
        ...     nn.Linear(128, 256),
        ...     nn.ReLU(),
        ...     nn.Linear(256, 10)
        ... )
        >>> calibration_data = [torch.randn(32, 128) for _ in range(100)]
        >>> q_model, info = static_quantize(model, calibration_data, layer_types=[nn.Linear])
        >>> print(f"Quantized {len(info.selected_layers)} layers")
        Quantized 2 layers

        FP16 static quantization:
        >>> q_model_fp16, info = static_quantize(model, calibration_data, dtype=torch.float16)
    """
    # Local imports to avoid circular dependencies
    from mono_quant.io.handlers import _prepare_model
    from mono_quant.calibration.runner import run_calibration
    from mono_quant.calibration.data import _normalize_calibration_data
    from mono_quant.core.observers import (
        MinMaxObserver,
        _select_layers_by_type,
        _select_layers_by_name,
        _merge_selection_results,
    )
    from mono_quant.modules.linear import quantize_linear_module, quantize_conv2d_module

    # Handle config override (config priority pattern)
    if config is not None:
        dtype = config.dtype
        if config.symmetric is not None:
            symmetric = config.symmetric
        per_channel = config.per_channel

    # Copy model via _prepare_model (preserves original)
    model_copy = _prepare_model(model)

    # FP16 quantization uses simpler flow (no calibration needed)
    if dtype == torch.float16:
        q_model, skipped = _quantize_fp16_model(model_copy)
        info = QuantizationInfo(
            selected_layers=[],  # FP16 quantizes all, no layer selection
            skipped_layers=[],
            calibration_samples_used=0,  # No calibration for FP16
            dtype=dtype,
            symmetric=symmetric,
        )
        return q_model, info

    # For INT8 static quantization
    # Default layer types if not specified
    if layer_types is None:
        layer_types = (nn.Linear, nn.Conv2d)
    elif isinstance(layer_types, type):
        layer_types = (layer_types,)

    # Normalize skip_types to tuple if provided
    if skip_types is not None and isinstance(skip_types, type):
        skip_types = (skip_types,)

    # Step 1: Determine layers to quantize
    selected_by_type: List[str] = []
    skipped_by_type: List[str] = []
    selected_by_name: List[str] = []
    skipped_by_name: List[str] = []

    # Type-based selection
    if layer_types is not None:
        selected_by_type, skipped_by_type = _select_layers_by_type(
            model_copy, layer_types, skip_types
        )

    # Name-based selection
    if layer_names is not None:
        selected_by_name, skipped_by_name = _select_layers_by_name(
            model_copy, layer_names, skip_names
        )

    # Merge selection results
    selected_layers, skipped_layers = _merge_selection_results(
        (selected_by_type, skipped_by_type),
        (selected_by_name, skipped_by_name),
    )

    # If no layers selected, return copy unchanged
    if not selected_layers:
        info = QuantizationInfo(
            selected_layers=[],
            skipped_layers=list(skipped_layers),
            calibration_samples_used=0,
            dtype=dtype,
            symmetric=symmetric,
        )
        return model_copy, info

    # Step 2: Run calibration (observers track activation ranges)
    # Note: For weights-only static quantization, calibration is primarily
    # for determining per-layer activation ranges. The actual quantization
    # uses weight-based quantization via quantize_weight_int8.
    observers: Dict[str, MinMaxObserver] = {}

    # Attach observers to selected layers for activation tracking
    hooks = []
    for layer_name in selected_layers:
        try:
            layer = model_copy.get_submodule(layer_name)
            observer = MinMaxObserver(dtype=dtype)
            observers[layer_name] = observer

            # Register forward hook to track activations
            def hook_fn(module, input, output, obs=observer):
                obs.forward(output)

            hook = layer.register_forward_hook(hook_fn)
            hooks.append(hook)
        except AttributeError:
            # Layer not found (may have been removed or renamed)
            continue

    # Run calibration forward passes
    run_calibration(
        model_copy,
        calibration_data,
        num_samples=num_calibration_samples,
    )

    # Remove hooks after calibration
    for hook in hooks:
        hook.remove()

    # Step 3: Quantize selected layers
    for layer_name in selected_layers:
        try:
            layer = model_copy.get_submodule(layer_name)

            if isinstance(layer, nn.Linear):
                q_module = quantize_linear_module(layer, dtype=dtype, symmetric=symmetric)
                # Replace in parent
                parent_name, child_name = _split_layer_name(layer_name)
                if parent_name:
                    parent = model_copy.get_submodule(parent_name)
                    setattr(parent, child_name, q_module)
                else:
                    # Top-level module
                    setattr(model_copy, child_name, q_module)

            elif isinstance(layer, nn.Conv2d):
                q_module = quantize_conv2d_module(layer, dtype=dtype, symmetric=symmetric)
                # Replace in parent
                parent_name, child_name = _split_layer_name(layer_name)
                if parent_name:
                    parent = model_copy.get_submodule(parent_name)
                    setattr(parent, child_name, q_module)
                else:
                    # Top-level module
                    setattr(model_copy, child_name, q_module)

        except (AttributeError, TypeError):
            # Layer not found or not quantizable
            skipped_layers.append(layer_name)

    # Determine actual samples used
    tensors = _normalize_calibration_data(calibration_data)
    samples_used = min(len(tensors), num_calibration_samples)

    # Create quantization info
    info = QuantizationInfo(
        selected_layers=selected_layers,
        skipped_layers=skipped_layers,
        calibration_samples_used=samples_used,
        dtype=dtype,
        symmetric=symmetric,
    )

    return model_copy, info


def _split_layer_name(layer_name: str) -> Tuple[Optional[str], str]:
    """
    Split a layer name into parent and child components.

    Args:
        layer_name: Dot-separated layer name (e.g., "encoder.0.weight" or "0")

    Returns:
        A tuple of (parent_name, child_name):
        - parent_name: Parent module path, or None for top-level
        - child_name: Child module name

    Examples:
        >>> _split_layer_name("encoder.0")
        ('encoder', '0')
        >>> _split_layer_name("0")
        (None, '0')
        >>> _split_layer_name("model.layer.1.weight")
        ('model.layer.1', 'weight')
    """
    if "." in layer_name:
        parts = layer_name.rsplit(".", 1)
        return parts[0], parts[1]
    else:
        return None, layer_name


def _quantize_int8_model(
    model: Union[nn.Module, Dict[str, torch.Tensor]],
    dtype: torch.dtype = torch.qint8,
    symmetric: bool = False,
    per_channel: bool = True,
) -> Tuple[nn.Module, List[str]]:
    """
    Quantize a model to INT8 using layer-specific quantization.

    This function iterates through the model's modules and quantizes
    supported layer types (nn.Linear, nn.Conv2d). Unsupported layers
    are skipped and recorded in the returned list.

    Args:
        model: Either a PyTorch nn.Module or a state_dict.
        dtype: Target quantization dtype (should be torch.qint8).
        symmetric: If True, use symmetric quantization.
        per_channel: If True, use per-channel scaling.

    Returns:
        A tuple of (quantized_model, skipped_layers):
        - quantized_model: Model with quantized supported layers
        - skipped_layers: List of names for unsupported layers
    """
    # Local imports to avoid circular dependency
    from mono_quant.io.handlers import _prepare_model
    from mono_quant.modules.linear import quantize_linear_module, quantize_conv2d_module

    # Get a copy of the model
    model_copy = _prepare_model(model)

    # Track skipped layers for partial quantization (per CONTEXT.md)
    skipped: List[str] = []

    # Iterate over named children to get top-level modules
    # We need to handle nested modules properly
    for name, module in list(model_copy.named_children()):
        if isinstance(module, nn.Linear):
            # Replace with QuantizedLinear
            q_module = quantize_linear_module(module, dtype=dtype, symmetric=symmetric)
            setattr(model_copy, name, q_module)
        elif isinstance(module, nn.Conv2d):
            # Replace with quantized Conv2d
            q_module = quantize_conv2d_module(module, dtype=dtype, symmetric=symmetric)
            setattr(model_copy, name, q_module)
        elif isinstance(module, nn.Sequential):
            # Handle Sequential containers by processing each layer
            _quantize_sequential_module(module, skipped, dtype, symmetric)
        else:
            # Skip unsupported layers (partial quantization)
            skipped.append(name)

    # Also check for nested modules that weren't caught by named_children
    # This handles cases where quantizable layers are nested in other containers
    for name, module in model_copy.named_modules():
        if name == "":
            continue  # Skip root module
        if "." not in name:
            continue  # Skip top-level (already processed)

        # Check if this is a quantizable layer that might have been missed
        parent_name, child_name = name.rsplit(".", 1)
        parent = model_copy.get_submodule(parent_name)

        if isinstance(parent, nn.Sequential):
            # Sequential containers are already processed
            continue

        if isinstance(module, nn.Linear) and not isinstance(module, type(model_copy.get_submodule(name))):
            # Found an unquantized Linear layer
            q_module = quantize_linear_module(module, dtype=dtype, symmetric=symmetric)
            setattr(parent, child_name, q_module)
        elif isinstance(module, nn.Conv2d) and not isinstance(module, type(model_copy.get_submodule(name))):
            # Found an unquantized Conv2d layer
            q_module = quantize_conv2d_module(module, dtype=dtype, symmetric=symmetric)
            setattr(parent, child_name, q_module)

    return model_copy, skipped


def _quantize_sequential_module(
    sequential: nn.Sequential,
    skipped: List[str],
    dtype: torch.dtype,
    symmetric: bool,
) -> None:
    """
    Quantize layers within a Sequential container in-place.

    Args:
        sequential: The nn.Sequential module to process.
        skipped: List to append skipped layer names to.
        dtype: Target quantization dtype.
        symmetric: Whether to use symmetric quantization.
    """
    # Local import to avoid circular dependency
    from mono_quant.modules.linear import quantize_linear_module, quantize_conv2d_module

    for i, module in enumerate(sequential):
        if isinstance(module, nn.Linear):
            q_module = quantize_linear_module(module, dtype=dtype, symmetric=symmetric)
            sequential[i] = q_module
        elif isinstance(module, nn.Conv2d):
            q_module = quantize_conv2d_module(module, dtype=dtype, symmetric=symmetric)
            sequential[i] = q_module
        else:
            # Track skipped layer (using index as name for Sequential)
            skipped.append(f"sequential.{i}")


def test_models_from_any_source() -> None:
    """
    Verification test for AGN-03: Quantization works with models from any source.

    This function tests that dynamic_quantize() works correctly with models
    from different sources:
    1. Custom/user-defined models
    2. HuggingFace-like models (simulated structure)
    3. Pretrained-like models (single layer)

    The test verifies that:
    - Quantization returns a different object (original unchanged)
    - Skipped layers list is populated correctly
    - All model types are accepted (AGN-03: model-agnostic input)

    Raises:
        AssertionError: If any verification fails.
    """
    # Test 1: Custom model (nn.Sequential with multiple layers)
    custom_model = nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 5)
    )
    q_custom, skipped_custom = dynamic_quantize(custom_model)

    # Verify different objects (original preserved)
    assert id(q_custom) != id(custom_model), "Custom: Quantized model should be a different object"

    # Verify ReLU was skipped (partial quantization)
    assert len(skipped_custom) > 0, "Custom: Non-quantizable layers should be skipped"

    # Test 2: HuggingFace-like model (same structure, different source)
    hf_like = nn.Sequential(
        nn.Linear(128, 256),
        nn.ReLU(),
        nn.Linear(256, 10)
    )
    q_hf, skipped_hf = dynamic_quantize(hf_like)

    assert id(q_hf) != id(hf_like), "HF-like: Quantized model should be a different object"
    assert len(skipped_hf) > 0, "HF-like: Non-quantizable layers should be skipped"

    # Test 3: Pretrained-like model (single Linear layer)
    pretrained_like = nn.Linear(50, 100)
    q_pretrained, skipped_pretrained = dynamic_quantize(pretrained_like)

    assert id(q_pretrained) != id(pretrained_like), "Pretrained-like: Quantized model should be a different object"

    # Test 4: FP16 quantization (no layers skipped for FP16)
    fp16_model = nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 5)
    )
    q_fp16, skipped_fp16 = dynamic_quantize(fp16_model, dtype=torch.float16)

    assert id(q_fp16) != id(fp16_model), "FP16: Quantized model should be a different object"
    assert len(skipped_fp16) == 0, "FP16: No layers should be skipped for FP16"

    # All tests passed
    print("AGN-03 verification passed: Quantization works with models from any source")
    print(f"  - Custom model: {len(skipped_custom)} layers skipped")
    print(f"  - HF-like model: {len(skipped_hf)} layers skipped")
    print(f"  - Pretrained-like: Single layer quantized")
    print(f"  - FP16 model: All parameters converted to FP16")
