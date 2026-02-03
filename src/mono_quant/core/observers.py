"""
MinMaxObserver for tracking activation ranges during calibration.

This module provides a custom MinMaxObserver implementation to avoid using
deprecated torch.ao.quantization APIs (scheduled for removal in PyTorch 2.10+).

The observer tracks min/max values from tensor activations during forward
passes and calculates scale and zero-point parameters for quantization.

Layer selection functions provide type-based and name-based filtering for
selective quantization of model layers.
"""

from typing import List, Tuple, Type, Union, Optional

import torch
import torch.nn as nn


class MinMaxObserver:
    """
    Observer that tracks min/max values for quantization parameter calculation.

    This is a custom implementation to avoid deprecated torch.ao.quantization APIs.
    It observes tensor activations during calibration and computes the scale and
    zero-point needed for affine quantization.

    The observer uses asymmetric affine quantization by default, which provides
    better accuracy for most neural network activations compared to symmetric.

    Args:
        dtype: Target quantization dtype. Default is torch.qint8.
               Only torch.qint8 is currently supported.

    Attributes:
        dtype: The target quantization dtype.
        min_val: Minimum value observed across all forward passes.
        max_val: Maximum value observed across all forward passes.

    Examples:
        >>> import torch
        >>> from mono_quant.core.observers import MinMaxObserver
        >>> obs = MinMaxObserver()
        >>> x = torch.randn(32, 64)
        >>> obs.forward(x)
        >>> scale, zp = obs.calculate_qparams()
        >>> assert scale.ndim == 0 and zp.ndim == 0
    """

    def __init__(self, dtype: torch.dtype = torch.qint8) -> None:
        self.dtype = dtype
        self.min_val = None
        self.max_val = None

    def forward(self, x: torch.Tensor) -> None:
        """
        Update min/max values from input tensor.

        This method updates the observer's internal min/max state by comparing
        the current tensor's minimum and maximum values with previously observed
        values. It handles the first call when min_val/max_val are None.

        Args:
            x: Input tensor to observe. Can be any shape; only global min/max
               are tracked.

        Examples:
            >>> obs = MinMaxObserver()
            >>> obs.forward(torch.tensor([1.0, 2.0, 3.0]))
            >>> assert obs.min_val == 1.0
            >>> assert obs.max_val == 3.0
        """
        x_min = x.amin().item()
        x_max = x.amax().item()

        if self.min_val is None:
            # First observation
            self.min_val = x_min
            self.max_val = x_max
        else:
            # Update min/max
            self.min_val = min(self.min_val, x_min)
            self.max_val = max(self.max_val, x_max)

    def calculate_qparams(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate scale and zero-point from observed min/max values.

        This computes the quantization parameters using asymmetric affine
        quantization:
            scale = (max_val - min_val) / (qmax - qmin)
            zero_point = qmin - (min_val / scale)

        The scale is clamped to a minimum of 1e-8 to avoid division by zero
        when the observed range is zero (constant tensor).

        Returns:
            A tuple of (scale, zero_point):
            - scale: 0-dim tensor with the quantization scale factor
            - zero_point: 0-dim tensor with int32 dtype containing the
              zero-point offset

        Raises:
            RuntimeError: If no data has been observed (min_val is None).

        Examples:
            >>> obs = MinMaxObserver()
            >>> obs.forward(torch.tensor([-1.0, 1.0]))
            >>> scale, zp = obs.calculate_qparams()
            >>> assert scale > 0
            >>> assert isinstance(zp.item(), int)
        """
        if self.min_val is None:
            raise RuntimeError(
                "No data observed. Call forward() with calibration data "
                "before calculating quantization parameters."
            )

        # int8 range: [-128, 127]
        qmin, qmax = -128, 127

        # Calculate scale
        range_val = self.max_val - self.min_val
        q_range = qmax - qmin
        scale = range_val / q_range

        # Clamp scale to avoid division by zero
        scale = max(scale, 1e-8)

        # Calculate zero-point and round to integer
        zero_point = qmin - (self.min_val / scale)
        zero_point = int(round(zero_point))

        return torch.tensor(scale), torch.tensor(zero_point, dtype=torch.int32)

    def reset(self) -> None:
        """
        Reset the observer to its initial state.

        This clears all observed min/max values, allowing the observer
        to be reused for a new calibration session.

        Examples:
            >>> obs = MinMaxObserver()
            >>> obs.forward(torch.randn(10))
            >>> obs.reset()
            >>> assert obs.min_val is None
            >>> assert obs.max_val is None
        """
        self.min_val = None
        self.max_val = None


# Type alias for layer types used in selection
LayerTypes = Union[Type[nn.Module], Tuple[Type[nn.Module], ...]]


def _select_layers_by_type(
    model: nn.Module,
    layer_types: LayerTypes,
    skip_types: Optional[LayerTypes] = None,
) -> Tuple[List[str], List[str]]:
    """
    Select model layers by type for quantization.

    This function iterates through a model's modules and selects layers
    that match the specified types. Layers matching skip_types are
    excluded from selection.

    Args:
        model: PyTorch model to scan for quantizable layers.
        layer_types: Layer type(s) to select for quantization. Can be a
                     single nn.Module type or a tuple of types.
        skip_types: Optional layer type(s) to exclude from quantization.
                    Can be a single nn.Module type or a tuple of types.

    Returns:
        A tuple of (selected_layers, skipped_layers):
        - selected_layers: List of module names matching layer_types
        - skipped_layers: List of module names not selected for quantization

    Examples:
        >>> import torch.nn as nn
        >>> from mono_quant.core.observers import _select_layers_by_type
        >>> model = nn.Sequential(
        ...     nn.Linear(10, 20),
        ...     nn.ReLU(),
        ...     nn.Linear(20, 5)
        ... )
        >>> selected, skipped = _select_layers_by_type(model, nn.Linear)
        >>> assert "0" in selected and "2" in selected  # Linear layers
        >>> assert "1" in skipped  # ReLU skipped

        With skip_types:
        >>> model = nn.Sequential(
        ...     nn.Linear(10, 20),
        ...     nn.BatchNorm1d(20),
        ...     nn.Linear(20, 5)
        ... )
        >>> selected, skipped = _select_layers_by_type(
        ...     model, (nn.Linear, nn.BatchNorm1d), skip_types=nn.BatchNorm1d
        ... )
        >>> assert "0" in selected and "2" in selected  # Linear layers
        >>> assert "1" in skipped  # BatchNorm skipped
    """
    # Normalize layer_types to tuple if not already
    if isinstance(layer_types, type):
        layer_types = (layer_types,)
    else:
        # Ensure it's a tuple
        layer_types = tuple(layer_types) if layer_types is not None else ()

    # Normalize skip_types to tuple if provided
    if skip_types is None:
        skip_types = ()
    elif isinstance(skip_types, type):
        skip_types = (skip_types,)
    else:
        skip_types = tuple(skip_types)

    selected: List[str] = []
    skipped: List[str] = []

    for name, module in model.named_modules():
        # Skip the root module (empty name)
        if name == "":
            continue

        # Check if module should be skipped
        if skip_types and isinstance(module, skip_types):
            skipped.append(name)
        # Check if module matches selection criteria
        elif layer_types and isinstance(module, layer_types):
            selected.append(name)
        else:
            # Non-matching layers are skipped
            skipped.append(name)

    return selected, skipped


def _select_layers_by_name(
    model: nn.Module,
    layer_names: Optional[List[str]] = None,
    skip_names: Optional[List[str]] = None,
) -> Tuple[List[str], List[str]]:
    """
    Select model layers by exact name matching.

    This function selects layers based on exact module names from the model's
    named_modules() hierarchy. Names are dot-separated paths (e.g., "encoder.0"
    for the first layer of a Sequential named "encoder").

    Args:
        model: PyTorch model to scan for quantizable layers.
        layer_names: Optional list of exact layer names to quantize.
                     If None, no layers are selected by name.
        skip_names: Optional list of layer names to exclude from quantization.

    Returns:
        A tuple of (selected_layers, skipped_layers):
        - selected_layers: Sorted list of layer names selected for quantization
        - skipped_layers: List of layer names skipped (skip_names or all if no layer_names)

    Examples:
        >>> import torch.nn as nn
        >>> from mono_quant.core.observers import _select_layers_by_name
        >>> model = nn.Sequential(
        ...     nn.Linear(10, 20),
        ...     nn.ReLU(),
        ...     nn.Linear(20, 5)
        ... )
        >>> # Select specific layers by name
        >>> selected, skipped = _select_layers_by_name(model, layer_names=["0", "2"])
        >>> assert selected == ["0", "2"]
        >>> assert "1" in skipped

        With skip_names:
        >>> selected, skipped = _select_layers_by_name(
        ...     model, layer_names=["0", "1", "2"], skip_names=["1"]
        ... )
        >>> assert selected == ["0", "2"]  # 1 is skipped
    """
    selected: set = set()
    skip_set: set = set(skip_names or [])

    if layer_names:
        # Build set of valid layer names from model
        valid_names = {name for name, _ in model.named_modules() if name != ""}

        # Select only valid names that aren't in skip_set
        for name in layer_names:
            if name in valid_names and name not in skip_set:
                selected.add(name)

    # Return sorted selected list and skip_set as list
    return sorted(list(selected)), list(skip_set)


def _merge_selection_results(
    *selection_results: Tuple[List[str], List[str]],
) -> Tuple[List[str], List[str]]:
    """
    Merge multiple layer selection results into a single selection.

    This function combines results from multiple selection criteria
    (e.g., type-based and name-based selection). Layers that appear
    in any selected list are included in the final selection.

    Args:
        *selection_results: Variable number of (selected, skipped) tuples
                           from layer selection functions.

    Returns:
        A tuple of (selected_layers, skipped_layers):
        - selected_layers: Sorted union of all selected layer names
        - skipped_layers: Sorted union of skipped layers, excluding any
                          layers that appear in selected_layers

    Examples:
        >>> from mono_quant.core.observers import _merge_selection_results
        >>> # First selection: type-based
        >>> result1 = (["0", "2"], ["1", "3"])
        >>> # Second selection: name-based
        >>> result2 = (["1"], [])
        >>> # Merge both selections
        >>> selected, skipped = _merge_selection_results(result1, result2)
        >>> assert set(selected) == {"0", "1", "2"}
        >>> assert "3" in skipped
        >>> assert "1" not in skipped  # In selected now
    """
    all_selected: set = set()
    all_skipped: set = set()

    for selected, skipped in selection_results:
        all_selected.update(selected)
        all_skipped.update(skipped)

    # Remove any skipped layers that are actually selected
    all_skipped -= all_selected

    return sorted(list(all_selected)), sorted(list(all_skipped))
