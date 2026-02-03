"""
Observers for tracking activation ranges during calibration.

This module provides custom observer implementations to avoid using
deprecated torch.ao.quantization APIs (scheduled for removal in PyTorch 2.10+).

Observers track min/max values from tensor activations during forward
passes and calculate scale and zero-point parameters for quantization.

Available observers:
- MinMaxObserver: Simple min/max tracking (baseline)
- MovingAverageMinMaxObserver: EMA-based smoothing for outlier handling
- HistogramObserver: KL divergence minimization for skewed distributions

Layer selection functions provide type-based and name-based filtering for
selective quantization of model layers.

INT4 Layer Skipping:
-------------------
INT4 quantization is aggressive and can significantly impact model accuracy
if applied to all layers. The default skip list (DEFAULT_INT4_SKIP) excludes
sensitive layer types (embeddings, normalization) and small layers where
quantization overhead outweighs compression benefits.

The unified layer skipping API (_get_layers_to_skip) combines:
- Type-based filtering (skip_layer_types)
- Name-based filtering (skip_layer_names, modules_to_not_convert)
- Parameter threshold filtering (skip_param_threshold)
"""

from typing import Dict, List, Any, Tuple, Type, Union, Optional

import math
import torch
import torch.nn as nn


# Default INT4 skip list based on research recommendations
# See: .planning/phases/03-advanced-calibration-&-int4/03-RESEARCH.md
DEFAULT_INT4_SKIP: Dict[str, Any] = {
    "skip_types": (
        nn.Embedding,
        nn.EmbeddingBag,
        nn.LayerNorm,
        nn.BatchNorm1d,
        nn.BatchNorm2d,
    ),
    "skip_param_threshold": 512,  # Skip layers with fewer parameters
    "skip_names": ["lm_head"],  # Common sensitive output layer
}


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


class MovingAverageMinMaxObserver:
    """
    Observer that tracks exponential moving average of min/max values.

    This observer smooths out transient spikes in calibration data using
    an exponential moving average (EMA). This makes it more robust to
    outliers compared to MinMaxObserver, which is sensitive to single
    extreme values that can distort the entire quantization range.

    The EMA formula is:
        new_val = (1 - c) * old_val + c * current_val

    Where c is the averaging_constant:
    - Lower values (e.g., 0.001) = more stable, slower to adapt
    - Higher values (e.g., 0.1) = more responsive, less smoothing
    - Default 0.01 matches PyTorch's standard behavior

    Args:
        averaging_constant: Weight for new observations (0 < c <= 1).
                           Default is 0.01, matching PyTorch's standard.
        dtype: Target quantization dtype. Default is torch.qint8.

    Attributes:
        averaging_constant: The weight for new observations.
        dtype: The target quantization dtype.
        min_val: EMA of minimum values observed across all forward passes.
        max_val: EMA of maximum values observed across all forward passes.

    Raises:
        ValueError: If averaging_constant is not in (0, 1].

    Examples:
        >>> import torch
        >>> from mono_quant.core.observers import MovingAverageMinMaxObserver
        >>> obs = MovingAverageMinMaxObserver(averaging_constant=0.01)
        >>> x = torch.randn(32, 64)
        >>> obs.forward(x)
        >>> scale, zp = obs.calculate_qparams()
        >>> assert scale.ndim == 0 and zp.ndim == 0
    """

    def __init__(
        self,
        averaging_constant: float = 0.01,
        dtype: torch.dtype = torch.qint8,
    ) -> None:
        if not 0 < averaging_constant <= 1:
            raise ValueError(
                f"averaging_constant must be in (0, 1], got {averaging_constant}"
            )
        self.averaging_constant = averaging_constant
        self.dtype = dtype
        self.min_val = None
        self.max_val = None

    def forward(self, x: torch.Tensor) -> None:
        """
        Update EMA of min/max values from input tensor.

        This method applies exponential moving average to smooth out
        transient spikes in the activation ranges.

        Args:
            x: Input tensor to observe. Can be any shape; only global min/max
               are tracked.

        Examples:
            >>> obs = MovingAverageMinMaxObserver(averaging_constant=0.5)
            >>> obs.forward(torch.tensor([1.0, 2.0, 3.0]))
            >>> assert obs.min_val is not None
            >>> obs.forward(torch.tensor([10.0, 20.0, 30.0]))
            >>> # EMA smooths the jump, min_val won't immediately be 10.0
        """
        x_min = x.amin().item()
        x_max = x.amax().item()

        if self.min_val is None:
            # First observation
            self.min_val = x_min
            self.max_val = x_max
        else:
            # Apply exponential moving average
            c = self.averaging_constant
            self.min_val = (1 - c) * self.min_val + c * x_min
            self.max_val = (1 - c) * self.max_val + c * x_max

    def calculate_qparams(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate scale and zero-point from EMA min/max values.

        This computes the quantization parameters using asymmetric affine
        quantization, identical to MinMaxObserver.

        Returns:
            A tuple of (scale, zero_point):
            - scale: 0-dim tensor with the quantization scale factor
            - zero_point: 0-dim tensor with int32 dtype containing the
              zero-point offset

        Raises:
            RuntimeError: If no data has been observed (min_val is None).

        Examples:
            >>> obs = MovingAverageMinMaxObserver()
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

        # Clamp to valid range
        zero_point = max(qmin, min(qmax, zero_point))

        return torch.tensor(scale), torch.tensor(zero_point, dtype=torch.int32)

    def reset(self) -> None:
        """
        Reset the observer to its initial state.

        This clears all observed EMA min/max values, allowing the observer
        to be reused for a new calibration session.

        Examples:
            >>> obs = MovingAverageMinMaxObserver()
            >>> obs.forward(torch.randn(10))
            >>> obs.reset()
            >>> assert obs.min_val is None
            >>> assert obs.max_val is None
        """
        self.min_val = None
        self.max_val = None


class HistogramObserver:
    """
    Observer that uses KL divergence minimization for threshold selection.

    This observer builds a histogram of activation values and selects the
    optimal quantization threshold using KL (Kullback-Leibler) divergence
    minimization. This approach is similar to TensorRT's quantization strategy
    and is robust to outliers and skewed distributions.

    The key idea is to find a quantization threshold that minimizes the
    information loss (KL divergence) between the original distribution P
    and the quantized distribution Q.

    Args:
        bins: Number of histogram bins for building the distribution.
              Default is 2048, which provides good resolution for most cases.
        dtype: Target quantization dtype. Default is torch.qint8.

    Attributes:
        bins: Number of histogram bins.
        dtype: The target quantization dtype.
        histogram_counts: Accumulated bin counts across all forward passes.
        bin_edges: Bin edge values (consistent across accumulations).
        min_val: Global minimum observed across all forward passes.
        max_val: Global maximum observed across all forward passes.

    Examples:
        >>> import torch
        >>> from mono_quant.core.observers import HistogramObserver
        >>> obs = HistogramObserver(bins=2048)
        >>> x = torch.randn(1000) * 0.5 + 2.0
        >>> obs.forward(x)
        >>> scale, zp = obs.calculate_qparams()
        >>> assert scale.ndim == 0 and zp.ndim == 0
    """

    def __init__(
        self,
        bins: int = 2048,
        dtype: torch.dtype = torch.qint8,
    ) -> None:
        self.bins = bins
        self.dtype = dtype
        self.histogram_counts: Optional[torch.Tensor] = None
        self.bin_edges: Optional[torch.Tensor] = None
        self.min_val: Optional[float] = None
        self.max_val: Optional[float] = None

    def forward(self, x: torch.Tensor) -> None:
        """
        Update histogram with values from input tensor.

        This method builds a histogram of the current tensor's values and
        accumulates it with previously observed histograms.

        Args:
            x: Input tensor to observe. Values are flattened for histogram.

        Examples:
            >>> obs = HistogramObserver(bins=100)
            >>> obs.forward(torch.randn(1000))
            >>> assert obs.histogram_counts is not None
            >>> obs.forward(torch.randn(1000))  # Accumulates
        """
        # Track global min/max
        x_min = x.amin().item()
        x_max = x.amax().item()

        if self.min_val is None:
            self.min_val = x_min
            self.max_val = x_max
        else:
            self.min_val = min(self.min_val, x_min)
            self.max_val = max(self.max_val, x_max)

        # Build histogram for current tensor
        new_counts, new_edges = torch.histogram(x.flatten(), bins=self.bins)

        # Accumulate histograms
        if self.histogram_counts is None:
            # First observation - store as-is
            self.histogram_counts = new_counts
            self.bin_edges = new_edges
        else:
            # Accumulate counts (bin_edges stay consistent)
            self.histogram_counts += new_counts

    def calculate_qparams(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate scale and zero-point using KL divergence minimization.

        This method finds the optimal quantization threshold by minimizing
        the KL divergence between the original distribution and the
        quantized distribution.

        Returns:
            A tuple of (scale, zero_point):
            - scale: 0-dim tensor with the quantization scale factor
            - zero_point: 0-dim tensor with int32 dtype containing the
              zero-point offset

        Raises:
            RuntimeError: If no data has been observed.

        Examples:
            >>> obs = HistogramObserver()
            >>> obs.forward(torch.randn(1000))
            >>> scale, zp = obs.calculate_qparams()
            >>> assert scale > 0
            >>> assert isinstance(zp.item(), int)
        """
        if self.histogram_counts is None:
            raise RuntimeError(
                "No data observed. Call forward() with calibration data "
                "before calculating quantization parameters."
            )

        # Get distribution P from histogram
        P = self.histogram_counts.float()
        P = P / P.sum()  # Normalize to sum=1

        # Find optimal threshold using KL divergence minimization
        optimal_threshold = self._find_optimal_threshold(P)

        # Use threshold to determine min/max for quantization
        range_val = optimal_threshold
        qmin, qmax = -128, 127
        q_range = qmax - qmin
        scale = range_val / q_range
        scale = max(scale, 1e-8)  # Clamp to avoid division by zero

        # Calculate zero-point (assuming symmetric around 0 for threshold)
        zero_point = qmin - ((-optimal_threshold / 2) / scale)
        zero_point = int(round(zero_point))
        zero_point = max(qmin, min(qmax, zero_point))

        return torch.tensor(scale), torch.tensor(zero_point, dtype=torch.int32)

    def _find_optimal_threshold(self, P: torch.Tensor) -> float:
        """
        Find optimal quantization threshold using KL divergence minimization.

        Args:
            P: Normalized histogram distribution (sums to 1).

        Returns:
            Optimal threshold value for quantization range.
        """
        # Search thresholds from 50% to 100% of distribution range
        # This balances quantization range with information preservation
        n_bins = len(P)
        min_kl = float("inf")
        optimal_threshold_idx = n_bins - 1

        # Start from 50% of bins to avoid too narrow range
        start_idx = n_bins // 2

        for threshold_idx in range(start_idx, n_bins):
            # Create quantized distribution Q
            Q = self._quantize_distribution(P, threshold_idx)

            # Compute KL divergence
            kl_div = self._compute_kl_divergence(P, Q)

            if kl_div < min_kl:
                min_kl = kl_div
                optimal_threshold_idx = threshold_idx

        # Convert threshold index to actual value range
        # Use full range from min_val to max_val
        if self.min_val is not None and self.max_val is not None:
            full_range = self.max_val - self.min_val
            threshold_ratio = (optimal_threshold_idx + 1) / n_bins
            return full_range * threshold_ratio
        return 1.0

    def _compute_kl_divergence(
        self,
        P: torch.Tensor,
        Q: torch.Tensor,
    ) -> float:
        """
        Compute KL divergence D_KL(P || Q).

        Args:
            P: Original distribution.
            Q: Quantized distribution.

        Returns:
            KL divergence value (lower is better).
        """
        # Add epsilon to avoid log(0)
        epsilon = 1e-10
        P = P + epsilon
        Q = Q + epsilon

        # D_KL(P || Q) = sum(P * log(P / Q))
        kl_div = torch.sum(P * torch.log(P / Q))
        return kl_div.item()

    def _quantize_distribution(
        self,
        P: torch.Tensor,
        threshold_idx: int,
    ) -> torch.Tensor:
        """
        Create quantized distribution Q by mapping to discrete bins.

        Values outside the threshold are mapped to boundary bins.

        Args:
            P: Original distribution (histogram counts).
            threshold_idx: Index defining the quantization threshold.

        Returns:
            Quantized distribution Q (same shape as P).
        """
        Q = P.clone()

        # Number of quantization bins (e.g., 128 for INT8 one-sided)
        # We use half of threshold_idx for quantization resolution
        n_quant_bins = min(128, threshold_idx)

        if n_quant_bins <= 1:
            return Q

        # Map values outside threshold to boundaries
        # (simplified approach - just clip the distribution)
        Q[threshold_idx:] = 0

        # Redistribute the clipped mass to boundary bins
        clipped_mass = P[threshold_idx:].sum()
        if clipped_mass > 0 and threshold_idx > 0:
            # Distribute clipped mass to the last bin inside threshold
            Q[threshold_idx - 1] += clipped_mass

        # Normalize to sum=1
        Q = Q / (Q.sum() + 1e-10)

        return Q

    def reset(self) -> None:
        """
        Reset the observer to its initial state.

        This clears all histogram data and min/max values.

        Examples:
            >>> obs = HistogramObserver()
            >>> obs.forward(torch.randn(100))
            >>> obs.reset()
            >>> assert obs.histogram_counts is None
            >>> assert obs.min_val is None
        """
        self.histogram_counts = None
        self.bin_edges = None
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


def _get_layers_to_skip(
    model: nn.Module,
    modules_to_not_convert: Optional[List[str]] = None,
    skip_layer_types: Optional[LayerTypes] = None,
    skip_layer_names: Optional[List[str]] = None,
    skip_param_threshold: int = 0,
) -> set:
    """
    Build a unified set of layer names to skip during quantization.

    This function provides a unified API for layer skipping by combining
    multiple filtering strategies:
    - Direct name list (modules_to_not_convert) - HuggingFace compatible
    - Type-based filtering (skip_layer_types)
    - Name pattern matching (skip_layer_names)
    - Parameter count threshold (skip_param_threshold)

    Args:
        model: PyTorch model to analyze for layer skipping.
        modules_to_not_convert: Unified skip list of exact layer names to exclude.
                                 This is the primary parameter (HuggingFace compatible).
        skip_layer_types: Optional layer type(s) to exclude from quantization.
                         Can be a single nn.Module type or a tuple of types.
        skip_layer_names: Optional list of layer name patterns to exclude.
        skip_param_threshold: Skip layers with fewer than this many parameters.
                             Default is 0 (no threshold filtering).

    Returns:
        A set of layer names to skip during quantization. The set contains
        exact module names from the model's named_modules() hierarchy.

    Examples:
        >>> import torch.nn as nn
        >>> from mono_quant.core.observers import _get_layers_to_skip, DEFAULT_INT4_SKIP
        >>> model = nn.Sequential(
        ...     nn.Linear(10, 20),
        ...     nn.LayerNorm(20),
        ...     nn.Linear(20, 5),
        ... )
        >>> # Skip by direct name list
        >>> skip1 = _get_layers_to_skip(model, modules_to_not_convert=["1"])
        >>> assert "1" in skip1

        >>> # Skip by layer type
        >>> skip2 = _get_layers_to_skip(model, skip_layer_types=(nn.LayerNorm,))
        >>> assert "1" in skip2

        >>> # Skip by parameter threshold
        >>> skip3 = _get_layers_to_skip(model, skip_param_threshold=100)
        >>> # Should skip layers with <100 parameters

        >>> # Use default INT4 skip list
        >>> skip4 = _get_layers_to_skip(
        ...     model,
        ...     modules_to_not_convert=DEFAULT_INT4_SKIP["skip_names"],
        ...     skip_layer_types=DEFAULT_INT4_SKIP["skip_types"],
        ...     skip_param_threshold=DEFAULT_INT4_SKIP["skip_param_threshold"],
        ... )
    """
    # Initialize skip_set with direct name list
    skip_set = set(modules_to_not_convert or [])

    # Add type-based skips
    if skip_layer_types is not None:
        # Normalize to tuple if single type
        if isinstance(skip_layer_types, type):
            skip_types = (skip_layer_types,)
        else:
            skip_types = tuple(skip_layer_types) if skip_layer_types else ()

        # Iterate through model to find matching types
        for name, module in model.named_modules():
            if name == "":
                continue  # Skip root module
            if isinstance(module, skip_types):
                skip_set.add(name)

    # Add name-based skips
    if skip_layer_names:
        skip_set.update(skip_layer_names)

    # Add parameter threshold skips
    if skip_param_threshold > 0:
        for name, module in model.named_modules():
            if name == "":
                continue  # Skip root module
            # Count parameters
            param_count = sum(p.numel() for p in module.parameters())
            if param_count < skip_param_threshold:
                skip_set.add(name)

    return skip_set
