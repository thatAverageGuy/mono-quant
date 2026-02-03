"""
MinMaxObserver for tracking activation ranges during calibration.

This module provides a custom MinMaxObserver implementation to avoid using
deprecated torch.ao.quantization APIs (scheduled for removal in PyTorch 2.10+).

The observer tracks min/max values from tensor activations during forward
passes and calculates scale and zero-point parameters for quantization.
"""

from typing import Tuple

import torch


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
