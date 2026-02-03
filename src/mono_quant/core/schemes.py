"""
Quantization scheme classes for symmetric and affine (asymmetric) quantization.

These schemes provide the mathematical foundation for calculating scale and zero-point
values used in quantization. They are pure functions with no PyTorch module dependencies,
making them highly testable.
"""

from abc import ABC, abstractmethod
from typing import Optional, Tuple

import torch


class QuantizationScheme(ABC):
    """
    Abstract base class for quantization schemes.

    A quantization scheme defines how to calculate the scale and zero-point
    parameters used to map floating-point values to quantized integers.
    """

    @abstractmethod
    def calculate(
        self, tensor: torch.Tensor, axis: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate scale and zero-point for quantizing the given tensor.

        Args:
            tensor: Input tensor to calculate quantization parameters for.
            axis: If None, calculate per-tensor (single scale/zp for entire tensor).
                  If an int, calculate per-channel along the specified axis.

        Returns:
            Tuple of (scale, zero_point) tensors.
            - scale: The scaling factor applied during quantization.
            - zero_point: The offset applied during quantization.
        """
        pass


class SymmetricScheme(QuantizationScheme):
    """
    Symmetric quantization scheme where zero_point is always zero.

    In symmetric quantization, the quantized range is centered around zero,
    which simplifies calculations but may be less efficient for asymmetric
    weight distributions.

    Formula:
        scale = max_abs(x) / qmax
        zero_point = 0

    Where qmax is the maximum quantized integer value (e.g., 127 for int8).
    """

    def __init__(self, qmin: int = -128, qmax: int = 127) -> None:
        """
        Initialize the symmetric quantization scheme.

        Args:
            qmin: Minimum quantized value (default: -128 for int8).
            qmax: Maximum quantized value (default: 127 for int8).
        """
        self.qmin = qmin
        self.qmax = qmax

    def calculate(
        self, tensor: torch.Tensor, axis: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate scale and zero-point for symmetric quantization.

        Args:
            tensor: Input tensor to calculate parameters for.
            axis: If None, per-tensor (single scale). If int, per-channel along
                  that axis (one scale per element along axis).

        Returns:
            Tuple of (scale, zero_point). Zero-point is always zero for symmetric.
        """
        if axis is None:
            # Per-tensor: single scale for entire tensor
            max_abs = tensor.abs().max()
        else:
            # Per-channel: reduce over all dimensions EXCEPT the specified axis
            # For axis=0 on shape (64, 128), we get shape (64,) - one per output channel
            reduce_dims = tuple(i for i in range(tensor.dim()) if i != axis)
            if reduce_dims:
                max_abs = tensor.abs().amax(dim=reduce_dims, keepdim=False)
            else:
                # Single dimension tensor - just take abs
                max_abs = tensor.abs()

        # Calculate scale: max_abs / qmax
        # Use qmax (not qmin) because symmetric is centered on zero
        scale = max_abs / self.qmax

        # Clamp scale to avoid division by zero
        scale = torch.clamp(scale, min=1e-8)

        # Zero-point is always zero for symmetric quantization
        zero_point = torch.zeros_like(scale)

        return scale, zero_point


class AsymmetricScheme(QuantizationScheme):
    """
    Asymmetric (affine) quantization scheme with configurable zero-point.

    Asymmetric quantization allows the quantized range to shift to better
    fit asymmetric data distributions. This can improve accuracy for
    weights that are not centered around zero.

    Formula:
        scale = (max_val - min_val) / (qmax - qmin)
        zero_point = qmin - (min_val / scale)
    """

    def __init__(self, qmin: int = -128, qmax: int = 127) -> None:
        """
        Initialize the asymmetric quantization scheme.

        Args:
            qmin: Minimum quantized value (default: -128 for int8).
            qmax: Maximum quantized value (default: 127 for int8).
        """
        self.qmin = qmin
        self.qmax = qmax

    def calculate(
        self, tensor: torch.Tensor, axis: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate scale and zero-point for asymmetric quantization.

        Args:
            tensor: Input tensor to calculate parameters for.
            axis: If None, per-tensor. If int, per-channel along that axis.

        Returns:
            Tuple of (scale, zero_point).
        """
        if axis is None:
            # Per-tensor: single min/max for entire tensor
            min_val = tensor.amin()
            max_val = tensor.amax()
        else:
            # Per-channel: reduce over all dimensions EXCEPT the specified axis
            # For axis=0 on shape (64, 128), we get shape (64,) - one per output channel
            reduce_dims = tuple(i for i in range(tensor.dim()) if i != axis)
            if reduce_dims:
                min_val = tensor.amin(dim=reduce_dims, keepdim=False)
                max_val = tensor.amax(dim=reduce_dims, keepdim=False)
            else:
                # Single dimension tensor
                min_val = tensor
                max_val = tensor

        # Calculate range of values
        range_val = max_val - min_val
        q_range = self.qmax - self.qmin

        # Calculate scale: (max - min) / (qmax - qmin)
        scale = range_val / q_range

        # Clamp scale to avoid division by zero
        scale = torch.clamp(scale, min=1e-8)

        # Calculate zero-point: qmin - (min_val / scale)
        zero_point = self.qmin - (min_val / scale)

        # Convert zero-point to integer (quantized zero-point must be integer)
        zero_point = zero_point.to(torch.int32)

        return scale, zero_point
