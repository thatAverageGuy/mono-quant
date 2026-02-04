"""
Calibration data handling utilities.

This module provides utilities for normalizing different calibration data formats
into a consistent tensor list representation. It handles both List[torch.Tensor]
and PyTorch DataLoader inputs, with support for the common (input, target) batching
pattern used in supervised learning.
"""

from typing import List, Union

import torch
from torch.utils.data import DataLoader

# Type alias for calibration data inputs
CalibrationData = Union[List[torch.Tensor], DataLoader]


def _normalize_calibration_data(data: CalibrationData) -> List[torch.Tensor]:
    """
    Normalize calibration data to a list of tensors.

    This function handles both List[torch.Tensor] and DataLoader inputs,
    converting them to a consistent list format for calibration processing.

    For DataLoader inputs, the function extracts input tensors from batches.
    Batches are commonly structured as (input, target) tuples in supervised
    learning, so we take the first element (batch[0]) as the input tensor.

    Args:
        data: Calibration data in one of the following formats:
            - List[torch.Tensor]: Direct list of input tensors
            - DataLoader: PyTorch DataLoader yielding batches

    Returns:
        A list of torch.Tensor containing the input data for calibration.

    Examples:
        >>> from mono_quant.calibration.data import _normalize_calibration_data
        >>> from torch.utils.data import DataLoader, TensorDataset
        >>> # List of tensors
        >>> data_list = [torch.randn(10, 20) for _ in range(5)]
        >>> result = _normalize_calibration_data(data_list)
        >>> assert len(result) == 5
        >>> # DataLoader (groups tensors into batches)
        >>> loader = DataLoader(data_list, batch_size=2)
        >>> result = _normalize_calibration_data(loader)
        >>> assert len(result) == 3  # 3 batches: 2+2+1 samples per batch
    """
    if isinstance(data, DataLoader):
        # Extract tensors from DataLoader batches
        tensors = []
        for batch in data:
            if isinstance(batch, (tuple, list)):
                # Assume (input, target) format - take input
                tensors.append(batch[0])
            else:
                # Batch is already the tensor
                tensors.append(batch)
        return tensors
    else:
        # Already a list of tensors
        return data


def _limit_samples(tensors: List[torch.Tensor], num_samples: int) -> List[torch.Tensor]:
    """
    Limit the number of calibration samples.

    This function truncates the tensor list to a maximum number of samples.
    It handles the case where the input list has fewer samples than requested.

    Args:
        tensors: List of calibration tensors.
        num_samples: Maximum number of samples to return.

    Returns:
        A truncated list of at most num_samples tensors.

    Examples:
        >>> from mono_quant.calibration.data import _limit_samples
        >>> data = [torch.randn(10, 20) for _ in range(10)]
        >>> limited = _limit_samples(data, 5)
        >>> assert len(limited) == 5
        >>> # Handles case with fewer samples
        >>> limited = _limit_samples(data, 20)
        >>> assert len(limited) == 10
    """
    return tensors[:num_samples]
