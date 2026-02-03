"""
Calibration runner for static quantization.

This module provides the run_calibration() function that executes forward passes
through a model with calibration data. This allows observers (like MinMaxObserver)
to track activation ranges for computing quantization parameters.

The runner includes optional progress reporting for large datasets (auto-detects
when tqdm is available and dataset exceeds threshold).
"""

from typing import Dict, List, Optional, Union

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from mono_quant.core.observers import MinMaxObserver
from .data import _normalize_calibration_data, _limit_samples, CalibrationData


def run_calibration(
    model: nn.Module,
    calibration_data: CalibrationData,
    observers: Optional[Dict[str, MinMaxObserver]] = None,
    num_samples: int = 150,
    show_progress: Optional[bool] = None,
) -> Dict[str, MinMaxObserver]:
    """
    Run calibration forward passes through a model.

    This function executes forward passes using calibration data, allowing
    observers attached to the model to track activation ranges. The model
    is set to eval mode and gradients are disabled during calibration.

    Progress bar is automatically shown for large datasets (default threshold
    is 50 samples) when tqdm is available. Can be controlled with show_progress.

    Args:
        model: PyTorch model to calibrate. Should have MinMaxObserver instances
               attached to layers where activation quantization is needed.
        calibration_data: Calibration data in one of the following formats:
            - List[torch.Tensor]: Direct list of input tensors
            - DataLoader: PyTorch DataLoader yielding batches
        observers: Optional dictionary of observer name to MinMaxObserver.
                   If None, returns empty dict (observers assumed attached by
                   caller via hook registration or custom layers).
        num_samples: Maximum number of samples to use for calibration.
                     Default is 150, based on research recommending 100-200
                     baseline samples for static quantization.
        show_progress: Whether to show progress bar. If None, auto-detects
                      based on sample count (True for >50 samples).

    Returns:
        Dictionary of observer names to MinMaxObserver instances with
        observed min/max values. The observers can be used to compute
        quantization parameters via calculate_qparams().

    Examples:
        >>> from mono_quant.calibration import run_calibration
        >>> import torch.nn as nn
        >>> model = nn.Sequential(nn.Linear(10, 20), nn.ReLU())
        >>> data = [torch.randn(4, 10) for _ in range(100)]
        >>> observers = run_calibration(model, data, num_samples=50)
        >>> # Model has run inference; observers have tracked activations
    """
    # Normalize calibration data to tensor list
    tensors = _normalize_calibration_data(calibration_data)

    # Limit to requested number of samples
    tensors = _limit_samples(tensors, num_samples)

    # Auto-detect progress bar based on dataset size
    if show_progress is None:
        show_progress = _auto_detect_progress_threshold(len(tensors))

    # Set model to eval mode for calibration
    model.eval()

    # Execute forward passes
    with torch.no_grad():
        if show_progress:
            # Try to use tqdm for progress reporting
            try:
                from tqdm import tqdm
                iterator = tqdm(tensors, desc="Calibrating")
            except ImportError:
                # tqdm not available, use plain iteration
                iterator = tensors
        else:
            iterator = tensors

        for tensor in iterator:
            _ = model(tensor)

    # Return observers dict (may be empty if caller handles attachment)
    if observers is None:
        observers = {}
    return observers


def _auto_detect_progress_threshold(num_samples: int) -> bool:
    """
    Determine whether to show progress bar based on sample count.

    Progress bars are shown for datasets larger than 50 samples, as
    calibration can take noticeable time for larger datasets.

    Args:
        num_samples: Number of calibration samples.

    Returns:
        True if progress bar should be shown, False otherwise.

    Examples:
        >>> from mono_quant.calibration.runner import _auto_detect_progress_threshold
        >>> _auto_detect_progress_threshold(10)
        False
        >>> _auto_detect_progress_threshold(100)
        True
    """
    return num_samples > 50
