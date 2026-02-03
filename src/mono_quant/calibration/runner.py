"""
Calibration runner for static quantization.

This module provides the run_calibration() function that executes forward passes
through a model with calibration data. This allows observers (like MinMaxObserver)
to track activation ranges for computing quantization parameters.

The runner includes optional progress reporting for large datasets (auto-detects
when tqdm is available and dataset exceeds threshold).

The observer factory enables easy instantiation of observers by string name:
- MinMaxObserver: Simple min/max tracking (baseline)
- MovingAverageMinMaxObserver: EMA-based smoothing for outlier handling
- HistogramObserver: KL divergence minimization for skewed distributions
- auto: Experimental auto-selection based on provided kwargs
"""

from typing import Dict, List, Optional, Union

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from mono_quant.core.observers import MinMaxObserver
from .data import _normalize_calibration_data, _limit_samples, CalibrationData


def create_observer(
    observer_type: str,
    **kwargs,
) -> Union["MinMaxObserver", "MovingAverageMinMaxObserver", "HistogramObserver"]:
    """
    Create an observer instance by string name.

    This factory function enables easy instantiation of observer classes
    without needing to import them directly. Observers are created by
    specifying the type as a string and passing any constructor arguments
    as kwargs.

    Args:
        observer_type: Type of observer to create. Case-insensitive.
                      Options:
                      - "minmax" or "minmaxobserver": MinMaxObserver
                      - "movingaverage", "movingaverageminmax", or "ema":
                        MovingAverageMinMaxObserver
                      - "histogram", "histogramobserver", or "kl":
                        HistogramObserver
                      - "auto": Auto-select based on kwargs (experimental)
        **kwargs: Additional arguments passed to the observer constructor.
                  Common kwargs:
                  - averaging_constant: For MovingAverageMinMaxObserver
                  - bins: For HistogramObserver
                  - dtype: For any observer

    Returns:
        An observer instance of the requested type.

    Raises:
        ValueError: If observer_type is not recognized.

    Examples:
        >>> from mono_quant.calibration.runner import create_observer
        >>> # Create MinMaxObserver
        >>> obs1 = create_observer("MinMax")
        >>> # Create MovingAverageMinMaxObserver with custom averaging
        >>> obs2 = create_observer("MovingAverage", averaging_constant=0.05)
        >>> # Create HistogramObserver with custom bins
        >>> obs3 = create_observer("Histogram", bins=1024)
        >>> # Auto-selection (experimental)
        >>> obs4 = create_observer("auto", averaging_constant=0.01)
    """
    # Normalize observer_type: lowercase, remove underscores and hyphens
    normalized = observer_type.lower().replace("_", "").replace("-", "")

    # Local import to avoid circular dependency
    from mono_quant.core.observers import (
        MinMaxObserver,
        MovingAverageMinMaxObserver,
        HistogramObserver,
    )

    # Match against observer types
    if normalized in ("minmax", "minmaxobserver"):
        return MinMaxObserver(**kwargs)
    elif normalized in ("movingaverage", "movingaverageminmax", "ema"):
        return MovingAverageMinMaxObserver(**kwargs)
    elif normalized in ("histogram", "histogramobserver", "kl"):
        return HistogramObserver(**kwargs)
    elif normalized == "auto":
        return _auto_select_observer(**kwargs)
    else:
        raise ValueError(
            f"Unknown observer_type: '{observer_type}'. "
            f"Valid options: 'minmax', 'movingaverage', 'histogram', 'auto'"
        )


def _auto_select_observer(
    **kwargs,
) -> Union["MinMaxObserver", "MovingAverageMinMaxObserver", "HistogramObserver"]:
    """
    Auto-select observer type based on provided kwargs.

    EXPERIMENTAL: Auto-selection is unreliable. Manual selection based on
    your data characteristics is recommended.

    This function uses simple heuristics to select an observer:
    - If averaging_constant is specified: MovingAverageMinMaxObserver
    - If bins is specified: HistogramObserver
    - Otherwise: MinMaxObserver (safe default)

    More sophisticated auto-selection (dataset size, distribution analysis)
    is deferred to future work.

    Args:
        **kwargs: Keyword arguments that may indicate observer preference.

    Returns:
        An observer instance selected based on heuristics.

    Examples:
        >>> from mono_quant.calibration.runner import create_observer
        >>> # Auto-selects MovingAverageMinMaxObserver
        >>> obs = create_observer("auto", averaging_constant=0.01)
    """
    # Local import to avoid circular dependency
    from mono_quant.core.observers import (
        MinMaxObserver,
        MovingAverageMinMaxObserver,
        HistogramObserver,
    )

    # Simple heuristics based on kwargs
    if "averaging_constant" in kwargs:
        return MovingAverageMinMaxObserver(**kwargs)
    elif "bins" in kwargs:
        return HistogramObserver(**kwargs)
    else:
        # Safe default
        return MinMaxObserver(**kwargs)


def run_calibration(
    model: nn.Module,
    calibration_data: CalibrationData,
    observers: Optional[Dict[str, Union["MinMaxObserver", "MovingAverageMinMaxObserver", "HistogramObserver"]]] = None,
    num_samples: int = 150,
    show_progress: Optional[bool] = None,
) -> Dict[str, Union["MinMaxObserver", "MovingAverageMinMaxObserver", "HistogramObserver"]]:
    """
    Run calibration forward passes through a model.

    This function executes forward passes using calibration data, allowing
    observers attached to the model to track activation ranges. The model
    is set to eval mode and gradients are disabled during calibration.

    Progress bar is automatically shown for large datasets (default threshold
    is 50 samples) when tqdm is available. Can be controlled with show_progress.

    Use create_observer() factory to instantiate observers by string name:
        - "minmax": Simple min/max tracking (baseline)
        - "movingaverage": EMA-based smoothing for outlier handling
        - "histogram": KL divergence minimization for skewed distributions
        - "auto": Experimental auto-selection

    Args:
        model: PyTorch model to calibrate. Should have observer instances
               attached to layers where activation quantization is needed.
        calibration_data: Calibration data in one of the following formats:
            - List[torch.Tensor]: Direct list of input tensors
            - DataLoader: PyTorch DataLoader yielding batches
        observers: Optional dictionary of observer name to observer instance.
                   Use create_observer() to create instances. If None, returns
                   empty dict (observers assumed attached by caller via hook
                   registration or custom layers).
        num_samples: Maximum number of samples to use for calibration.
                     Default is 150, based on research recommending 100-200
                     baseline samples for static quantization.
        show_progress: Whether to show progress bar. If None, auto-detects
                      based on sample count (True for >50 samples).

    Returns:
        Dictionary of observer names to observer instances with observed
        values. The observers can be used to compute quantization parameters
        via calculate_qparams().

    Examples:
        >>> from mono_quant.calibration import run_calibration
        >>> from mono_quant.calibration.runner import create_observer
        >>> import torch.nn as nn
        >>> model = nn.Sequential(nn.Linear(10, 20), nn.ReLU())
        >>> data = [torch.randn(4, 10) for _ in range(100)]
        >>> # Create observers for calibration
        >>> observers = {
        ...     "0": create_observer("MinMax"),
        ...     "2": create_observer("MovingAverage", averaging_constant=0.01)
        ... }
        >>> result = run_calibration(model, data, observers=observers, num_samples=50)
        >>> scale, zp = result["0"].calculate_qparams()
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
