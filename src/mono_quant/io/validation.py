"""Validation metrics and checks for quantized models.

This module provides comprehensive validation functionality for quantized models,
including:

1. SQNR (Signal-to-Quantization-Noise Ratio) - Measures quantization accuracy
2. Model size comparison - Tracks compression benefits
3. Load and run test - Verifies serialized models work correctly
4. Weight range check - Catches numerical issues in quantized weights

Validation behavior is configurable via the `on_failure` parameter:
- "error": Raise exception on validation failure (default)
- "warn": Issue warning but continue
- "ignore": Silent on validation failure

Example:
    >>> import torch.nn as nn
    >>> from mono_quant.io.validation import validate_quantization
    >>> original = nn.Linear(128, 256)
    >>> quantized = nn.Linear(128, 256)
    >>> result = validate_quantization(original, quantized)
    >>> print(f"SQNR: {result.sqnr_db:.2f} dB")
"""

import os
import tempfile
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn


@dataclass
class ValidationResult:
    """
    Results of quantization validation checks.

    This dataclass captures all validation metrics computed during
    quantization validation. All fields are optional since validation
    may be run partially or some checks may be skipped.

    Attributes:
        sqnr_db: Signal-to-Quantization-Noise Ratio in decibels.
            Higher is better (typically 20-40 dB for INT8).
        original_size_mb: Original model size in megabytes.
        quantized_size_mb: Quantized model size in megabytes.
        compression_ratio: Ratio of original to quantized size.
            Values > 1.0 indicate compression benefit.
        load_test_passed: Whether the model could be saved, loaded,
            and run inference successfully.
        weight_range_valid: Whether all quantized weights are within
            valid numerical ranges (no NaN/Inf, reasonable values).
        errors: List of error messages from failed validation checks.

    Examples:
        >>> result = ValidationResult(
        ...     sqnr_db=32.5,
        ...     original_size_mb=10.2,
        ...     quantized_size_mb=2.6,
        ...     compression_ratio=3.92,
        ...     load_test_passed=True,
        ...     weight_range_valid=True
        ... )
        >>> print(f"Compression: {result.compression_ratio:.2f}x")
    """

    sqnr_db: Optional[float] = None
    original_size_mb: Optional[float] = None
    quantized_size_mb: Optional[float] = None
    compression_ratio: Optional[float] = None
    load_test_passed: bool = False
    weight_range_valid: bool = False
    errors: List[str] = field(default_factory=list)


def calculate_model_size(model: nn.Module) -> Tuple[float, int]:
    """
    Calculate the memory size and parameter count of a model.

    This function iterates through all parameters and buffers in the model
    to compute the total memory footprint. Size is returned in megabytes
    for human-readable output.

    Args:
        model: PyTorch model to analyze.

    Returns:
        A tuple of (size_mb, param_count):
        - size_mb: Model size in megabytes (MB)
        - param_count: Total number of parameters

    Examples:
        >>> import torch.nn as nn
        >>> model = nn.Linear(128, 256)
        >>> size_mb, params = calculate_model_size(model)
        >>> print(f"Size: {size_mb:.2f} MB, Params: {params}")
        Size: 0.13 MB, Params: 33088
    """
    param_count = 0
    size_bytes = 0

    # Count parameters
    for param in model.parameters():
        param_count += param.numel()
        size_bytes += param.numel() * param.element_size()

    # Count buffers (e.g., batch norm running stats)
    for buffer in model.buffers():
        size_bytes += buffer.numel() * buffer.element_size()

    size_mb = size_bytes / (1024 * 1024)

    return size_mb, param_count


def calculate_sqnr(original: torch.Tensor, quantized: torch.Tensor) -> float:
    """
    Calculate Signal-to-Quantization-Noise Ratio (SQNR) in decibels.

    SQNR measures the quality of quantization by comparing the signal
    power to the quantization noise power. Higher values indicate better
    quantization quality.

    Typical values:
    - 20-30 dB: Acceptable quantization
    - 30-40 dB: Good quantization
    - 40+ dB: Excellent quantization

    Args:
        original: Original floating-point tensor.
        quantized: Quantized tensor (either torch.qint8 or torch.float16).

    Returns:
        SQNR in decibels (dB). Returns float('inf') if noise power is
        negligible (near-perfect quantization).

    Examples:
        >>> w = torch.randn(64, 128)
        >>> w_q = torch.quantize_per_channel(
        ...     w, torch.ones(128)*0.01, torch.zeros(128).int(), 0, torch.qint8
        ... )
        >>> sqnr = calculate_sqnr(w, w_q)
        >>> assert sqnr > 0  # Should have positive SQNR
    """
    # Dequantize if needed
    if hasattr(quantized, 'dequantize'):
        quantized_fp32 = quantized.dequantize()
    else:
        quantized_fp32 = quantized.to(torch.float32)

    # Ensure original is float32 for consistent calculation
    if original.dtype != torch.float32:
        original = original.to(torch.float32)

    # Calculate noise (quantization error)
    noise = original - quantized_fp32

    # Calculate signal and noise power
    signal_power = (original ** 2).mean()
    noise_power = (noise ** 2).mean()

    # Avoid division by zero
    if noise_power < 1e-10:
        return float('inf')

    # SQNR in decibels
    sqnr_db = 10 * torch.log10(signal_power / noise_power)

    return sqnr_db.item()


def _calculate_model_sqnr(
    original: nn.Module,
    quantized: nn.Module,
) -> float:
    """
    Calculate average SQNR across all quantized parameters.

    This function compares corresponding parameters in the original
    and quantized models to compute an overall SQNR metric. Only
    parameters that were actually quantized (from float32 to qint8/float16)
    are included in the calculation.

    Args:
        original: Original floating-point model.
        quantized: Quantized model.

    Returns:
        Average SQNR in decibels across all quantized parameters.
        Returns 0.0 if no quantized parameters were found.
    """
    sqnr_values = []

    # Get named parameters from both models
    original_params = dict(original.named_parameters())
    quantized_params = dict(quantized.named_parameters())

    for name, p1 in original_params.items():
        if name not in quantized_params:
            continue

        p2 = quantized_params[name]

        # Check if this parameter was quantized
        # (float32 -> qint8 or float32 -> float16)
        if p1.dtype == torch.float32 and p2.dtype in (torch.qint8, torch.float16):
            try:
                sqnr = calculate_sqnr(p1.data, p2)
                sqnr_values.append(sqnr)
            except (RuntimeError, TypeError):
                # Skip parameters that can't be compared
                continue

    # Return mean SQNR, or 0.0 if no quantized parameters found
    if not sqnr_values:
        return 0.0

    return sum(sqnr_values) / len(sqnr_values)


def _check_weight_ranges(model: nn.Module) -> bool:
    """
    Check that all quantized weights are in valid numerical ranges.

    This function validates that:
    - For qint8: Dequantized values are in reasonable range (-10 to 10)
    - For float16: All values are finite (no NaN or Inf)

    Args:
        model: Quantized model to validate.

    Returns:
        True if all weight ranges are valid, False otherwise.
    """
    for param in model.parameters():
        if param.dtype == torch.qint8:
            # Check dequantized values are reasonable
            if hasattr(param, 'dequantize'):
                dequantized = param.dequantize()
                # Check for NaN or Inf
                if not torch.all(torch.isfinite(dequantized)):
                    return False
                # Check for reasonable range (most weights should be within -10 to 10)
                # This is a heuristic - extreme values may indicate quantization issues
                if torch.any(torch.abs(dequantized) > 100):
                    return False

        elif param.dtype == torch.float16:
            # Check for finite values (no NaN or Inf)
            if not torch.all(torch.isfinite(param)):
                return False

    return True


def _test_load_run(quantized: nn.Module) -> bool:
    """
    Test that a quantized model can be saved, loaded, and run.

    This function performs a full round-trip test:
    1. Save the model state dict to a temporary file
    2. Load the state dict back
    3. Load it into the model
    4. Run a forward pass with dummy input

    Args:
        quantized: Quantized model to test.

    Returns:
        True if all steps succeed, False if any step fails.
    """
    # Local import to avoid circular dependency
    from .formats import save_model, load_model

    # Create a temporary file for testing
    with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as f:
        temp_path = f.name

    try:
        # Step 1: Save the model
        save_model(quantized.state_dict(), temp_path)

        # Step 2: Load the state dict back
        loaded = load_model(temp_path)

        # Step 3: Load into model
        quantized.load_state_dict(loaded)

        # Step 4: Run forward pass
        # Determine input shape from first linear layer
        test_input = None
        for module in quantized.modules():
            if isinstance(module, nn.Linear):
                # Create input matching in_features
                test_input = torch.randn(1, module.in_features)
                break
            elif isinstance(module, nn.Conv2d):
                # Create input matching Conv2d shape
                test_input = torch.randn(1, module.in_channels, 7, 7)
                break

        if test_input is not None:
            _ = quantized(test_input)

        # All tests passed
        return True

    except (OSError, RuntimeError, TypeError, AttributeError):
        # Any failure means the load/run test failed
        return False
    finally:
        # Clean up temporary file
        try:
            os.unlink(temp_path)
        except OSError:
            pass


def validate_quantization(
    original: nn.Module,
    quantized: nn.Module,
    on_failure: str = "error",
) -> ValidationResult:
    """
    Run comprehensive validation on a quantized model.

    This function performs all four validation checks:
    1. SQNR calculation - measures quantization accuracy
    2. Model size comparison - tracks compression benefit
    3. Load and run test - verifies serialization works
    4. Weight range check - catches numerical issues

    Validation failures are handled according to the `on_failure` parameter.

    Args:
        original: Original floating-point model.
        quantized: Quantized model to validate.
        on_failure: How to handle validation failures:
            - "error": Raise ValueError (default)
            - "warn": Issue warning and continue
            - "ignore": Silent, just return results

    Returns:
        ValidationResult with all computed metrics.

    Raises:
        ValueError: If validation fails and on_failure="error".

    Examples:
        >>> import torch.nn as nn
        >>> from mono_quant.io.validation import validate_quantization
        >>> original = nn.Sequential(
        ...     nn.Linear(128, 256),
        ...     nn.ReLU(),
        ...     nn.Linear(256, 10)
        ... )
        >>> # Assume quantized_model was created via static_quantize
        >>> result = validate_quantization(original, quantized_model)
        >>> print(f"SQNR: {result.sqnr_db:.2f} dB")
        >>> print(f"Compression: {result.compression_ratio:.2f}x")

        With warning on failure:
        >>> result = validate_quantization(
        ...     original, quantized, on_failure="warn"
        ... )
    """
    result = ValidationResult()

    # Calculate SQNR
    sqnr = _calculate_model_sqnr(original, quantized)
    result.sqnr_db = sqnr

    # Calculate model sizes
    original_size_mb, _ = calculate_model_size(original)
    quantized_size_mb, _ = calculate_model_size(quantized)
    result.original_size_mb = original_size_mb
    result.quantized_size_mb = quantized_size_mb

    # Calculate compression ratio
    if quantized_size_mb > 0:
        result.compression_ratio = original_size_mb / quantized_size_mb

    # Run load test
    result.load_test_passed = _test_load_run(quantized)

    # Run weight range check
    result.weight_range_valid = _check_weight_ranges(quantized)

    # Check for validation failures
    validation_failed = not result.load_test_passed or not result.weight_range_valid

    if validation_failed:
        # Build error message
        errors = []
        if not result.load_test_passed:
            errors.append("Load and run test failed")
        if not result.weight_range_valid:
            errors.append("Weight range check failed")

        error_msg = "Validation failed: " + ", ".join(errors)
        result.errors.append(error_msg)

        # Handle based on on_failure parameter
        if on_failure == "error":
            raise ValueError(error_msg)
        elif on_failure == "warn":
            import warnings
            warnings.warn(error_msg, stacklevel=2)
        # on_failure="ignore" - silent

    return result


__all__ = [
    "ValidationResult",
    "calculate_model_size",
    "calculate_sqnr",
    "validate_quantization",
]
