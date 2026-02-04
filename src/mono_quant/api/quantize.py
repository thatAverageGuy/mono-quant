"""Unified quantization API.

This module provides the single `quantize()` function that serves as the main
entry point for model quantization. It automatically dispatches to dynamic_quantize
or static_quantize based on the `dynamic` parameter, and handles input normalization
for nn.Module, state_dict, or file path inputs.

Example:
    >>> from mono_quant.api import quantize
    >>> result = quantize(model, bits=8, dynamic=True)
    >>> if result:
    ...     result.save("quantized.safetensors")
"""

from pathlib import Path
from typing import Dict, List, Optional, Union

import torch
import torch.nn as nn

from .result import QuantizationResult

# Type alias for calibration data (re-exported from core)
CalibrationData = Union[List[torch.Tensor], "torch.utils.data.DataLoader"]


def quantize(
    model: Union[nn.Module, Dict[str, torch.Tensor], str, Path],
    bits: int = 8,
    dynamic: bool = False,
    scheme: str = "symmetric",
    calibration_data: Optional[CalibrationData] = None,
    show_progress: bool = False,
    **kwargs,
) -> QuantizationResult:
    """
    Quantize a PyTorch model using a unified API.

    This function provides a single entry point for model quantization,
    automatically dispatching to the appropriate quantization method based
    on the `dynamic` parameter. It accepts multiple input formats and returns
    a QuantizationResult object with the quantized model and metadata.

    Args:
        model: The model to quantize. Can be:
            - nn.Module: A PyTorch model (will be copied, original preserved)
            - Dict[str, torch.Tensor]: A state_dict
            - str/Path: Path to a model file (.pt, .pth, .safetensors)
        bits: Quantization bit width. Options:
            - 4: INT4 quantization (requires calibration_data, static only)
            - 8: INT8 quantization (default)
            - 16: FP16 quantization (simple dtype casting)
        dynamic: If True, use dynamic quantization (no calibration needed).
                 If False (default), use static quantization with calibration.
        scheme: Quantization scheme. Options:
                - "symmetric": Symmetric quantization (default)
                - "asymmetric": Asymmetric quantization
        calibration_data: Calibration data for static quantization.
            Required for static quantization (dynamic=False).
            Can be a list of tensors or a DataLoader.
        show_progress: If True, show progress bars during calibration.
                      Default is False (silent for library use).
        **kwargs: Additional advanced options passed to underlying functions:
            - dtype: Override torch.dtype (auto-detected from bits)
            - symmetric: Override scheme (auto-detected from scheme param)
            - per_channel: Use per-channel scaling (default: True)
            - layer_types: Layer types to quantize (for static)
            - skip_types: Layer types to skip (for static)
            - layer_names: Exact layer names to quantize (for static)
            - skip_names: Layer names to skip (for static)
            - num_calibration_samples: Max calibration samples (default: 150)
            - config: QuantizationConfig object (overrides other params)
            - on_failure: How to handle validation failures ("error"/"warn"/"ignore")
            - run_validation: Whether to run validation (default: True)
            - modules_to_not_convert: Layer names to skip (HuggingFace-compatible)
            - skip_layer_types: Layer types to skip
            - skip_layer_names: Layer name patterns to skip
            - skip_param_threshold: Skip layers below this parameter count
            - group_size: Group size for INT4 (default: 128)
            - accuracy_warning: How to handle accuracy warnings ("warn"/"error"/"ignore")

    Returns:
        QuantizationResult containing:
        - model: The quantized model (nn.Module)
        - info: QuantizationInfo with metadata
        - success: Whether quantization succeeded
        - errors: List of error messages if failed
        - warnings: List of warning messages

    Raises:
        ConfigurationError: If parameters are invalid (bits, scheme).
        InputError: If calibration_data is missing for static quantization.
        InputError: If model file cannot be loaded.

    Examples:
        Dynamic quantization (simple, no calibration):
        >>> import torch.nn as nn
        >>> from mono_quant.api import quantize
        >>> model = nn.Sequential(nn.Linear(128, 256), nn.ReLU())
        >>> result = quantize(model, bits=8, dynamic=True)
        >>> result.save("quantized.safetensors")

        Static quantization (better accuracy, needs calibration):
        >>> calibration_data = [torch.randn(32, 128) for _ in range(100)]
        >>> result = quantize(model, bits=8, calibration_data=calibration_data)
        >>> print(f"SQNR: {result.info.sqnr_db:.2f} dB")

        INT4 quantization (maximum compression):
        >>> result = quantize(model, bits=4, calibration_data=calibration_data)

        FP16 quantization (simple, faster):
        >>> result = quantize(model, bits=16, dynamic=True)

        From file path:
        >>> result = quantize("model.pt", bits=8, dynamic=True)

        With advanced options:
        >>> result = quantize(
        ...     model, bits=8,
        ...     calibration_data=data,
        ...     layer_types=[nn.Linear],
        ...     num_calibration_samples=200,
        ... )
    """
    # Local imports to avoid circular dependencies
    from mono_quant.core.quantizers import (
        QuantizationInfo,
        dynamic_quantize,
        static_quantize,
    )
    from mono_quant.io import load_model
    from mono_quant.io.handlers import _prepare_model

    from .exceptions import ConfigurationError, InputError

    # Step 1: Input normalization
    # If model is a file path, load it first
    if isinstance(model, (str, Path)):
        model_path = str(model)
        try:
            state_dict = load_model(model_path)
            # Convert state_dict to model via _prepare_model
            # Note: We don't have architecture info, so this will fail for pure state_dict
            # Users should provide nn.Module or use dynamic quantization
            if isinstance(state_dict, dict):
                # Try to handle as state_dict - user must provide architecture
                # For now, load_model returns state_dict, we'll pass it through
                # and let the quantizer handle it
                model = state_dict
        except Exception as e:
            raise InputError(
                f"Failed to load model from '{model_path}'",
                suggestion=f"Ensure the file exists and is a valid .pt/.pth/.safetensors file. Error: {e}"
            )

    # Prepare model (copy if nn.Module, handle state_dict)
    try:
        prepared_model = _prepare_model(model)
    except (TypeError, ValueError) as e:
        raise InputError(
            f"Invalid model input: {e}",
            suggestion="Provide an nn.Module, state_dict, or valid file path"
        )

    # Step 2: Parameter validation and mapping

    # Validate and map bits to dtype
    valid_bits = {4, 8, 16}
    if bits not in valid_bits:
        raise ConfigurationError(
            f"Invalid bits value: {bits}. Must be one of {valid_bits}",
            suggestion="Use bits=4 for INT4, bits=8 for INT8, or bits=16 for FP16"
        )

    # Map bits to dtype (can be overridden by **kwargs)
    dtype_map = {
        4: torch.qint8,  # INT4 uses packed int8 storage
        8: torch.qint8,
        16: torch.float16,
    }
    dtype = kwargs.pop("dtype", dtype_map[bits])

    # Validate scheme parameter
    valid_schemes = {"symmetric", "asymmetric"}
    if scheme not in valid_schemes:
        raise ConfigurationError(
            f"Invalid scheme: '{scheme}'. Must be one of {valid_schemes}",
            suggestion="Use scheme='symmetric' or scheme='asymmetric'"
        )

    # Map scheme to symmetric bool (can be overridden by **kwargs)
    symmetric = kwargs.pop("symmetric", scheme == "symmetric")

    # For static quantization, calibration_data is required
    if not dynamic and calibration_data is None:
        raise InputError(
            "calibration_data is required for static quantization (dynamic=False)",
            suggestion="Provide calibration_data or use dynamic=True for dynamic quantization"
        )

    # Step 3: Dispatch to appropriate quantization function
    try:
        if dynamic:
            # Dynamic quantization
            q_model, skipped = dynamic_quantize(
                prepared_model,
                dtype=dtype,
                symmetric=symmetric,
                **kwargs
            )

            # Create QuantizationInfo for dynamic quantization
            info = QuantizationInfo(
                selected_layers=[],  # Dynamic doesn't track selected layers
                skipped_layers=skipped,
                calibration_samples_used=0,
                dtype=dtype,
                symmetric=symmetric,
            )

            # Build result
            result = QuantizationResult(
                model=q_model,
                info=info,
                success=True,
                warnings=list(info.warnings) if info.warnings else [],
            )

        else:
            # Static quantization
            q_model, info = static_quantize(
                prepared_model,
                calibration_data=calibration_data,
                dtype=dtype,
                symmetric=symmetric,
                **kwargs
            )

            # Build result with warnings from info
            result = QuantizationResult(
                model=q_model,
                info=info,
                success=True,
                warnings=list(info.warnings) if info.warnings else [],
            )

        return result

    except Exception as e:
        # On error, return a failed result
        # Check if it's one of our custom exceptions
        from .exceptions import MonoQuantError

        if isinstance(e, MonoQuantError):
            # Re-raise our custom exceptions
            raise

        # For unexpected errors, wrap in a result with error info
        return QuantizationResult(
            model=prepared_model,  # Return the prepared (unquantized) model
            info=QuantizationInfo(
                selected_layers=[],
                skipped_layers=[],
                calibration_samples_used=0,
                dtype=dtype,
                symmetric=symmetric,
            ),
            success=False,
            errors=[f"{type(e).__name__}: {str(e)}"],
        )


__all__ = [
    "quantize",
]
