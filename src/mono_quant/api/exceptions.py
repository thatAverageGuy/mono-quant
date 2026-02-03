"""Custom exception hierarchy for mono-quant.

This module provides a structured exception hierarchy with actionable suggestions
to help users resolve quantization errors quickly.

All exceptions inherit from MonoQuantError and support an optional `suggestion`
parameter that provides guidance on how to fix the issue.

Example:
    >>> from mono_quant.api.exceptions import QuantizationError
    >>> raise QuantizationError(
    ...     "Layer dimension 64 is smaller than group_size 128",
    ...     suggestion="Use group_size=64 or smaller for this model"
    ... )
"""

from typing import Optional


class MonoQuantError(Exception):
    """
    Base exception for all mono-quant errors.

    This exception class supports an optional `suggestion` parameter that
    provides actionable guidance for resolving the error.

    Attributes:
        message: The error message describing what went wrong.
        suggestion: Optional suggestion for how to fix the issue.

    Examples:
        >>> from mono_quant.api.exceptions import MonoQuantError
        >>> try:
        ...     raise MonoQuantError(
        ...         "Something went wrong",
        ...         suggestion="Try adjusting your configuration"
        ...     )
        ... except MonoQuantError as e:
        ...     print(str(e))
        Something went wrong

        Suggestion: Try adjusting your configuration
    """

    def __init__(self, message: str, suggestion: Optional[str] = None) -> None:
        """
        Initialize the exception with message and optional suggestion.

        Args:
            message: The error message describing what went wrong.
            suggestion: Optional suggestion for how to fix the issue.
        """
        self.message = message
        self.suggestion = suggestion

        # Build full message with suggestion if provided
        full_message = message
        if suggestion:
            full_message += f"\n\nSuggestion: {suggestion}"

        super().__init__(full_message)


class QuantizationError(MonoQuantError):
    """
    Raised when quantization fails during the quantization process.

    This exception indicates that the quantization operation itself failed,
    such as when a layer's dimension is incompatible with the group size,
    or when numerical issues prevent successful quantization.

    Examples:
        >>> from mono_quant.api.exceptions import QuantizationError
        >>> raise QuantizationError(
        ...     "Layer dimension 64 is smaller than group_size 128",
        ...     suggestion="Use group_size=64 or smaller for this model"
        ... )
    """

    pass


class ValidationError(MonoQuantError):
    """
    Raised when model validation fails.

    This exception is raised when post-quantization validation detects issues
    such as poor SQNR scores, load test failures, or weight range violations.

    Examples:
        >>> from mono_quant.api.exceptions import ValidationError
        >>> raise ValidationError(
        ...     "SQNR too low: 8.5 dB",
        ...     suggestion="Consider using fewer layers or INT8 instead of INT4"
        ... )
    """

    pass


class ConfigurationError(MonoQuantError):
    """
    Raised when configuration parameters are invalid.

    This exception is raised when the provided quantization parameters
    are invalid or incompatible, such as unsupported bit widths,
    invalid scheme names, or conflicting options.

    Examples:
        >>> from mono_quant.api.exceptions import ConfigurationError
        >>> raise ConfigurationError(
        ...     "Invalid bits value: 5. Must be 4, 8, or 16",
        ...     suggestion="Use bits=4 for INT4, bits=8 for INT8, or bits=16 for FP16"
        ... )
    """

    pass


class InputError(MonoQuantError):
    """
    Raised when input model or data is invalid.

    This exception is raised when the input model is not recognized,
    cannot be loaded, or is in an incompatible format.

    Examples:
        >>> from mono_quant.api.exceptions import InputError
        >>> raise InputError(
        ...     "Unsupported model format: .onnx",
        ...     suggestion="Provide a PyTorch .pt/.pth file or an nn.Module"
        ... )
    """

    pass


__all__ = [
    "MonoQuantError",
    "QuantizationError",
    "ValidationError",
    "ConfigurationError",
    "InputError",
]
