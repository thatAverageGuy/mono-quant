"""Quantization result dataclass with convenience methods.

This module provides the QuantizationResult dataclass which encapsulates
the results of a quantization operation, including the quantized model,
metadata, success status, and any errors or warnings.

Example:
    >>> from mono_quant.api import quantize
    >>> result = quantize(model, bits=8, dynamic=True)
    >>> if result:
    ...     result.save("quantized_model.safetensors")
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Union

import torch
import torch.nn as nn


@dataclass
class QuantizationResult:
    """
    Result of a quantization operation.

    This dataclass captures the output of the unified `quantize()` function,
    providing both direct attribute access to the quantized model and metadata,
    as well as convenience methods for common operations like saving and validation.

    Attributes:
        model: The quantized PyTorch model (nn.Module).
        info: QuantizationInfo metadata containing quantization details
              such as selected layers, calibration samples, SQNR, etc.
        success: Whether quantization succeeded. If errors list is non-empty,
                 this is automatically set to False in __post_init__.
        errors: List of error messages if quantization failed.
        warnings: List of warning messages about potential issues.

    Examples:
        >>> import torch.nn as nn
        >>> from mono_quant.api.result import QuantizationResult
        >>> from mono_quant.core.quantizers import QuantizationInfo
        >>> model = nn.Linear(10, 20)
        >>> info = QuantizationInfo(
        ...     selected_layers=["0"],
        ...     skipped_layers=[],
        ...     calibration_samples_used=100,
        ...     dtype=torch.qint8,
        ...     symmetric=False,
        ... )
        >>> result = QuantizationResult(model=model, info=info)
        >>> result.success
        True
        >>> bool(result)
        True
    """

    model: nn.Module
    info: "QuantizationInfo"
    success: bool = True
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        """
        Post-initialization processing.

        Automatically sets success=False if errors list is non-empty.
        This ensures consistency between the errors list and success flag.
        """
        if self.errors:
            self.success = False

    def save(self, path: Union[str, Path]) -> None:
        """
        Save the quantized model to disk.

        This is a convenience method that wraps the save_model function
        from mono_quant.io, automatically including the quantization info
        in the saved file metadata.

        Args:
            path: Path where to save the model. File extension determines
                  format (.safetensors for Safetensors, .pt/.pth for PyTorch).

        Raises:
            OSError: If the file cannot be written.
            RuntimeError: If the model cannot be serialized.

        Examples:
            >>> from mono_quant.api import quantize
            >>> result = quantize(model, bits=8, dynamic=True)
            >>> result.save("quantized.safetensors")

            With Path object:
            >>> from pathlib import Path
            >>> result.save(Path("models") / "quantized.pt")
        """
        # Local import to avoid circular dependency
        from mono_quant.io import save_model

        save_model(self.model, path, quantization_info=self.info)

    def validate(
        self,
        original: Optional[nn.Module] = None,
        on_failure: str = "error",
    ) -> "ValidationResult":
        """
        Validate the quantized model.

        This method runs validation checks on the quantized model, including
        SQNR calculation, model size comparison, load/run testing, and
        weight range checks.

        Note: This requires the original model for comparison. If the original
        model is not provided, validation will be skipped with a warning.

        Args:
            original: The original pre-quantization model for comparison.
                      If None, validation will be limited.
            on_failure: How to handle validation failures:
                        - "error": Raise exception (default)
                        - "warn": Issue warning and continue
                        - "ignore": Silent, just return results

        Returns:
            ValidationResult with validation metrics.

        Raises:
            ValueError: If validation fails and on_failure="error".

        Examples:
            >>> result = quantize(model, bits=8, dynamic=True)
            >>> validation = result.validate(original=original_model)
            >>> print(f"SQNR: {validation.sqnr_db:.2f} dB")
        """
        if original is None:
            raise NotImplementedError(
                "Validation requires the original model for comparison. "
                "Use validate_quantization(original, result.model) directly "
                "or pass the original model to this method."
            )

        # Local import to avoid circular dependency
        from mono_quant.io.validation import validate_quantization

        return validate_quantization(original, self.model, on_failure=on_failure)

    def __bool__(self) -> bool:
        """
        Boolean conversion for truthiness checks.

        Allows the result to be used directly in conditional statements,
        checking whether quantization succeeded.

        Returns:
            True if success is True, False otherwise.

        Examples:
            >>> result = quantize(model, bits=8, dynamic=True)
            >>> if result:
            ...     print("Quantization succeeded!")
            ...     result.save("model.safetensors")
        """
        return self.success


__all__ = [
    "QuantizationResult",
]
