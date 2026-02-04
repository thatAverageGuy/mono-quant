"""Quantization configuration dataclass."""

from dataclasses import dataclass
from typing import Optional, Union

import torch


@dataclass
class QuantizationConfig:
    """
    Configuration for model quantization.

    Args:
        dtype: Target quantization dtype. Must be one of:
            - torch.qint8 (default): 8-bit signed integer quantization
            - torch.float16: 16-bit floating point
            - torch.quint8: 8-bit unsigned integer quantization
        symmetric: Symmetric vs asymmetric quantization. None means
            layer-dependent (default behavior for PyTorch quantization).
        per_channel: If True, use per-channel scaling. If False, use
            per-tensor scaling (default: True).
        bits: Quantization bit width (default: 8).

    Raises:
        ValueError: If dtype is not a supported quantization dtype.
    """

    dtype: torch.dtype = torch.qint8
    symmetric: Optional[bool] = None
    per_channel: bool = True
    bits: int = 8

    _VALID_DTYPES = {
        torch.qint8,
        torch.quint8,
        torch.float16,
    }

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if self.dtype not in self._VALID_DTYPES:
            raise ValueError(
                f"Invalid dtype {self.dtype}. Must be one of: "
                f"{', '.join(str(d) for d in sorted(self._VALID_DTYPES, key=str))}"
            )
        if self.bits not in {4, 8, 16}:
            raise ValueError(f"Invalid bits {self.bits}. Must be 4, 8, or 16.")
        if self.bits == 4 and self.dtype != torch.qint8:
            raise ValueError("4-bit quantization only supports qint8 dtype.")
        if self.bits == 16 and self.dtype != torch.float16:
            raise ValueError("16-bit quantization only supports float16 dtype.")

    @classmethod
    def from_kwargs(
        cls,
        config: Optional["QuantizationConfig"] = None,
        **kwargs: Union[torch.dtype, bool, int],
    ) -> "QuantizationConfig":
        """
        Create config with priority: kwargs > config > defaults.

        Args:
            config: Optional base configuration.
            **kwargs: Override parameters (dtype, symmetric, per_channel, bits).

        Returns:
            New QuantizationConfig with applied overrides.

        Examples:
            >>> default_config = QuantizationConfig()
            >>> custom = QuantizationConfig.from_kwargs(
            ...     default_config,
            ...     dtype=torch.float16,
            ...     per_channel=False
            ... )
        """
        if config is None:
            config = cls()

        # Extract only valid kwargs
        valid_keys = {"dtype", "symmetric", "per_channel", "bits"}
        overrides = {k: v for k, v in kwargs.items() if k in valid_keys}

        return cls(**{**config.__dict__, **overrides})
