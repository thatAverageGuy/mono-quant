"""Core quantization functionality.

This module provides the core quantization API including:
- Quantization schemes (symmetric, asymmetric)
- Weight quantization and dequantization functions (INT8, INT4, FP16)
- Dynamic and static quantization
- Calibration observers for tracking activation ranges
- Group-wise quantization for INT4 weights

Observers available:
- MinMaxObserver: Simple min/max tracking (baseline)
- MovingAverageMinMaxObserver: EMA-based smoothing for outlier handling
- HistogramObserver: KL divergence minimization for skewed distributions

Choose MovingAverageMinMaxObserver for data with transient spikes or outliers.
Choose HistogramObserver for data with skewed distributions or heavy tails.
Use MinMaxObserver for well-behaved, normally distributed activations.
"""

from mono_quant.core.mappers import (
    calculate_scale_zp_groupwise,
    calculate_scale_zp_per_channel,
    calculate_scale_zp_per_tensor,
    get_dtype_range,
)
from mono_quant.core.observers import (
    HistogramObserver,
    MinMaxObserver,
    MovingAverageMinMaxObserver,
    _select_layers_by_name,
    _select_layers_by_type,
)
from mono_quant.core.quantizers import (
    QuantizationInfo,
    dequantize_model,
    dequantize_weight,
    dynamic_quantize,
    quantize_weight_fp16,
    quantize_weight_int4,
    quantize_weight_int8,
    revert_to_standard_modules,
    static_quantize,
    test_models_from_any_source,
)
from mono_quant.core.schemes import (
    AsymmetricScheme,
    QuantizationScheme,
    SymmetricScheme,
)

__all__ = [
    # Schemes
    "QuantizationScheme",
    "SymmetricScheme",
    "AsymmetricScheme",
    # Mappers
    "calculate_scale_zp_per_tensor",
    "calculate_scale_zp_per_channel",
    "calculate_scale_zp_groupwise",
    "get_dtype_range",
    # Quantizers
    "quantize_weight_int8",
    "quantize_weight_int4",
    "quantize_weight_fp16",
    "dequantize_weight",
    "dequantize_model",
    "revert_to_standard_modules",
    # Dynamic quantization
    "dynamic_quantize",
    # Static quantization
    "static_quantize",
    "QuantizationInfo",
    # Observers
    "MinMaxObserver",
    "MovingAverageMinMaxObserver",
    "HistogramObserver",
    # Layer selection (internal but exported for advanced use)
    "_select_layers_by_type",
    "_select_layers_by_name",
    # Testing
    "test_models_from_any_source",
]
