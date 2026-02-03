# Phase 3: Advanced Calibration & INT4 - Research

**Researched:** 2026-02-03
**Domain:** Neural Network Quantization (INT4, Advanced Calibration, Layer Skipping)
**Confidence:** HIGH

## Summary

Phase 3 extends the mono-quant library with INT4 quantization using group-wise scaling, advanced calibration observers (MovingAverageMinMax, Histogram), layer skipping mechanisms to protect sensitive components, and an accuracy warning system for aggressive quantization scenarios.

**Primary recommendations:**
1. Use group_size=128 as default for INT4 (industry standard from HuggingFace, AWQ, GPTQ)
2. Implement MovingAverageMinMaxObserver with averaging_constant=0.01 (PyTorch standard)
3. Implement HistogramObserver using KL-divergence entropy minimization
4. Use unified layer skipping API with `modules_to_not_convert` parameter
5. Set SQNR warning thresholds: <20dB (error), 20-30dB (warn), >30dB (good)

**Key insight:** INT4 quantization differs fundamentally from INT8 in that it requires group-wise scaling to maintain accuracy. Per-channel scaling (INT8 approach) results in unacceptable accuracy loss at 4-bit precision, hence the need for group-wise quantization with shared scales across groups of channels.

## Standard Stack

### Core

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| PyTorch | 2.8+ | Base quantization framework | Native per-channel quantization support, observer implementations |
| NumPy | Latest | Numerical operations | Standard for tensor manipulation |

### Supporting

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| safetensors | Latest | Serialized quantized model format | Already used in Phase 1-2, continue for consistency |
| tqdm | Latest | Optional progress display | Calibration can take time for large datasets |

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Custom INT4 | bitsandbytes NF4/FP4 | bitsandbytes is LLM-focused, NF4 is special-purpose |
| Custom observers | torch.ao.quantization observers | PyTorch's observers are deprecated for removal in 2.10+ |
| Custom group-wise | AutoGPTQ/AWQ libraries | Heavy dependencies, not model-agnostic |

**Installation:**
```bash
# No new dependencies required
# Existing dependencies from Phase 1-2 are sufficient
```

## Architecture Patterns

### Recommended Project Structure

```
src/mono_quant/
├── core/
│   ├── observers.py       # Add MovingAverageMinMaxObserver, HistogramObserver
│   ├── mappers.py         # Add group-wise scale/zp calculation functions
│   └── quantizers.py      # Add quantize_weight_int4 function
├── calibration/
│   ├── runner.py          # Extend for new observer types
│   └── data.py            # No changes needed
├── modules/
│   └── linear.py          # Add QuantizedLinearInt4 module
├── io/
│   ├── validation.py      # Add accuracy warning system
│   └── formats.py         # No changes needed
└── config/
    └── quant_config.py    # Add INT4 configuration classes
```

### Pattern 1: INT4 Group-Wise Quantization

**What:** Divide weight tensor into groups along the output channel dimension, with each group sharing a scale and zero-point.

**When to use:** INT4 quantization (mandatory). Group-wise scaling is required for 4-bit precision to maintain acceptable accuracy.

**Key parameters:**
- `group_size`: Number of channels per group (default: 128)
- `axis`: Channel dimension for grouping (default: 0 for output channels)
- `dtype`: Target quantization dtype (torch.int4 or custom storage)

**Example implementation approach:**
```python
# Source: Based on AWQ/GPTQ group-wise patterns
def calculate_scale_zp_groupwise(
    weight: torch.Tensor,
    group_size: int = 128,
    axis: int = 0,
    symmetric: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Calculate per-group scale and zero-point for INT4 quantization.

    Args:
        weight: Weight tensor of shape (out_features, in_features)
        group_size: Number of channels per group (default: 128)
        axis: Channel axis (default: 0 for output channels)
        symmetric: Use symmetric quantization (recommended for INT4)

    Returns:
        scale: Shape (num_groups,) - one scale per group
        zero_point: Shape (num_groups,) - one zero-point per group
    """
    # Implementation details in PLAN.md
```

**Industry standard:** Group size of 128 is the de facto standard across AWQ, GPTQ, HuggingFace, and modern research (2024-2025). This value provides the best accuracy/computation trade-off for transformer models.

### Pattern 2: MovingAverageMinMaxObserver

**What:** Observer that computes quantization parameters using exponential moving average of min/max values.

**When to use:** Calibration data has outliers or transient spikes that would distort MinMaxObserver's range estimation.

**Example:**
```python
# Source: PyTorch 2.8 MovingAverageMinMaxObserver documentation
# https://docs.pytorch.org/docs/stable/generated/torch.ao.quantization.observer.MovingAverageMinMaxObserver.html

class MovingAverageMinMaxObserver:
    """
    Observer using exponential moving average of min/max values.

    Formula:
        x_min = (1 - c) * x_min + c * min(X)  # if x_min exists
        x_max = (1 - c) * x_max + c * max(X)  # if x_max exists

    Where c is the averaging_constant (default: 0.01 per PyTorch).

    Use when:
    - Calibration data contains outliers
    - Want smoother calibration statistics
    - MinMaxObserver produces too wide ranges due to spikes
    """
    def __init__(self, averaging_constant: float = 0.01, dtype: torch.dtype = torch.qint8):
        self.averaging_constant = averaging_constant
        self.min_val = None
        self.max_val = None

    def forward(self, x: torch.Tensor) -> None:
        x_min = x.amin().item()
        x_max = x.amax().item()

        if self.min_val is None:
            self.min_val = x_min
            self.max_val = x_max
        else:
            # Exponential moving average
            c = self.averaging_constant
            self.min_val = (1 - c) * self.min_val + c * x_min
            self.max_val = (1 - c) * self.max_val + c * x_max
```

**Default averaging_constant:** PyTorch uses 0.01. This provides a good balance between responsiveness and stability. Lower values (e.g., 0.001) are more stable but adapt slower; higher values (e.g., 0.1) adapt faster but are more sensitive to outliers.

### Pattern 3: HistogramObserver (KL Divergence Minimization)

**What:** Observer that builds a histogram of activation values and finds optimal quantization thresholds using KL divergence (entropy) minimization.

**When to use:**
- Need outlier-aware calibration
- Data distribution is highly non-uniform
- Want to minimize information loss (KL divergence)
- TensorRT uses this approach for production quantization

**Algorithm:**
```python
# Source: PyTorch quantization-in-practice blog
# https://pytorch.org/blog/quantization-in-practice/
#
# TensorRT uses entropy minimization (KL divergence) to find
# optimal quantization parameters that minimize information loss.

class HistogramObserver:
    """
    Observer using KL divergence minimization for threshold selection.

    Algorithm:
    1. Build histogram of activation values
    2. For candidate threshold values:
       - Compute KL divergence between original and quantized distributions
    3. Select threshold with minimum KL divergence

    Use when:
    - Data has significant outliers
    - Distribution is highly skewed
    - MinMaxObserver produces suboptimal ranges
    """
    def __init__(self, bins: int = 2048, dtype: torch.dtype = torch.qint8):
        self.bins = bins
        self.histogram = None  # (counts, bin_edges)

    def forward(self, x: torch.Tensor) -> None:
        # Update histogram with new values
        new_hist = torch.histogram(x.flatten(), bins=self.bins)
        if self.histogram is None:
            self.histogram = new_hist
        else:
            # Accumulate histograms
            self.histogram = (self.histogram[0] + new_hist[0], new_hist[1])

    def calculate_qparams(self) -> Tuple[torch.Tensor, torch.Tensor]:
        # Find optimal threshold using KL divergence minimization
        # Implementation in PLAN.md
```

**Performance note:** HistogramObserver is computationally more expensive than MinMaxObserver. Use it when calibration quality matters more than speed, or when dealing with challenging distributions.

### Pattern 4: Layer Skipping API (Unified)

**What:** Unified interface for excluding sensitive layers from quantization.

**When to use:** INT4 quantization (always), or when certain layers are known to be quantization-sensitive.

**Recommended API:**
```python
# Source: HuggingFace AwqConfig, GPTQConfig patterns
# https://huggingface.co/docs/transformers/main_classes/quantization

# Unified API supporting multiple input formats
def static_quantize(
    model: nn.Module,
    calibration_data: CalibrationData,
    dtype: torch.dtype = torch.qint8,
    group_size: int = 128,  # NEW for INT4

    # Layer skipping - UNIFIED APPROACH
    modules_to_not_convert: Optional[List[str]] = None,  # NEW: primary parameter
    skip_layer_types: Optional[LayerTypes] = None,        # Optional: type-based
    skip_layer_names: Optional[List[str]] = None,        # Optional: name-based

    # Observers
    observer: str = "MinMax",  # "MinMax", "MovingAverage", "Histogram", "Auto"
    averaging_constant: float = 0.01,  # For MovingAverage

    # Accuracy warnings
    accuracy_warning: str = "warn",  # "error", "warn", "ignore"
    sqnr_threshold: float = 20.0,  # Warning threshold in dB
) -> Tuple[nn.Module, QuantizationInfo]:
```

**Recommended default skip list for INT4:**
```python
DEFAULT_INT4_SKIP = {
    # By type
    "skip_types": (nn.Embedding, nn.EmbeddingBag, nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d),

    # By name pattern (layer < 512 parameters)
    "skip_param_threshold": 512,

    # Commonly sensitive layers (name-based)
    "skip_names": ["lm_head"],  # Output head often sensitive
}
```

**Why unified approach:**
- HuggingFace uses `modules_to_not_convert` across all quantization configs (AwqConfig, GPTQConfig, etc.)
- Single parameter is clearer than multiple type/name/pattern parameters
- Still allows advanced users to use type/name-based filtering via helper functions

### Pattern 5: Accuracy Warning System

**What:** System that warns users when quantization quality metrics indicate potential accuracy loss.

**When to use:** All quantization operations, especially aggressive quantization (INT4).

**Metrics to track:**

| Metric | Description | Warning Threshold | Error Threshold |
|--------|-------------|-------------------|-----------------|
| SQNR | Signal-to-Quantization-Noise Ratio | < 20 dB | < 10 dB |
| Quantization ratio | Fraction of layers quantized | > 95% | 100% (all layers) |
| Outlier count | Parameters with extreme values | > 1% | > 5% |

**Example integration:**
```python
# Extend existing ValidationResult
@dataclass
class ValidationResult:
    # Existing fields...
    sqnr_db: Optional[float] = None
    warnings: List[str] = field(default_factory=list)  # NEW

def check_accuracy_warnings(
    result: ValidationResult,
    on_failure: str = "warn",
    sqnr_threshold: float = 20.0,
) -> ValidationResult:
    """
    Add warnings based on quality metrics.

    SQNR quality guide:
    - > 30 dB: Excellent quantization
    - 20-30 dB: Good quantization
    - 10-20 dB: Warning - may impact accuracy
    - < 10 dB: Error - significant accuracy loss likely
    """
    if result.sqnr_db is not None:
        if result.sqnr_db < 10.0:
            result.errors.append(f"Very low SQNR: {result.sqnr_db:.2f} dB")
        elif result.sqnr_db < sqnr_threshold:
            result.warnings.append(
                f"Low SQNR ({result.sqnr_db:.2f} dB < {sqnr_threshold} dB): "
                "quantization may impact accuracy"
            )

    # Handle based on on_failure parameter
    if on_failure == "error" and result.errors:
        raise ValueError("; ".join(result.errors))
    elif on_failure == "warn" and result.errors:
        import warnings
        for error in result.errors:
            warnings.warn(error)

    return result
```

**CI/CD compatibility:** Use `on_failure="ignore"` for automated pipelines where warnings should not break builds.

### Anti-Patterns to Avoid

- **Anti-pattern: Using PyTorch's built-in observers (torch.ao.quantization.observer)**
  - Why: These are deprecated for removal in PyTorch 2.10+
  - Instead: Implement custom observers following Phase 2's MinMaxObserver pattern

- **Anti-pattern: INT4 with per-channel scaling**
  - Why: Results in unacceptable accuracy loss (<10 dB SQNR typical)
  - Instead: Always use group-wise scaling for INT4

- **Anti-pattern: HistogramObserver for small calibration datasets (<50 samples)**
  - Why: Insufficient data to build meaningful histogram
  - Instead: Use MinMaxObserver or MovingAverageMinMaxObserver

- **Anti-pattern: Hard-coding module type checks throughout code**
  - Why: Difficult to extend for new layer types
  - Instead: Use centralized configuration with helper functions

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| INT4 quantization algorithm | Custom group-wise logic | AWQ/GPTQ patterns | Group-wise INT4 requires careful tuning; use proven approaches |
| Observer selection heuristics | Complex auto-selection logic | Simple rules + manual override | Auto-selection is unreliable; mark as experimental |
| Layer type detection | Manual isinstance checks | PyTorch's named_modules() traversal | More robust, handles nested modules |
| SQNR calculation | Custom implementation | Existing Phase 2 code | Already tested, handles edge cases |

**Key insight:** INT4 quantization is non-trivial. The group-wise scaling approach, optimal group sizes, and calibration strategies are the result of extensive research. Don't invent new algorithms; implement established patterns from AWQ, GPTQ, and HuggingFace.

## Common Pitfalls

### Pitfall 1: Layer Smaller Than Group Size

**What goes wrong:** When layer output dimension < group_size, cannot form complete groups. Leads to index errors or incorrect quantization.

**Why it happens:** Group-wise quantization assumes the channel dimension is divisible by group_size.

**How to avoid:**
```python
# Safe approach: Fallback to per-channel quantization
if weight.shape[axis] < group_size:
    # Fallback to per-channel (each channel is its own "group")
    return calculate_scale_zp_per_channel(weight, dtype=dtype, symmetric=symmetric, axis=axis)
```

**Alternative approaches:**
1. **Error:** Raise ValueError telling user to reduce group_size
2. **Skip:** Don't quantize this layer, add to skipped_layers list
3. **Pad:** Add zero padding (NOT recommended - distorts statistics)

**Recommendation:** Fallback to per-channel quantization. This maintains the "quantize what you can" philosophy and provides a reasonable default.

### Pitfall 2: Embedding Layer Quantization

**What goes wrong:** Quantizing embeddings to INT4 often causes significant accuracy degradation, especially for NLP tasks.

**Why it happens:** Embeddings have different statistical properties than linear layer weights. They often have wide value ranges and critical semantic information that is sensitive to quantization.

**How to avoid:** Default skip list for INT4 should always include:
- `nn.Embedding`
- `nn.EmbeddingBag`

**User override:** Allow users to explicitly opt-in to embedding quantization if desired.

### Pitfall 3: MovingAverage Constant Misconfiguration

**What goes wrong:** averaging_constant too high = sensitive to outliers; too low = ignores data.

**Why it happens:** User confusion about what the constant means. It's the "new information weight", not a smoothing factor in the traditional sense.

**How to avoid:**
- Use PyTorch's default of 0.01
- Document clearly: "higher = adapt faster but more sensitive to outliers"
- Provide presets: "conservative" (0.001), "balanced" (0.01), "responsive" (0.1)

### Pitfall 4: HistogramObserver Performance

**What goes wrong:** Calibration takes extremely long time (can be 10-100x slower than MinMax).

**Why it happens:** Histogram computation and KL divergence minimization are computationally expensive.

**Warning signs:**
- Calibration takes >10 seconds for 150 samples
- Memory usage spikes during calibration

**How to avoid:**
- Use HistogramObserver only when needed (outliers, non-uniform distributions)
- Limit histogram bins (default 2048 is reasonable)
- Document performance characteristics
- Provide progress indication for large calibration runs

### Pitfall 5: Accuracy Warnings Breaking CI/CD

**What goes wrong:** Quantization warnings cause test failures in automated pipelines.

**Why it happens:** Default `on_failure="error"` raises exceptions.

**How to avoid:**
- Default to `on_failure="warn"` (warning, not error)
- Provide explicit `accuracy_warning="ignore"` for CI/CD
- Document: "use `accuracy_warning="ignore"` for automated pipelines"

## Code Examples

### INT4 Group-Wise Quantization

```python
# Source: AWQ/GPTQ pattern via research
import torch
from typing import Tuple

def quantize_weight_int4(
    weight: torch.Tensor,
    group_size: int = 128,
    symmetric: bool = True,
    axis: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Quantize weight tensor to INT4 using group-wise scaling.

    Args:
        weight: Shape (out_features, in_features) for Linear layer
        group_size: Number of channels per group (default: 128)
        symmetric: Use symmetric quantization (recommended for INT4)
        axis: Channel axis (default: 0 for output channels)

    Returns:
        int4_weight: Quantized INT4 weights (packed int8 storage)
        scales: Per-group scale factors, shape (num_groups,)
        zero_points: Per-group zero points, shape (num_groups,)
    """
    out_features = weight.shape[axis]

    # Handle layers smaller than group_size
    if out_features < group_size:
        # Fallback to per-channel quantization
        return _quantize_int4_per_channel(weight, symmetric=symmetric, axis=axis)

    # Reshape for group-wise processing
    # (out_features, in_features) -> (num_groups, group_size, in_features)
    num_groups = out_features // group_size
    if out_features % group_size != 0:
        # Handle remainder by padding or adjusting group size
        # Implementation detail: use smaller groups for remainder
        num_groups = (out_features + group_size - 1) // group_size

    # Calculate per-group min/max
    # Implementation continues in PLAN.md...

    return int4_weight, scales, zero_points
```

### MovingAverageMinMaxObserver

```python
# Source: Based on PyTorch 2.8 MovingAverageMinMaxObserver
# https://docs.pytorch.org/docs/stable/generated/torch.ao.quantization.observer.MovingAverageMinMaxObserver.html

from typing import Optional
import torch

class MovingAverageMinMaxObserver:
    """
    Observer using exponential moving average of min/max values.

    Averaging formula:
        min_new = (1 - c) * min_old + c * min_current
        max_new = (1 - c) * max_old + c * max_current

    Where c is the averaging_constant.

    Args:
        averaging_constant: Weight for new observations (default: 0.01)
            - Lower (0.001): More stable, adapts slowly
            - Default (0.01): Balanced (PyTorch standard)
            - Higher (0.1): Adapts quickly, more sensitive to outliers
        dtype: Target quantization dtype

    Examples:
        >>> obs = MovingAverageMinMaxObserver(averaging_constant=0.01)
        >>> for data in calibration_batch:
        ...     obs.forward(data)
        >>> scale, zp = obs.calculate_qparams()
    """

    def __init__(
        self,
        averaging_constant: float = 0.01,
        dtype: torch.dtype = torch.qint8,
    ):
        if not 0 < averaging_constant <= 1:
            raise ValueError(
                f"averaging_constant must be in (0, 1], got {averaging_constant}"
            )
        self.averaging_constant = averaging_constant
        self.dtype = dtype
        self.min_val: Optional[float] = None
        self.max_val: Optional[float] = None

    def forward(self, x: torch.Tensor) -> None:
        """Update running min/max with exponential moving average."""
        x_min = x.amin().item()
        x_max = x.amax().item()

        if self.min_val is None:
            # First observation
            self.min_val = x_min
            self.max_val = x_max
        else:
            c = self.averaging_constant
            self.min_val = (1 - c) * self.min_val + c * x_min
            self.max_val = (1 - c) * self.max_val + c * x_max

    def calculate_qparams(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Calculate scale and zero-point from averaged min/max."""
        if self.min_val is None:
            raise RuntimeError("No data observed")

        qmin, qmax = -128, 127  # INT8 range
        range_val = self.max_val - self.min_val
        q_range = qmax - qmin
        scale = range_val / q_range
        scale = max(scale, 1e-8)  # Clamp to avoid division by zero

        zero_point = qmin - (self.min_val / scale)
        zero_point = int(round(zero_point))
        zero_point = max(qmin, min(qmax, zero_point))  # Clamp to valid range

        return torch.tensor(scale), torch.tensor(zero_point, dtype=torch.int32)

    def reset(self) -> None:
        """Reset observer state."""
        self.min_val = None
        self.max_val = None
```

### Layer Skipping with Unified API

```python
# Source: HuggingFace modules_to_not_convert pattern
from typing import List, Optional, Type, Union
import torch.nn as nn

LayerTypes = Union[Type[nn.Module], Tuple[Type[nn.Module], ...]]

def _get_layers_to_skip(
    model: nn.Module,
    modules_to_not_convert: Optional[List[str]] = None,
    skip_layer_types: Optional[LayerTypes] = None,
    skip_layer_names: Optional[List[str]] = None,
    skip_param_threshold: int = 0,
) -> set:
    """
    Build unified set of layers to skip during quantization.

    Args:
        model: PyTorch model
        modules_to_not_convert: Unified skip list (module paths)
        skip_layer_types: Layer types to skip (e.g., nn.Embedding)
        skip_layer_names: Layer names to skip
        skip_param_threshold: Skip layers with fewer parameters than this

    Returns:
        Set of layer names to skip

    Examples:
        >>> # Skip embeddings by type
        >>> skip_set = _get_layers_to_skip(
        ...     model, skip_layer_types=(nn.Embedding, nn.LayerNorm)
        ... )

        >>> # Skip specific layers by name
        >>> skip_set = _get_layers_to_skip(
        ...     model, modules_to_not_convert=["lm_head", "encoder.0"]
        ... )

        >>> # Skip small layers
        >>> skip_set = _get_layers_to_skip(
        ...     model, skip_param_threshold=512
        ... )
    """
    skip_set = set(modules_to_not_convert or [])

    # Add type-based skips
    if skip_layer_types is not None:
        if isinstance(skip_layer_types, type):
            skip_layer_types = (skip_layer_types,)
        for name, module in model.named_modules():
            if name and isinstance(module, skip_layer_types):
                skip_set.add(name)

    # Add name-based skips
    if skip_layer_names:
        skip_set.update(skip_layer_names)

    # Add small layer skips
    if skip_param_threshold > 0:
        for name, module in model.named_modules():
            if name:
                param_count = sum(p.numel() for p in module.parameters())
                if param_count < skip_param_threshold:
                    skip_set.add(name)

    return skip_set
```

### Accuracy Warning Check

```python
# Source: QuantuneV2 and Nikolic 2024 SQNR research
from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class QuantizationInfo:
    """Extended with warnings support."""
    selected_layers: List[str]
    skipped_layers: List[str]
    calibration_samples_used: int
    dtype: torch.dtype
    symmetric: bool
    sqnr_db: Optional[float] = None
    warnings: List[str] = field(default_factory=list)  # NEW
    # ... other fields ...

def check_accuracy_warnings(
    info: QuantizationInfo,
    sqnr_warning_threshold: float = 20.0,
    sqnr_error_threshold: float = 10.0,
    all_layers_quantized_warning: bool = True,
) -> QuantizationInfo:
    """
    Add accuracy warnings to QuantizationInfo based on quality metrics.

    SQNR thresholds (based on industry practice):
    - > 30 dB: Excellent quantization
    - 20-30 dB: Good quantization (warning below 20)
    - 10-20 dB: Marginal quantization (error below 10)
    - < 10 dB: Poor quantization - significant accuracy loss likely

    Args:
        info: QuantizationInfo to check and update
        sqnr_warning_threshold: SQNR below this triggers warning (dB)
        sqnr_error_threshold: SQNR below this triggers error (dB)
        all_layers_quantized_warning: Warn if all layers were quantized

    Returns:
        Updated QuantizationInfo with warnings populated
    """
    # Check SQNR
    if info.sqnr_db is not None:
        if info.sqnr_db < sqnr_error_threshold:
            info.warnings.append(
                f"CRITICAL: Very low SQNR ({info.sqnr_db:.2f} dB). "
                "Significant accuracy loss likely. Consider INT8 or "
                "skipping more layers."
            )
        elif info.sqnr_db < sqnr_warning_threshold:
            info.warnings.append(
                f"Warning: Low SQNR ({info.sqnr_db:.2f} dB). "
                "Quantization may impact accuracy. Verify with task-specific evaluation."
            )

    # Check if all layers were quantized (aggressive quantization)
    if all_layers_quantized_warning and not info.skipped_layers:
        info.warnings.append(
            "All layers were quantized. For INT4, consider skipping embeddings "
            "and normalization layers to preserve accuracy."
        )

    # Check calibration sample count
    if info.calibration_samples_used < 50:
        info.warnings.append(
            f"Low calibration sample count ({info.calibration_samples_used}). "
            "Recommend at least 100 samples for reliable calibration."
        )

    return info
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Per-tensor INT4 | Group-wise INT4 (group_size=128) | ~2022 (GPTQ) | INT4 became practical for LLMs |
| MinMax only | MinMax + MovingAverage + Histogram | 2023-2024 | Better outlier handling, robust calibration |
| Manual layer selection | modules_to_not_convert API | 2023-2024 | Standardized across HuggingFace, AWQ, GPTQ |
| No accuracy warnings | SQNR-based warnings | 2024-2025 (QuantuneV2) | Users get early feedback on quantization quality |

**Deprecated/outdated:**
- **torch.ao.quantization.observer**: PyTorch's built-in observers are deprecated for removal in 2.10+. Implement custom observers (as done in Phase 2).
- **Per-tensor INT4**: Results in poor accuracy. Always use group-wise for 4-bit precision.
- **bitsandbytes for non-LLM**: bitsandbytes is heavily optimized for transformers. For general model quantization, use model-agnostic approaches.

## Open Questions

### 1. Custom INT4 dtype vs. packed int8 storage

**What we know:**
- PyTorch 2.10 doesn't have native `torch.qint4`
- Most implementations use packed int8 storage (2 INT4 values per int8)
- AWQ/GPTQ use custom packed formats

**What's unclear:**
- Which packing format to use (row-major vs column-major)
- How to store group-wise scales/zero-points efficiently

**Recommendation:** Use packed int8 storage with metadata for scales/zp. Follow the established pattern from existing code (reuse safetensors format). Implement custom unpacking in QuantizedLinearInt4 module.

### 2. Observer auto-selection criteria

**What we know:**
- Auto-selection is complex and data-dependent
- Different observers work better for different distributions
- Research papers don't provide clear heuristics

**What's unclear:**
- Reliable dataset size thresholds for each observer
- How to detect "outliers" programmatically
- Distribution analysis metrics (skewness? kurtosis?)

**Recommendation:** Mark auto-selection as experimental. Implement simple heuristic based on dataset size and variance, but allow user override. Document clearly that results may vary.

### 3. Symmetric vs asymmetric for INT4

**What we know:**
- AWQ uses symmetric by default
- GPTQ supports both but symmetric is common
- Symmetric simplifies computation (zero_point = 0)

**What's unclear:**
- When asymmetric provides meaningful benefit for INT4
- Accuracy trade-off between symmetric and asymmetric at 4-bit

**Recommendation:** Default to symmetric quantization for INT4 (simpler, faster, good accuracy). Make asymmetric available as an option for advanced users.

## Sources

### Primary (HIGH confidence)

- [PyTorch MovingAverageMinMaxObserver Documentation](https://docs.pytorch.org/docs/stable/generated/torch.ao.quantization.observer.MovingAverageMinMaxObserver.html) - Official observer API and formulas
- [HuggingFace Transformers Quantization Documentation](https://huggingface.co/docs/transformers/main_classes/quantization) - AwqConfig, GPTQConfig, group_size=128, modules_to_not_convert API
- [PyTorch Quantization in Practice Blog](https://pytorch.org/blog/quantization-in-practice/) - Histogram observer and KL divergence minimization

### Secondary (MEDIUM confidence)

- [Dual Precision Quantization (arXiv, May 2025)](https://arxiv.org/pdf/2505.14638) - INT4 with group_size=128
- [AFPQ - Asymmetric Floating Point Quantization (ACL 2024)](https://aclanthology.org/2024.findings-acl.3.pdf) - Group size 128 as industry standard
- [AFPQ - Asymmetric Floating Point Quantization (ACL 2024)](https://aclanthology.org/2024.findings-acl.3.pdf) - Group size 128 commonly used
- [AFPQ - Asymmetric Floating Point Quantization (ACL 2024)](https://aclanthology.org/2024.findings-acl.3.pdf) - Group size 128 in Frantar 2022, Lin 2023
- [GWQ - Group-Wise Quantization (PMLR 2024)](https://proceedings.mlr.press/v222/yang24a/yang24a.pdf) - Group-wise quantization framework
- [Rethinking Channel Dimensions (ICLR 2024)](https://proceedings.iclr.cc/paper_files/2024/file/374050dc3f211267bd6bf0ea24eae184-Paper-Conference.pdf) - Group size 128 accuracy/computation trade-off
- [QuantuneV2: Compiler-Based Metric-Driven Quantization (arXiv 2025)](https://arxiv.org/pdf/2501.07161) - SQNR as local metric for quantization quality
- [Bit Rate Allocation, SQNR and Accuracy Degradation (J. Electrical Engineering 2024)](https://reference-global.com/2/v2/download/article/10.2478/jee-2024-0051.pdf) - SQNR-accuracy relationship
- [Mixed-Precision Quantization for Language Models (arXiv 2025)](https://arxiv.org/html/2510.16805v1) - AWQ techniques for embeddings
- [Pushing LLMs to the Edge: Structured Compression (ETH Zurich 2024)](https://www.research-collection.ethz.ch/bitstreams/12b56a18-30ed-4c9b-b0a8-642a92d0d10a/download) - Embedding/head layer quantization

### Tertiary (LOW confidence)

- Various blog posts and tutorials on quantization techniques (verified against primary sources where applicable)

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - Based on official PyTorch and HuggingFace documentation
- Architecture: HIGH - Patterns from verified research papers and official docs
- Pitfalls: MEDIUM - Some edge cases (small layers, observer auto-selection) lack definitive guidance
- Code examples: HIGH - Based on official documentation and established research patterns

**Research date:** 2026-02-03
**Valid until:** 30 days (quantization landscape is evolving rapidly, especially for INT4; verify HuggingFace and PyTorch docs for any updates before implementation)

---

*Phase: 03-advanced-calibration-&-int4*
*Research: 2026-02-03*
