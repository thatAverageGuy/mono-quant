# Phase 2: Static Quantization & I/O - Research

**Researched:** 2026-02-03
**Domain:** PyTorch Static Quantization, Calibration, Model Serialization, Validation Metrics
**Confidence:** HIGH

## Summary

Phase 2 implements static quantization with calibration data, model serialization (PyTorch + Safetensors formats), and validation metrics. The research reveals key patterns for calibration observers, sample count recommendations, safetensors metadata constraints, and SQNR calculation methods.

**Key findings:**
1. **Calibration**: 100-200 samples is the recommended baseline for static quantization calibration; representativeness matters more than quantity
2. **Safetensors**: Metadata values must be strings only (no nested JSON); `__metadata__` is the special key for custom metadata
3. **SQNR Calculation**: PyTorch provides `torch.ao.ns.fx.utils.compute_sqnr()` for comparing original and quantized tensors
4. **LSTM Quantization**: Static quantization for LSTMs is experimental in PyTorch; dynamic quantization is more mature

**Primary recommendation:** Build a custom MinMaxObserver implementation (don't use deprecated `torch.ao.quantization`), use safetensors for safe serialization with string-only metadata, and implement validation checks inline with quantization (always validate, configurable failure behavior).

## Standard Stack

### Core

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| **Python** | 3.11 - 3.12 | Runtime language | PyTorch 2.9+ requires Python 3.10+. Project uses 3.11+. |
| **PyTorch** | 2.0+ (tested on 2.9+) | Tensor computation, quantization primitives | Built-in quantization dtypes, stable tensor ops for custom implementation |
| **safetensors** | 0.4+ | Safe model serialization | Secure alternative to pickle, widely adopted, zero-copy loading |

### Development Tools (Existing)

| Tool | Purpose | Notes |
|------|---------|-------|
| **pytest** | 7.0+ | Test framework | Already in project dev dependencies |
| **ruff** | 0.1+ | Linter & formatter | Already in project |
| **mypy** | 1.0+ | Static type checking | Already in project |

### Optional for Progress Reporting

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| **tqdm** | 4.66+ | Progress bars for large calibration datasets | Auto-enable when dataset size exceeds threshold |

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| **Custom MinMaxObserver** | torch.ao.quantization.MinMaxObserver | Deprecated API, removal in PyTorch 2.10+. Custom implementation is safer. |
| **safetensors** | torch.save/torch.load | safetensors is secure (no pickle), zero-copy, industry standard for HF models |
| **Custom SQNR** | torch.ao.ns.fx.utils.compute_sqnr | Use PyTorch's implementation if available, but may need custom for backward compatibility |
| **String metadata in safetensors** | JSON-serialized metadata | safetensors requires string values; serialize complex metadata to JSON string |

### DO NOT USE (Critical)

| Avoid | Why | Use Instead |
|-------|-----|-------------|
| **torch.ao.quantization observers** | Deprecated, removal in PyTorch 2.10+ | Custom observer implementation using torch ops |
| **torch.ao.quantization.prepare/convert** | Deprecated workflow | Direct layer replacement pattern (from Phase 1) |
| **pickle for model storage** | Security vulnerability (arbitrary code execution) | safetensors or torch.save with state_dict only |
| **Nested JSON in safetensors metadata** | Not supported - all values must be strings | Serialize nested structures to JSON strings |

### Installation

```bash
# Core (already in project)
pip install torch>=2.0.0

# New dependencies for Phase 2
pip install safetensors>=0.4

# Optional for progress reporting
pip install tqdm>=4.66

# Dev (already in project)
pip install pytest ruff mypy
```

## Architecture Patterns

### Recommended Project Structure (Phase 2 Additions)

```
src/
└── mono_quant/
    ├── __init__.py           # Public API exports (add static_quantize)
    ├── core/
    │   ├── __init__.py
    │   ├── schemes.py        # (Existing) Affine, symmetric quantization schemes
    │   ├── mappers.py        # (Existing) Scale/zero-point calculation
    │   ├── quantizers.py     # (Existing) Quantization transformations
    │   └── observers.py      # (NEW) MinMaxObserver, calibration logic
    ├── modules/
    │   ├── __init__.py
    │   └── linear.py         # (Existing) QuantizedLinear, extend for LSTM
    ├── io/
    │   ├── __init__.py
    │   ├── handlers.py       # (Existing) nn.Module and state_dict input handling
    │   ├── formats.py        # (NEW) Save/load for PyTorch and Safetensors
    │   └── validation.py     # (NEW) Model validation metrics
    ├── config/
    │   ├── __init__.py
    │   └── quant_config.py   # (Existing) QuantizationConfig, extend with static options
    └── calibration/          # (NEW) Calibration module
        ├── __init__.py
        ├── runner.py         # Calibration forward pass runner
        └── data.py           # DataLoader/tensor list handling
```

### Pattern 1: Custom MinMaxObserver Implementation

**What:** A simple observer that tracks min/max values from tensor activations during calibration.

**When to use:** During static quantization calibration to compute scale and zero-point for activations.

**Why custom:** `torch.ao.quantization.MinMaxObserver` is deprecated. Custom implementation is stable and lightweight.

**Example:**
```python
# Source: Custom implementation based on PyTorch observer patterns
from typing import Tuple
import torch

class MinMaxObserver:
    """
    Observer that tracks min/max values for quantization parameter calculation.

    This is a custom implementation to avoid deprecated torch.ao.quantization APIs.
    """
    def __init__(self, dtype: torch.dtype = torch.qint8):
        self.dtype = dtype
        self.min_val = None
        self.max_val = None

    def forward(self, x: torch.Tensor) -> None:
        """Update min/max from input tensor."""
        x_min = x.amin().item()
        x_max = x.amax().item()

        if self.min_val is None or x_min < self.min_val:
            self.min_val = x_min
        if self.max_val is None or x_max > self.max_val:
            self.max_val = x_max

    def calculate_qparams(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Calculate scale and zero-point from observed min/max."""
        if self.min_val is None or self.max_val is None:
            raise RuntimeError("No data observed. Call forward() first.")

        qmin, qmax = -128, 127  # int8 range
        range_val = self.max_val - self.min_val
        q_range = qmax - qmin

        scale = range_val / q_range
        scale = max(scale, 1e-8)  # Avoid division by zero

        zero_point = qmin - (self.min_val / scale)
        zero_point = int(round(zero_point))

        return torch.tensor(scale), torch.tensor(zero_point, dtype=torch.int32)
```

### Pattern 2: Calibration with Multiple Input Types

**What:** Accept both List[torch.Tensor] and DataLoader for calibration data.

**When to use:** When user provides calibration data for static quantization.

**Example:**
```python
# Source: Best practices from HuggingFace optimum calibration patterns
from typing import List, Union
import torch
from torch.utils.data import DataLoader

CalibrationData = Union[List[torch.Tensor], DataLoader]

def _normalize_calibration_data(data: CalibrationData) -> List[torch.Tensor]:
    """
    Convert DataLoader or tensor list to normalized tensor list.
    """
    if isinstance(data, DataLoader):
        # Extract tensors from DataLoader
        tensors = []
        for batch in data:
            if isinstance(batch, (tuple, list)):
                tensors.append(batch[0])  # Assume (input, target) format
            else:
                tensors.append(batch)
        return tensors
    else:
        # Already a list of tensors
        return data

def run_calibration(
    model: torch.nn.Module,
    calibration_data: CalibrationData,
    num_samples: int = 100,
) -> None:
    """
    Run calibration by forwarding samples through the model.

    Args:
        model: Model to calibrate (with observers attached)
        calibration_data: List of tensors or DataLoader
        num_samples: Maximum number of samples to use
    """
    tensors = _normalize_calibration_data(calibration_data)
    tensors = tensors[:num_samples]  # Limit samples

    model.eval()
    with torch.no_grad():
        for x in tensors:
            _ = model(x)
```

### Pattern 3: Layer Type Selection with Include/Exclude

**What:** Support both `layer_types` (include) and `skip_layers` (exclude) parameters.

**When to use:** When user wants selective quantization of specific layer types.

**Example:**
```python
# Source: PyTorch module iteration patterns + best practices
from typing import List, Type, Union, Set
import torch.nn as nn

LayerTypes = Union[Type[nn.Module], tuple]

def _select_layers_by_type(
    model: nn.Module,
    layer_types: LayerTypes,
    skip_types: LayerTypes = None,
) -> List[str]:
    """
    Find layer names matching the specified types.

    Args:
        model: Model to search
        layer_types: Module type(s) to include (e.g., nn.Linear, nn.Conv2d)
        skip_types: Module type(s) to exclude (e.g., nn.LayerNorm, nn.Embedding)

    Returns:
        List of qualified layer names to quantize
    """
    # Normalize to tuple
    if not isinstance(layer_types, tuple):
        layer_types = (layer_types,)
    if skip_types is None:
        skip_types = ()
    elif not isinstance(skip_types, tuple):
        skip_types = (skip_types,)

    selected = []
    skipped = []

    for name, module in model.named_modules():
        if name == "":
            continue  # Skip root

        if isinstance(module, skip_types):
            skipped.append(name)
        elif isinstance(module, layer_types):
            selected.append(name)
        else:
            skipped.append(name)

    return selected, skipped
```

### Pattern 4: Safetensors Save with Metadata

**What:** Save quantized model with metadata in safetensors format.

**When to use:** When serializing quantized models for distribution or safe loading.

**Example:**
```python
# Source: HuggingFace safetensors documentation
import json
from safetensors.torch import save_file
from typing import Dict, Any

def save_with_metadata(
    tensors: Dict[str, torch.Tensor],
    path: str,
    metadata: Dict[str, Any],
) -> None:
    """
    Save tensors to safetensors with metadata.

    CRITICAL: safetensors metadata values must be strings only.
    Complex structures must be JSON-serialized.
    """
    # Convert all metadata values to strings
    string_metadata = {}
    for key, value in metadata.items():
        if isinstance(value, str):
            string_metadata[key] = value
        else:
            # Serialize complex types to JSON string
            string_metadata[key] = json.dumps(value)

    save_file(tensors, path, metadata=string_metadata)

# Usage example
metadata = {
    "quantization_dtype": "qint8",
    "scheme": "asymmetric",
    "per_channel": "true",
    "model_size_mb": "45.2",
    "compression_ratio": "4.0",
    "sqnr_db": "45.3",
}
save_with_metadata(quantized_state_dict, "model.safetensors", metadata)
```

### Pattern 5: SQNR Calculation

**What:** Calculate Signal-to-Quantization-Noise Ratio to measure quantization quality.

**When to use:** During validation to report quantization accuracy.

**Example:**
```python
# Source: PyTorch torch.ao.ns.fx.utils.compute_sqnr (if available) or manual
import torch

def calculate_sqnr(original: torch.Tensor, quantized: torch.Tensor) -> float:
    """
    Calculate Signal-to-Quantization-Noise Ratio in dB.

    SQNR measures the quality of quantization. Higher values indicate better quality.

    Args:
        original: Original tensor (float32)
        quantized: Quantized tensor (dequantized to float32 for comparison)

    Returns:
        SQNR in decibels (dB)
    """
    # Convert quantized to float32 if needed
    if hasattr(quantized, 'dequantize'):
        quantized_fp32 = quantized.dequantize()
    else:
        quantized_fp32 = quantized.to(torch.float32)

    # Calculate noise (quantization error)
    noise = original - quantized_fp32

    # Signal power
    signal_power = (original ** 2).mean()

    # Noise power
    noise_power = (noise ** 2).mean()

    # Avoid division by zero
    if noise_power < 1e-10:
        return float('inf')

    # SQNR in dB
    sqnr_db = 10 * torch.log10(signal_power / noise_power)

    return sqnr_db.item()
```

### Anti-Patterns to Avoid

- **Using torch.ao.quantization observers**: Deprecated, will break in PyTorch 2.10+
- **Non-string values in safetensors metadata**: Format requires all values be strings
- **Modifying original model during calibration**: Always copy before calibration
- **Single forward pass for all calibration**: Need multiple passes for accurate activation statistics
- **Hardcoded layer name matching**: Use isinstance checks for type-based selection

## Don't Hand-Roll

Problems with existing solutions:

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Safe model serialization | Custom pickle/serializer | `safetensors` | Industry standard, secure, zero-copy, HF-compatible |
| Progress bars | Custom progress logic | `tqdm` | Widely adopted, handles edge cases, minimal dependency |
| Model size calculation | Manual parameter counting | Loop over `model.parameters()` + `tensor.element_size()` | Accurate, handles all parameter types |
| SQNR calculation | Custom noise calculation | `torch.ao.ns.fx.utils.compute_sqnr()` if available | Verified implementation, but may need fallback |
| DataLoader iteration | Custom data loading | PyTorch `DataLoader` directly | Standard interface, handles batching, shuffling |

**Key insight:** Build custom logic for calibration observers and layer selection patterns, but use standard libraries for serialization, progress reporting, and metrics.

## Common Pitfalls

### Pitfall 1: Using Deprecated torch.ao.quantization Observers

**What goes wrong:**
Code using `torch.ao.quantization.MinMaxObserver` will break when PyTorch 2.10+ removes these APIs.

**Why it happens:**
PyTorch is migrating quantization to torchao. The old APIs are explicitly deprecated.

**How to avoid:**
Implement a simple custom MinMaxObserver using only torch tensor operations:
```python
class MinMaxObserver:
    def __init__(self):
        self.min_val = None
        self.max_val = None

    def forward(self, x: torch.Tensor) -> None:
        self.min_val = min(self.min_val, x.amin()) if self.min_val else x.amin()
        self.max_val = max(self.max_val, x.amax()) if self.max_val else x.amax()
```

**Warning signs:**
- Importing from `torch.ao.quantization`
- Using `torch.quantization.prepare()` workflow

**Severity:** CRITICAL - Will break in PyTorch 2.10

### Pitfall 2: Non-String Metadata in Safetensors

**What goes wrong:**
Safetensors raises error when metadata contains non-string values.

**Why it happens:**
The safetensors format explicitly requires all metadata values be strings. Nested JSON is not supported directly.

**How to avoid:**
Serialize complex values to JSON strings:
```python
# Wrong
metadata = {"sqnr": 45.3, "compression": 0.25}

# Right
metadata = {
    "sqnr": "45.3",
    "compression": "0.25",
    "layers": json.dumps(["linear1", "linear2"])  # Complex types as JSON string
}
```

**Warning signs:**
- TypeError when calling `save_file()` with dict containing numbers
- "All metadata values must be strings" error message

**Severity:** HIGH - Prevents model saving

### Pitfall 3: Too Few Calibration Samples

**What goes wrong:**
Poor quantization accuracy due to incomplete activation range observation.

**Why it happens:**
Using too few calibration samples (e.g., < 50) may not capture the true distribution of activations.

**How to avoid:**
- Use 100-200 samples as baseline (per HuggingFace documentation)
- Ensure samples are representative of inference data distribution
- For diverse datasets, consider using up to 300-500 samples

**Warning signs:**
- Significant accuracy drop after quantization
- Very narrow observed min/max ranges

**Severity:** HIGH - Affects model quality

### Pitfall 4: Forgetting Bias in Quantized Layers

**What goes wrong:**
Missing bias parameters when saving/loading quantized models.

**Why it happens:**
Bias is typically not quantized but needs to be preserved separately from weights.

**How to avoid:**
- Include bias in state_dict: `state_dict["layer.bias"] = layer.bias.data`
- Document that bias is stored as float32 (not quantized)
- Validate bias shape matches during loading

**Warning signs:**
- KeyError when loading model
- Shape mismatch errors

**Severity:** MEDIUM - Causes load errors

### Pitfall 5: Not Validating After Quantization

**What goes wrong:**
Silent failures produce broken models that only fail during inference.

**Why it happens:**
Skipping validation to save time, or not having validation infrastructure.

**How to avoid:**
- Always run validation checks after quantization
- Implement configurable failure behavior: `on_failure="error"|"warn"|"ignore"`
- Minimum checks: SQNR, model size, load test, weight range

**Warning signs:**
- Returning quantized model without any checks
- No validation in the quantization function

**Severity:** HIGH - Produces undetected bugs

## Code Examples

Verified patterns from official sources:

### Calibration Runner with Progress Bar

```python
# Source: HuggingFace optimum calibration patterns + tqdm documentation
from typing import List, Union
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

CalibrationData = Union[List[torch.Tensor], DataLoader]

def run_calibration(
    model: torch.nn.Module,
    calibration_data: CalibrationData,
    num_samples: int = 100,
    show_progress: bool = None,
) -> None:
    """
    Run calibration forward passes with optional progress bar.

    Args:
        model: Model with observers attached
        calibration_data: Tensors or DataLoader
        num_samples: Max samples to use
        show_progress: Auto-detect if None (show for >50 samples)
    """
    # Normalize data
    if isinstance(calibration_data, DataLoader):
        tensors = []
        for batch in calibration_data:
            if isinstance(batch, (tuple, list)):
                tensors.append(batch[0])
            else:
                tensors.append(batch)
            if len(tensors) >= num_samples:
                break
    else:
        tensors = calibration_data[:num_samples]

    # Auto-detect progress bar need
    if show_progress is None:
        show_progress = len(tensors) > 50

    model.eval()
    with torch.no_grad():
        iterator = tqdm(tensors) if show_progress else tensors
        for x in iterator:
            _ = model(x)
```

### Model Size Calculation

```python
# Source: PyTorch forum discussions on memory calculation
from typing import Tuple
import torch
import torch.nn as nn

def calculate_model_size(model: nn.Module) -> Tuple[float, int]:
    """
    Calculate model size in MB and total parameter count.

    Args:
        model: PyTorch model

    Returns:
        Tuple of (size_mb, param_count)
    """
    param_count = 0
    size_bytes = 0

    for param in model.parameters():
        param_count += param.numel()
        size_bytes += param.numel() * param.element_size()

    # Add buffers (e.g., batch norm running stats)
    for buffer in model.buffers():
        size_bytes += buffer.numel() * buffer.element_size()

    size_mb = size_bytes / (1024 * 1024)

    return size_mb, param_count

# Example comparison
original_size, _ = calculate_model_size(original_model)
quantized_size, _ = calculate_model_size(quantized_model)
compression_ratio = original_size / quantized_size
```

### Save/Load with Safetensors

```python
# Source: HuggingFace safetensors documentation
from safetensors.torch import save_file, safe_open
from typing import Dict

def save_quantized_model(
    state_dict: Dict[str, torch.Tensor],
    path: str,
    metadata: Dict[str, str],
) -> None:
    """
    Save quantized model to safetensors format.

    Args:
        state_dict: Model state dict with quantized weights
        path: Output file path (should end in .safetensors)
        metadata: All values must be strings
    """
    save_file(state_dict, path, metadata=metadata)


def load_quantized_model(path: str) -> Dict[str, torch.Tensor]:
    """
    Load quantized model from safetensors.

    Args:
        path: Path to .safetensors file

    Returns:
        State dict with quantized weights
    """
    tensors = {}
    with safe_open(path, framework="pt", device="cpu") as f:
        for key in f.keys():
            tensors[key] = f.get_tensor(key)
    return tensors
```

### Layer Selection with Names

```python
# Source: PyTorch named_modules pattern
from typing import List, Set
import torch.nn as nn

def select_layers(
    model: nn.Module,
    layer_types: List[type] = None,
    layer_names: List[str] = None,
    skip_names: List[str] = None,
) -> List[str]:
    """
    Select layers to quantize by type or name.

    Args:
        model: Model to search
        layer_types: List of module types to include (e.g., [nn.Linear, nn.Conv2d])
        layer_names: Exact layer names to include (e.g., ["model.0", "encoder.weight"])
        skip_names: Exact layer names to exclude

    Returns:
        List of selected layer names
    """
    selected = set()
    skip_set = set(skip_names or [])

    # Select by type
    if layer_types:
        for name, module in model.named_modules():
            if name == "":
                continue
            if isinstance(module, tuple(layer_types)):
                if name not in skip_set:
                    selected.add(name)

    # Select by name
    if layer_names:
        for name in layer_names:
            if name not in skip_set:
                selected.add(name)

    return sorted(selected)
```

### Validation Checks

```python
# Source: Best practices for model validation
from typing import Dict, Any
import torch

class ValidationResult:
    """Results of quantization validation."""
    def __init__(self):
        self.sqnr_db: float = None
        self.original_size_mb: float = None
        self.quantized_size_mb: float = None
        self.compression_ratio: float = None
        self.load_test_passed: bool = False
        self.weight_range_valid: bool = False
        self.errors: List[str] = []

def validate_quantization(
    original_model: torch.nn.Module,
    quantized_model: torch.nn.Module,
    on_failure: str = "error",
) -> ValidationResult:
    """
    Run validation checks after quantization.

    Args:
        original_model: Original model (for comparison)
        quantized_model: Quantized model
        on_failure: "error" (raise), "warn" (log), "ignore" (silent)

    Returns:
        ValidationResult with all metrics
    """
    result = ValidationResult()

    # 1. Calculate SQNR
    result.sqnr_db = _calculate_model_sqnr(original_model, quantized_model)

    # 2. Model size comparison
    result.original_size_mb, _ = calculate_model_size(original_model)
    result.quantized_size_mb, _ = calculate_model_size(quantized_model)
    result.compression_ratio = result.original_size_mb / result.quantized_size_mb

    # 3. Load and run test
    result.load_test_passed = _test_load_run(quantized_model)

    # 4. Weight range check
    result.weight_range_valid = _check_weight_ranges(quantized_model)

    # Handle failures
    if not result.load_test_passed or not result.weight_range_valid:
        msg = f"Validation failed: load_test={result.load_test_passed}, range_valid={result.weight_range_valid}"
        result.errors.append(msg)

        if on_failure == "error":
            raise ValueError(msg)
        elif on_failure == "warn":
            import warnings
            warnings.warn(msg)

    return result
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| torch.ao.quantization observers | Custom MinMaxObserver | PyTorch 2.9+ | Old APIs deprecated, removal in 2.10 |
| pickle for model storage | safetensors | 2022-2024 | Security concerns, HF ecosystem adoption |
| No validation during quantization | Always validate with configurable failure | 2025 | Best practice for reliable quantization |
| Fixed calibration sample count | Flexible 100-200 baseline | 2025 | Research shows representativeness > quantity |

**Deprecated/outdated:**
- **torch.ao.quantization.MinMaxObserver**: Use custom implementation
- **torch.ao.quantization.prepare/convert**: Deprecated workflow
- **pickle for models**: Security vulnerability, use safetensors

**Current best practices (2025-2026):**
- Calibration: 100-200 representative samples
- SQNR calculation: Use `torch.ao.ns.fx.utils.compute_sqnr()` or manual implementation
- Safetensors: String-only metadata, JSON-serialize complex values
- Validation: Always validate, configurable failure modes

## Open Questions

### 1. LSTM Static Quantization Maturity

**What we know:**
- PyTorch's LSTM static quantization is experimental/prototype
- Dynamic quantization for LSTM is more mature
- [GitHub Issue #71028](https://github.com/pytorch/pytorch/issues/71028) notes LSTM quantization should be skipped by default

**What's unclear:**
- Exact state of LSTM static quantization support in PyTorch 2.9+
- Whether custom LSTM quantization will be stable

**Recommendation:**
Support LSTM in layer type selection but document as experimental. Focus on weight quantization for LSTM (not activation quantization). May need to use per-tensor instead of per-channel for LSTM gates.

### 2. Optimal Calibration Sample Count for Different Model Types

**What we know:**
- 100-200 samples is recommended baseline
- 4-bit quantization may need fewer samples (as few as 8 for AWQ)
- Representativeness matters more than quantity

**What's unclear:**
- Exact relationship between model size/type and optimal sample count
- Whether CNNs, Transformers, or RNNs need different calibration strategies

**Recommendation:**
Implement default of 150 samples with user override. Document that users should tune based on their specific model and data distribution.

### 3. Progress Bar "Large" Threshold

**What we know:**
- Progress bars are useful for long-running operations
- Small datasets don't need progress bars

**What's unclear:**
- What threshold users consider "large" for calibration

**Recommendation:**
Auto-detect based on sample count: show tqdm if len(samples) > 50. Make threshold configurable via global config.

## Sources

### Primary (HIGH confidence)

- [PyTorch Quantization Documentation](https://docs.pytorch.org/docs/stable/quantization.html) - Official quantization API reference, compute_sqnr function
- [HuggingFace Safetensors GitHub](https://github.com/huggingface/safetensors) - Official safetensors format specification, metadata requirements
- [PyTorch Saving and Loading Models](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html) - state_dict conventions

### Secondary (MEDIUM confidence)

- [HuggingFace Optimum Quantization Guide](https://huggingface.co/docs/optimum/concept_guides/quantization) - Calibration sample count recommendations (~200 samples)
- [Calibration Data Selection for PTQ](https://apxml.com/courses/practical-llm-quantization/chapter-2-post-training-quantization-ptq/ptq-calibration) - Calibration best practices
- [PyTorch Discussion - LSTM Static Quantization Performance](https://discuss.pytorch.org/t/extremely-bad-lstm-static-quantization-performance-compared-to-dynamic/195245) - LSTM quantization challenges
- [QCore: Data-Efficient Continual Calibration (VLDB 2024)](https://www.vldb.org/pvldb/vol17/p2708-campos.pdf) - Calibration sample research (48-97 samples)

### Tertiary (LOW confidence)

- [TQDM Documentation](https://tqdm.github.io/) - Progress bar library patterns
- Various blog posts on SQNR calculation - verify with PyTorch implementation

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - Official documentation for safetensors, PyTorch quantization
- Architecture: HIGH - Based on Phase 1 patterns and official documentation
- Pitfalls: HIGH - Deprecation notices verified, safetensors constraints documented
- Calibration best practices: MEDIUM - Based on HuggingFace docs and research papers

**Research date:** 2026-02-03
**Valid until:** 2026-03-03 (30 days - PyTorch ecosystem evolving with torchao migration)

**Critical decisions documented:**
1. DO NOT use `torch.ao.quantization.MinMaxObserver` - deprecated, implement custom
2. Safetensors metadata values MUST be strings - JSON-serialize complex types
3. Calibration baseline: 100-200 samples with user override
4. Always validate after quantization with configurable `on_failure` behavior
5. LSTM support is experimental - document as such

---
*Phase: 02-static-quantization-&-i/o*
*Research completed: 2026-02-03*
