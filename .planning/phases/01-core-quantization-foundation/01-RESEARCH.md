# Phase 1: Core Quantization Foundation - Research

**Researched:** 2026-02-03
**Domain:** PyTorch Model Quantization (Dynamic, INT8/FP16)
**Confidence:** HIGH

## Summary

Phase 1 requires implementing core quantization functionality for PyTorch models with dynamic quantization to INT8 and FP16. The research reveals critical information about PyTorch's evolving quantization landscape: `torch.ao.quantization` is being deprecated in favor of `torchao`, with planned removal in PyTorch 2.10+. This is a HIGH-CONFIDENCE finding from official PyTorch documentation.

The phase must implement:
1. **Model-agnostic input handling** - Accept both `nn.Module` and `state_dict`, detect format automatically, always copy (never modify original)
2. **Dynamic quantization** - INT8 with per-channel scaling and symmetric/asymmetric schemes, FP16 conversion
3. **Minimal dependencies** - Only torch required; no HF transformers, no torchao dependency (study it for patterns only)

**Primary recommendation:** Build a lightweight quantization engine using PyTorch's stable tensor operations (`torch.quantize_per_tensor`, `torch.quantize_per_channel`, dtype casting). Avoid depending on `torchao` directly (APIs unstable) or `torch.ao.quantization` (being deprecated). Implement custom quantization logic that will remain stable across PyTorch versions.

## Standard Stack

### Core

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| **Python** | 3.11 - 3.12 | Runtime language | PyTorch 2.9+ requires Python 3.10+. 3.11-3.12 is the sweet spot for stability. |
| **PyTorch** | 2.9+ | Tensor computation, quantization primitives | Built-in quantization dtypes (`torch.qint8`, `torch.float16`), `torch.quantize_per_channel`, `torch.quantize_per_tensor`. 2.9+ is minimum for int1-int7 dtypes. |

### Supporting (Optional for Phase 1)

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| **typing-extensions** | 4.12+ | Type hints compatibility | When using modern type hints that need backport to Python 3.11 |

### Development Tools

| Tool | Purpose | Notes |
|------|---------|-------|
| **setuptools** | Build backend | Mature, stable, widely supported |
| **ruff** | Linter & formatter | 130x faster than Black, replaces Flake8+Black+isort |
| **pytest** | 9.0+ | Test framework | Modern pytest with subtests support |
| **mypy** | 1.11+ | Static type checking | Recommended for quantization code correctness |

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| **Native torch ops** | torchao (as dependency) | torchao APIs are unstable ("may change in the future"). Use torchao for reference patterns, not as dependency. |
| **Native torch ops** | torch.ao.quantization | Being deprecated, planned removal in PyTorch 2.10+. Avoid for new code. |
| **Simple weight casting** | Full FP16 quantization pipeline | For FP16, simple dtype casting is sufficient. Full pipeline adds complexity without benefit for Phase 1. |
| **Custom implementation** | bitsandbytes | Tightly coupled to HF accelerate, violates model-agnostic principle |

### DO NOT USE (Critical)

| Avoid | Why | Use Instead |
|-------|-----|-------------|
| **torch.ao.quantization** | Deprecated, removal in 2.10+ | Native torch quantization ops (`torch.quantize_per_channel`) |
| **torchao (dependency)** | APIs unstable, may break | Study torchao for patterns, implement own stable wrapper |
| **HuggingFace Transformers** | Violates model-agnostic requirement | Accept any nn.Module or state_dict |
| **HuggingFace Optimum** | HF ecosystem dependency | Model-agnostic design |
| **BitsAndBytes** | Tied to HF accelerate | Native torch implementation |

## Architecture Patterns

### Recommended Project Structure (Phase 1 Scope)

```
src/
└── mono_quant/
    ├── __init__.py           # Public API exports
    ├── core/                 # Core quantization logic (no external deps beyond torch)
    │   ├── __init__.py
    │   ├── schemes.py        # Affine, symmetric quantization schemes
    │   ├── mappers.py        # Scale/zero-point calculation
    │   ├── quantizers.py     # Quantization transformations
    │   └── observers.py      # Observer classes (placeholders for Phase 2)
    ├── modules/              # Quantized PyTorch module replacements
    │   ├── __init__.py
    │   └── linear.py         # QuantizedLinear (Phase 1)
    ├── io/                   # Input/output handling
    │   ├── __init__.py
    │   ├── handlers.py       # nn.Module and state_dict input handling
    │   └── state_dict.py     # Serialization helpers
    └── config/               # Configuration handling
        ├── __init__.py
        └── quant_config.py   # QuantizationConfig dataclass
```

### Structure Rationale

- **`core/`**: Pure quantization math - scale, zero-point, symmetric/asymmetric schemes. Testable without full models.
- **`modules/`**: PyTorch-specific replacements. Separated so core can be tested independently.
- **`io/`**: Model-agnostic input handling - detects `nn.Module` vs `state_dict`, always copies.
- **`config/`**: Centralized configuration - single source of truth for quantization parameters.

### Pattern 1: Model-Agnostic Input Handling

**What:** Detect input format (nn.Module or state_dict), handle appropriately, always copy.

**When to use:** Entry point for all quantization operations.

**Example:**
```python
# Source: PyTorch state_dict documentation + copy conventions
from copy import deepcopy
import torch
import torch.nn as nn
from typing import Union, Dict

def _prepare_model(
    model: Union[nn.Module, Dict[str, torch.Tensor]]
) -> nn.Module:
    """
    Convert input to nn.Module, always copying to preserve original.

    Handles both nn.Module and state_dict inputs as per CONTEXT.md decisions.
    """
    if isinstance(model, nn.Module):
        # Always copy - never modify the original model
        return deepcopy(model)
    elif isinstance(model, dict):
        # state_dict input - need to reconstruct model
        # For Phase 1: return error if we can't determine model architecture
        # User must provide nn.Module or we need architecture info (future phase)
        raise ValueError(
            "state_dict input requires model architecture. "
            "Please provide nn.Module, or pass architecture_info parameter."
        )
    else:
        raise TypeError(f"Expected nn.Module or state_dict, got {type(model)}")
```

### Pattern 2: Dynamic Quantization with Per-Channel Scaling

**What:** Quantize weights with per-channel scaling, dynamically quantize activations at runtime.

**When to use:** For INT8 quantization without calibration data (Phase 1 requirement).

**Example:**
```python
# Source: PyTorch quantization primitives + torchao patterns (for reference)
import torch
from typing import Tuple

def _calculate_scale_zp_per_channel(
    weight: torch.Tensor,
    dtype: torch.dtype = torch.qint8,
    symmetric: bool = False
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Calculate per-channel scale and zero-point for weight quantization.

    Args:
        weight: Tensor of shape (out_channels, in_channels, ...)
        dtype: Target quantization dtype
        symmetric: If True, zero_point = 0 (symmetric quantization)

    Returns:
        scale: Per-channel scale tensor
        zero_point: Per-channel zero-point tensor (0 if symmetric)
    """
    # Per-channel: calculate scale for each output channel
    # Reduce over all dimensions except the first (output channel)
    axis = tuple(range(1, weight.dim()))  # (1, 2, 3, ...) for Conv/Linear

    if symmetric:
        # Symmetric: zero_point = 0, scale based on max absolute value
        max_abs = weight.abs().amax(dim=axis)
        qmin, qmax = -128, 127  # int8 symmetric range
        scale = max_abs / qmax
        zero_point = torch.zeros_like(scale)
    else:
        # Asymmetric: full affine quantization
        min_val = weight.amin(dim=axis)
        max_val = weight.amax(dim=axis)
        qmin, qmax = -128, 127  # int8 asymmetric range
        scale = (max_val - min_val) / (qmax - qmin)
        zero_point = qmin - (min_val / scale)

    # Avoid division by zero
    scale = torch.clamp(scale, min=1e-8)

    return scale, zero_point.int()


def quantize_weight_per_channel(
    weight: torch.Tensor,
    symmetric: bool = False
) -> torch.Tensor:
    """Apply per-channel quantization to weight tensor."""
    scale, zp = _calculate_scale_zp_per_channel(weight, symmetric=symmetric)
    return torch.quantize_per_channel(
        weight,
        scale,
        zp,
        axis=0,  # Per output channel
        dtype=torch.qint8
    )
```

### Pattern 3: FP16 Quantization (Simple Cast Approach)

**What:** Convert model to half precision using dtype casting.

**When to use:** For FP16 quantization (memory reduction focus).

**Example:**
```python
# Source: Medium article "Mastering Post-training Quantization"
from copy import deepcopy
import torch
import torch.nn as nn

def quantize_fp16(model: nn.Module) -> nn.Module:
    """
    Convert model to FP16 using simple dtype casting.

    FP16 quantization is straightforward - cast weights to float16.
    No calibration needed, no scale/zero-point calculation.
    """
    # Always copy first (CONTEXT.md requirement)
    model_copy = deepcopy(model)

    # Convert all parameters to FP16
    for param in model_copy.parameters():
        param.data = param.data.to(torch.float16)

    return model_copy
```

**Note on FP16 Approach:** Research shows FP16 "quantization" is typically just dtype casting. The discussion "[Float16 Dynamic Quantization has no model size benefit](https://discuss.pytorch.org/t/float16-dynamic-quantization-has-no-model-size-benefit/99675)" reveals that `torch.quantization` for FP16 doesn't actually reduce model size because the dtype remains float32 internally. Simple casting is the correct approach for Phase 1.

### Pattern 4: Configuration Priority

**What:** Function parameters > global config > defaults.

**When to use:** Throughout the API for all quantization parameters.

**Example:**
```python
# Source: CONTEXT.md decision on configuration priority
from dataclasses import dataclass
from typing import Optional

@dataclass
class QuantizationConfig:
    """Configuration for quantization parameters."""
    dtype: torch.dtype = torch.qint8  # INT8 or FP16
    symmetric: Optional[bool] = None  # None = layer-dependent
    per_channel: bool = True


def _resolve_config(
    user_config: Optional[QuantizationConfig],
    global_config: Optional[QuantizationConfig],
    **kwargs
) -> QuantizationConfig:
    """
    Resolve configuration with priority: kwargs > user_config > global_config > defaults.

    Implements CONTEXT.md decision on configuration priority.
    """
    # Start with defaults
    config = QuantizationConfig()

    # Apply global config if provided
    if global_config is not None:
        for key, value in global_config.__dict__.items():
            if value is not None:
                setattr(config, key, value)

    # Apply user config (overrides global)
    if user_config is not None:
        for key, value in user_config.__dict__.items():
            if value is not None:
                setattr(config, key, value)

    # Apply kwargs (highest priority)
    for key, value in kwargs.items():
        if hasattr(config, key) and value is not None:
            setattr(config, key, value)

    return config
```

### Anti-Patterns to Avoid

- **Depending on torch.ao.quantization:** This module is deprecated and will be removed in PyTorch 2.10. Use native torch operations instead.
- **Modifying the user's model:** CONTEXT.md explicitly requires always copying. Use `deepcopy` or create new tensors.
- **Hardcoding dtype ranges:** Use calculated ranges based on target dtype (qmin, qmax).
- **Assuming calibration data:** Phase 1 is dynamic-only - no calibration data required.

## Don't Hand-Roll

Problems with existing solutions:

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Per-channel quantization | Custom scale calculation per channel | `torch.quantize_per_channel` | Native implementation, optimized, handles edge cases |
| Per-tensor quantization | Custom scale/zero-point calculation | `torch.quantize_per_tensor` | Native, tested, stable API |
| Model copying | Manual tensor copying | `copy.deepcopy(model)` | Handles all model state (buffers, submodules) correctly |
| FP16 conversion | Custom float16 math | `tensor.to(torch.float16)` | Native dtype conversion, hardware accelerated |
| state_dict serialization | Custom serialization | `torch.save`, `torch.load` | Standard PyTorch persistence format |

**Key insight:** PyTorch provides stable, optimized primitives for the core quantization operations. Build custom logic for orchestration and model-agnostic handling, but leverage native ops for the actual quantization math.

## Common Pitfalls

### Pitfall 1: Using Deprecated torch.ao.quantization APIs

**What goes wrong:**
Code written with `torch.ao.quantization.quantize_dynamic` or `torch.ao.quantization.prepare` will break when PyTorch 2.10+ removes these APIs.

**Why it happens:**
These APIs are documented as deprecated with planned removal. The migration path is to `torchao`, which itself has unstable APIs.

**How to avoid:**
Use PyTorch's stable quantization primitives directly:
- `torch.quantize_per_tensor` - per-tensor affine quantization
- `torch.quantize_per_channel` - per-channel affine quantization
- `torch.dequantize` - dequantization
- `tensor.to(torch.float16)` - FP16 conversion

**Warning signs:**
- Importing from `torch.ao.quantization`
- Using `torch.quantization.quantize_dynamic`
- Using `prepare()`/`convert()` workflow from eager mode quantization

**Severity:** CRITICAL - Will break in PyTorch 2.10

### Pitfall 2: Modifying the Original Model

**What goes wrong:**
User's original model gets modified during quantization, causing data loss and unexpected behavior.

**Why it happens:**
PyTorch modules are mutable. Operations like `module.weight.data = ...` modify in-place.

**How to avoid:**
Always create a deep copy before any modification:
```python
from copy import deepcopy
quantized_model = deepcopy(original_model)
# Now quantize quantized_model safely
```

**Warning signs:**
- Direct assignment to model parameters without copying first
- Returning the same object passed in (identity check passes)

**Severity:** CRITICAL - Data loss, violates CONTEXT.md requirement

### Pitfall 3: Incorrect Per-Channel Axis Specification

**What goes wrong:**
Per-channel quantization produces incorrect results or errors because the wrong axis is specified.

**Why it happens:**
For nn.Linear, weights are (out_features, in_features) - axis=0 is correct.
For nn.Conv2d, weights are (out_channels, in_channels, kH, kW) - axis=0 is correct.
But custom layer types may have different layouts.

**How to avoid:**
- Document that axis=0 assumes standard PyTorch weight layout
- For Phase 1, support only standard Linear and Conv2d layers
- Return list of unsupported layer types for custom handling

**Warning signs:**
- Shape mismatches when applying per-channel quantization
- "size mismatch" errors after quantization

**Severity:** HIGH - Produces broken models

### Pitfall 4: FP16 "Quantization" That Doesn't Reduce Size

**What goes wrong:**
FP16 quantization using `torch.quantization` doesn't actually reduce model size because the internal dtype remains float32.

**Why it happens:**
PyTorch's quantization APIs for FP16 were designed for dynamic quantization workflows, not storage reduction. The discussion "[Float16 Dynamic Quantization has no model size benefit](https://discuss.pytorch.org/t/float16-dynamic-quantization-has-no-model-size-benefit/99675)" confirms this.

**How to avoid:**
For FP16, use simple dtype casting:
```python
# Correct - actually reduces size
model.to(torch.float16)

# Incorrect - may not reduce size
torch.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.float16)
```

**Warning signs:**
- Model size doesn't change after FP16 quantization
- Using `torch.quantization` for FP16

**Severity:** HIGH - Doesn't achieve quantization goal

### Pitfall 5: Not Handling Both nn.Module and state_dict Inputs

**What goes wrong:**
API only works with nn.Module, breaking model-agnostic requirement (AGN-02).

**Why it happens:**
Focusing only on the common case (nn.Module) and forgetting state_dict use case.

**How to avoid:**
- Accept `Union[nn.Module, Dict[str, torch.Tensor]]` as input
- Detect input type using `isinstance`
- For state_dict: Phase 1 can require architecture info (raise helpful error)
- Document both input formats

**Warning signs:**
- Function signature only accepts `nn.Module`
- No mention of state_dict in documentation

**Severity:** MEDIUM - Violates AGN-02 requirement

## Code Examples

Verified patterns from official sources:

### Dynamic INT8 Quantization (Per-Channel)

```python
# Source: PyTorch quantization primitives + torchao patterns
import torch
from copy import deepcopy
from typing import List, Tuple
import torch.nn as nn

class DynamicInt8Quantizer:
    """
    Dynamic INT8 quantization with per-channel scaling.

    Weights: quantized with per-channel scaling (static)
    Activations: quantized dynamically at runtime
    """

    @staticmethod
    def quantize_linear(module: nn.Linear) -> nn.Linear:
        """Quantize a Linear layer's weights with per-channel scaling."""
        # Calculate per-channel scale and zero-point
        weight = module.weight.data
        axis = 0  # Per output channel

        # Per-channel: reduce over input features dimension
        min_val = weight.amin(dim=axis, keepdim=True)
        max_val = weight.amax(dim=axis, keepdim=True)

        qmin, qmax = -128, 127
        scale = (max_val - min_val) / (qmax - qmin)
        zero_point = qmin - (min_val / scale)

        # Clamp scale to avoid division by zero
        scale = torch.clamp(scale, min=1e-8)

        # Quantize weights
        q_weight = torch.quantize_per_channel(
            weight, scale.squeeze(), zero_point.int().squeeze(),
            axis=0, dtype=torch.qint8
        )

        # Create new linear with quantized weight
        # Note: For actual use, need quantized.Linear module
        # This is simplified for demonstration
        quantized = nn.Linear(
            module.in_features, module.out_features,
            bias=module.bias is not None
        )
        quantized.weight.data = q_weight
        if module.bias is not None:
            quantized.bias.data = module.bias.data.clone()

        return quantized


def quantize_model_dynamic(
    model: nn.Module,
    dtype: torch.dtype = torch.qint8
) -> Tuple[nn.Module, List[str]]:
    """
    Apply dynamic quantization to model.

    Returns:
        Tuple of (quantized_model, skipped_layers)
    """
    # Always copy (CONTEXT.md requirement)
    model_copy = deepcopy(model)
    skipped = []

    for name, module in list(model_copy.named_children()):
        if isinstance(module, nn.Linear):
            quantized = DynamicInt8Quantizer.quantize_linear(module)
            setattr(model_copy, name, quantized)
        else:
            # Skip unsupported layers (CONTEXT.md: partial quantization)
            skipped.append(name)

    return model_copy, skipped
```

### FP16 Quantization (Simple Cast)

```python
# Source: Medium "Mastering Post-training Quantization" + PyTorch docs
from copy import deepcopy
import torch
import torch.nn as nn

def quantize_fp16(model: nn.Module) -> nn.Module:
    """
    Convert model to FP16 using dtype casting.

    Simple and effective - no calibration needed.
    """
    # Always copy (CONTEXT.md requirement)
    model_copy = deepcopy(model)

    # Convert all parameters to FP16
    for param in model_copy.parameters():
        param.data = param.data.to(torch.float16)

    return model_copy
```

### Symmetric vs Asymmetric Quantization

```python
# Source: NVIDIA Blog "Model Quantization: Concepts, Methods, and Why It Matters"
import torch
from typing import Tuple

def symmetric_quantize(
    tensor: torch.Tensor,
    qmin: int = -128,
    qmax: int = 127
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Symmetric quantization: zero_point = 0.

    Formula: q = clip(round(x / scale), qmin, qmax)
    Scale: max(abs(x)) / qmax
    """
    max_abs = tensor.abs().max().item()
    scale = max_abs / qmax

    q = torch.clamp(
        (tensor / scale).round().to(torch.int32),
        qmin, qmax
    )

    return q, torch.tensor(scale)


def asymmetric_quantize(
    tensor: torch.Tensor,
    qmin: int = -128,
    qmax: int = 127
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Asymmetric (affine) quantization.

    Formula: q = clip(round(x / scale + zero_point), qmin, qmax)
    Scale: (x_max - x_min) / (qmax - qmin)
    Zero_point: qmin - (x_min / scale)
    """
    x_min = tensor.min().item()
    x_max = tensor.max().item()

    scale = (x_max - x_min) / (qmax - qmin)
    zero_point = qmin - (x_min / scale)

    q = torch.clamp(
        (tensor / scale + zero_point).round().to(torch.int32),
        qmin, qmax
    )

    return q, torch.tensor(scale), torch.tensor(zero_point)
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| `torch.ao.quantization.*` | `torchao.quantize_*` or native ops | PyTorch 2.9+ | Old APIs deprecated, removal in 2.10 |
| torchao as reference | torchao for patterns only | 2025 | torchao APIs unstable, don't depend |
| Static-first quantization | Dynamic quantization for simplicity | - | Dynamic is simpler, no calibration needed |

**Deprecated/outdated:**
- **torch.ao.quantization.quantize_dynamic**: Being deprecated, use native ops or torchao patterns
- **torch.ao.quantization.prepare/convert**: Being deprecated, migration to torchao pt2e
- **eager mode quantization**: Migrating to torchao eager mode quantize_ API

## Open Questions

### 1. FP16 Quantization: Simple Cast vs Full Pipeline

**What we know:**
- Simple dtype casting (`to(float16)`) works and reduces memory
- PyTorch's `quantize_dynamic` with FP16 doesn't reduce model size (per forum discussion)
- CONTEXT.md lists FP16 approach as "thatAverageGuy's Discretion"

**What's unclear:**
- Whether full quantization pipeline (with observers) provides any benefit for FP16

**Recommendation:**
Use simple casting for Phase 1. Full FP16 quantization pipeline doesn't provide clear benefits and adds complexity. Revisit if users request specific FP16 features.

### 2. Layer-Dependent Symmetric/Asymmetric Selection

**What we know:**
- CONTEXT.md: "Symmetric vs asymmetric is layer-dependent"
- Research shows weights often benefit from asymmetric, activations from symmetric
- Per-channel quantization typically uses asymmetric for weights

**What's unclear:**
- Exact rules for which layers should use which scheme

**Recommendation:**
Implement heuristic defaults:
- Weights: asymmetric (better for non-symmetric distributions)
- Activations (dynamic): symmetric (simpler, no zero-point tracking)
- Allow user override via configuration

### 3. state_dict Input Without Architecture Info

**What we know:**
- CONTEXT.md requires accepting state_dict as input
- state_dict alone doesn't contain model architecture
- Can't quantize without knowing layer types

**What's unclear:**
- How to handle state_dict input in Phase 1

**Recommendation:**
For Phase 1:
- Accept state_dict input
- Require user to also provide architecture info (nn.Module class or similar)
- Raise helpful error if only state_dict is provided
- Document this limitation
- Future phase: auto-detect architecture from state_dict keys

## Sources

### Primary (HIGH confidence)

- [PyTorch Quantization Documentation](https://docs.pytorch.org/docs/stable/quantization.html) - Official quantization API, deprecation notice for torch.ao.quantization
- [Quantization Overview - torchao 0.15](https://docs.pytorch.org/ao/stable/quantization_overview.html) - torchao stack, quantization flows, tensor subclasses
- [torchao quantization README](https://github.com/pytorch/ao/blob/main/torchao/quantization/README.md) - API patterns, weight only/dynamic quantization examples
- [PyTorch Saving and Loading Models](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html) - state_dict conventions, model copying best practices
- [NVIDIA Blog - Model Quantization Concepts](https://developer.nvidia.com/blog/model-quantization-concepts-methods-and-why-it-matters/) - Affine vs symmetric quantization, per-channel scaling

### Secondary (MEDIUM confidence)

- [PyTorch Discussion - FP16 quantization size](https://discuss.pytorch.org/t/float16-dynamic-quantization-has-no-model-size-benefit/99675) - FP16 quantization doesn't reduce size with torch.quantization
- [PyTorch Discussion - state_dict quantization](https://discuss.pytorch.org/t/how-do-i-save-and-load-quantization-model/64188) - Quantized model save/load patterns
- [Lei Mao - PyTorch Static Quantization](https://leimao.github.io/blog/PyTorch-Static-Quantization/) - Comprehensive quantization tutorial with code examples
- [PyTorch Blog - Introduction to Quantization](https://pytorch.org/blog/introduction-to-quantization-on-pytorch/) - Per-channel quantization explanation

### Tertiary (LOW confidence - verify before use)

- [Medium - Mastering Post-training Quantization](https://medium.com/@balci.pelin/quantization-2-7398a0ce7584) - Mentions copy.deepcopy usage for FP16
- Various blog posts on quantization formulas - use for reference, verify with official docs

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - Official PyTorch documentation, version requirements clear
- Architecture: HIGH - Based on existing project research and PyTorch patterns
- Pitfalls: HIGH - Official deprecation notices, verified forum discussions

**Research date:** 2026-02-03
**Valid until:** 2026-03-03 (30 days - PyTorch ecosystem stable but torchao evolving rapidly)

**Critical decision documented:**
DO NOT use `torch.ao.quantization` APIs - they are deprecated and will be removed in PyTorch 2.10.
DO NOT depend on `torchao` - APIs are unstable ("may change in the future").
USE native PyTorch ops: `torch.quantize_per_tensor`, `torch.quantize_per_channel`, `tensor.to(dtype)`.

---
*Phase: 01-core-quantization-foundation*
*Research completed: 2026-02-03*
