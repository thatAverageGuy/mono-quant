# Architecture Research

**Domain:** Model Quantization Tools (PyTorch)
**Researched:** 2026-02-03
**Confidence:** HIGH

## Standard Architecture

### System Overview

Lightweight quantization tools follow a layered architecture with clear separation between core quantization logic and user interfaces (CLI/API). Based on research of PyTorch quantization, torchao, optimum-quanto, and bitsandbytes:

```
┌─────────────────────────────────────────────────────────────────────┐
│                         User Interface Layer                         │
├─────────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐     │
│  │     CLI     │  │ Python API  │  │  Config / YAML          │     │
│  │  (click/typer)│  │  @quantize │  │  QuantizationConfig    │     │
│  └──────┬──────┘  └──────┬──────┘  └──────────┬──────────────┘     │
│         │                │                     │                    │
└─────────┼────────────────┼─────────────────────┼────────────────────┘
          │                │                     │
          ▼                ▼                     ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      Quantization Orchestrator                       │
├─────────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐     │
│  │   Quantize  │  │  Calibrate  │  │      Freeze             │     │
│  │    ()       │  │     ()      │  │       ()                │     │
│  └──────┬──────┘  └──────┬──────┘  └──────────┬──────────────┘     │
│         │                │                     │                    │
└─────────┼────────────────┼─────────────────────┼────────────────────┘
          │                │                     │
          ▼                ▼                     ▼
┌─────────────────────────────────────────────────────────────────────┐
│                           Core Quantization                          │
├─────────────────────────────────────────────────────────────────────┤
│  ┌───────────┐  ┌───────────┐  ┌───────────┐  ┌─────────────────┐  │
│  │ Observers │  │  Mappers  │  │ Quantizers│  │   Validators    │  │
│  │           │  │           │  │           │  │                 │  │
│  │ MinMax    │  │ Affine    │  │ int8/4    │  │ accuracy_check  │  │
│  │ Histogram │  │ Symmetric │  │ fp16/8    │  │ size_compare    │  │
│  │ MovingAvg │  │ PerTensor │  │ dynamic   │  │                 │  │
│  │           │  │ PerChannel│  │ static    │  │                 │  │
│  └───────────┘  └───────────┘  └───────────┘  └─────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
          │                │                     │
          ▼                ▼                     ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      PyTorch Integration Layer                       │
├─────────────────────────────────────────────────────────────────────┤
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │                    Module Replacement                         │   │
│  │  nn.Linear -> QuantizedLinear   nn.Conv2d -> QuantizedConv2d │   │
│  └──────────────────────────────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │                    State Dict Utils                          │   │
│  │  save/load  quantization_map  serialize/deserialize           │   │
│  └──────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

### Component Responsibilities

| Component | Responsibility | Communicates With |
|-----------|----------------|-------------------|
| **CLI** | Command-line interface for batch processing, CI/CD integration | Orchestrator, Config |
| **Python API** | Decorator-based and function-based quantization interface | Orchestrator, user code |
| **Config** | Quantization parameters (dtype, granularity, scheme) | Orchestrator, Observers |
| **Orchestrator** | Coordinates quantization workflow (quantize -> calibrate -> freeze) | All components |
| **Observers** | Collect tensor statistics (min/max, histograms) during calibration | Mappers, Quantizers |
| **Mappers** | Calculate scale/zero-point from observed statistics | Observers, Quantizers |
| **Quantizers** | Apply quantization transformation to tensors | Mappers, PyTorch modules |
| **Validators** | Verify quantized model accuracy, size, correctness | Orchestrator, Quantizers |
| **Module Replacement** | Swap nn.Linear/Conv2d with quantized variants | Quantizers, PyTorch |

## Recommended Project Structure

```
mono-quant/
├── src/
│   └── mono_quant/
│       ├── __init__.py           # Public API exports
│       ├── core/                 # Core quantization logic (no external deps)
│       │   ├── __init__.py
│       │   ├── observers.py      # Observer classes (MinMax, Histogram, etc.)
│       │   ├── mappers.py        # Scale/zero-point calculation
│       │   ├── quantizers.py     # Quantization transformations
│       │   └── schemes.py        # Affine, symmetric quantization schemes
│       ├── calibration/          # Calibration logic
│       │   ├── __init__.py
│       │   └── runner.py         # Calibration data collection
│       ├── modules/              # Quantized PyTorch module replacements
│       │   ├── __init__.py
│       │   ├── linear.py         # QuantizedLinear
│       │   └── conv.py           # QuantizedConv2d (optional, later)
│       ├── api/                  # Python API surface
│       │   ├── __init__.py
│       │   ├── decorators.py     # @quantize decorator
│       │   └── functions.py      # quantize() function API
│       ├── config/               # Configuration handling
│       │   ├── __init__.py
│       │   └── quant_config.py   # QuantizationConfig dataclass
│       ├── validation/           # Model validation after quantization
│       │   ├── __init__.py
│       │   └── metrics.py        # Accuracy, size comparison
│       └── serialization/        # Save/load quantized models
│           ├── __init__.py
│           └── state_dict.py     # quantization_map, save/load
├── cli/                          # CLI entry point
│   ├── __init__.py
│   └── main.py                   # typer/click CLI
├── tests/                        # Tests organized by component
│   ├── test_observers.py
│   ├── test_mappers.py
│   ├── test_quantizers.py
│   ├── test_api.py
│   └── test_cli.py
└── pyproject.toml
```

### Structure Rationale

- **`core/`**: Pure quantization logic, no PyTorch dependencies where possible. Contains math for scale/zero-point, quantization schemes. This is the foundation.
- **`calibration/`**: Separated because it's distinct from pure quantization - involves running the model to collect statistics. Can be optional for weight-only quantization.
- **`modules/`**: PyTorch-specific replacements. Separated to make core logic testable without PyTorch overhead.
- **`api/`**: The public interface. Changes here don't affect core logic. Easy to extend with new interfaces.
- **`config/`**: Centralized configuration. Single source of truth for quantization parameters.
- **`validation/`**: Optional but important. Users want to know if quantization hurt accuracy.
- **`serialization/`**: Critical for build-phase tools. Users quantize, save, deploy elsewhere.

## Architectural Patterns

### Pattern 1: Observer Pattern for Calibration

**What:** Observers collect tensor statistics during a forward pass without modifying model behavior.

**When to use:** Static quantization where you need to determine activation ranges before quantizing.

**Trade-offs:**
- (+) Clean separation between data collection and quantization
- (+) Multiple observer types can be swapped
- (-) Requires a calibration forward pass
- (-) Adds temporary memory overhead during calibration

**Example:**
```python
class MinMaxObserver:
    """Records min/max values to determine quantization range."""
    def __init__(self):
        self.min_val = float('inf')
        self.max_val = float('-inf')

    def update(self, tensor: torch.Tensor) -> None:
        self.min_val = min(self.min_val, tensor.min().item())
        self.max_val = max(self.max_val, tensor.max().item())

    def calculate_qparams(self):
        """Returns (scale, zero_point) for affine quantization."""
        qmin, qmax = -128, 127  # int8 range
        scale = (self.max_val - self.min_val) / (qmax - qmin)
        zero_point = qmin - round(self.min_val / scale)
        return scale, zero_point
```

### Pattern 2: Module Replacement Pattern

**What:** Systematically replace PyTorch modules (`nn.Linear`, `nn.Conv2d`) with quantized equivalents that wrap quantization/dequantization logic.

**When to use:** When you need transparent quantization - the user's model code doesn't change.

**Trade-offs:**
- (+) User model code remains unchanged
- (+) Works with any PyTorch model
- (-) Requires careful handling of module hierarchies
- (-) Module fusion becomes complex

**Example:**
```python
def replace_with_quantized(model: nn.Module, config: QuantizationConfig):
    """Replace nn.Linear with QuantizedLinear throughout model."""
    for name, module in model.named_children():
        if isinstance(module, nn.Linear):
            quantized = QuantizedLinear.from_float(module, config)
            setattr(model, name, quantized)
        else:
            replace_with_quantized(module, config)  # recurse
```

### Pattern 3: Decorator API Pattern

**What:** A simple decorator interface for one-shot quantization of functions.

**When to use:** For simple use cases where users just want to quantize a model and save it.

**Trade-offs:**
- (+) Minimal code for users
- (+) Great for Jupyter notebooks
- (-) Less control than the full API
- (-) Not suitable for complex calibration workflows

**Example:**
```python
def quantize(dtype=torch.int8, calibration_data=None):
    """Decorator to quantize a model function."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            model = func(*args, **kwargs)
            # Apply quantization
            quantize_module(model, dtype=dtype)
            if calibration_data:
                calibrate(model, calibration_data)
            return model
        return wrapper
    return decorator

@quantize(dtype=torch.int8)
def my_model():
    return nn.Linear(1024, 1024)
```

### Pattern 4: Phase-based Workflow

**What:** Quantization proceeds through distinct phases: Quantize -> Calibrate -> Freeze.

**When to use:** For static quantization where calibration is required.

**Trade-offs:**
- (+) Clear mental model for users
- (+) Each phase can be run independently
- (-) More complex than simple one-shot quantization
- (-) Requires state management between phases

**Example:**
```python
class QuantizationWorkflow:
    def quantize(self, model, weights_dtype, activations_dtype):
        # Insert quantization stubs, replace modules
        pass

    def calibrate(self, model, calibration_data):
        # Run forward pass, collect statistics
        pass

    def freeze(self, model):
        # Replace dynamic quantization with static quantized weights
        pass
```

## Data Flow

### Quantization Flow

```
[User Model]  [Calibration Data]
     │                │
     ▼                ▼
┌─────────────────────────────────┐
│    Phase 1: Quantize()          │
│  - Replace nn.Linear with       │
│    QuantizedLinear              │
│  - Insert quant/dequant stubs   │
└────────────┬────────────────────┘
             │
             ▼
┌─────────────────────────────────┐
│    Phase 2: Calibrate()         │
│  - Forward pass with data       │
│  - Observers collect stats      │
│  - Calculate scales/zp          │
└────────────┬────────────────────┘
             │
             ▼
┌─────────────────────────────────┐
│    Phase 3: Freeze()            │
│  - Convert weights to target    │
│    dtype in-place               │
│  - Remove observers             │
└────────────┬────────────────────┘
             │
             ▼
    [Quantized Model]
             │
             ▼
       [Save/Deploy]
```

### Request Flow (CLI)

```
[CLI Command]
    │
    ▼
[Parse Args] → [Load Config] → [Load Model]
    │                                    │
    ▼                                    ▼
[Create Calibration]            [Quantize Weights]
    │                                    │
    └────────────┬───────────────────────┘
                 │
                 ▼
          [Run Calibration]
                 │
                 ▼
           [Freeze Weights]
                 │
                 ▼
           [Save Model]
                 │
                 ▼
           [Exit / Report]
```

### Request Flow (Python API)

```
[@quantize decorator or quantize() call]
    │
    ▼
[Validate Inputs]
    │
    ▼
[Create QuantizationConfig]
    │
    ├──────────────────┐
    ▼                  ▼
[Quantize Path]   [Dynamic Path]
    │                  │
    ▼                  ▼
[Replace Modules] [Insert Quant/Dequant]
    │                  │
    └──────────────────┘
              │
              ▼
       [Return Quantized Model]
```

### Key Data Flows

1. **Weight-only quantization:** Model -> Quantize weights directly -> Save. No calibration needed.
2. **Static quantization:** Model -> Insert observers -> Calibration forward pass -> Calculate qparams -> Freeze -> Save.
3. **Dynamic quantization:** Model -> Insert quant/dequant ops -> Quantize weights -> Save (no calibration).

## Scaling Considerations

| Scale | Architecture Adjustments |
|-------|--------------------------|
| 0-100 models | Single-process, in-memory quantization is fine |
| 100-10K models | Batch processing, calibration data caching, parallel quantization |
| 10K+ models | Distributed calibration, incremental quantization, model-specific configs |

### Scaling Priorities

1. **First bottleneck:** Calibration data loading. Use lazy loading, streaming, or caching.
2. **Second bottleneck:** Model size during replacement. For very large models, consider in-place replacement without copying.

## Anti-Patterns

### Anti-Pattern 1: Tightly Coupling Quantization to Model Architecture

**What people do:** Writing quantization code that assumes specific model architectures (e.g., "transformer has 12 layers").

**Why it's wrong:** Breaks model-agnostic design promise. Every new model format requires code changes.

**Do this instead:** Use generic module replacement patterns. Iterate over `named_modules()` and check `isinstance(module, nn.Linear)`.

### Anti-Pattern 2: Mixing Concerns in a Single Quantize Function

**What people do:** A single `quantize()` function that handles calibration, validation, and serialization all together.

**Why it's wrong:** Can't test components independently. Hard to extend with new calibration methods.

**Do this instead:** Separate phases: `quantize()`, `calibrate()`, `freeze()`. Each has a single responsibility.

### Anti-Pattern 3: Hardcoding Quantization Parameters

**What people do:** Embedding scale formulas and dtype ranges directly in quantization logic.

**Why it's wrong:** Can't experiment with different schemes. Users can't customize.

**Do this instead:** Use a `QuantizationConfig` dataclass that encapsulates all parameters. Allow users to pass custom configs.

### Anti-Pattern 4: Ignoring Module State During Replacement

**What people do:** Creating new modules without copying weights, biases, and other state from original modules.

**Why it's wrong:** Quantized model loses trained parameters. Model is broken.

**Do this instead:** Always implement `from_float()` classmethod that properly copies all state from original module.

## Integration Points

### External Services

| Service | Integration Pattern | Notes |
|---------|---------------------|-------|
| PyTorch | Direct import, nn.Module subclassing | Core dependency |
| Hugging Face transformers | Optional wrapper classes | For model loading/saving convenience |
| safetensors | Optional serialization format | Recommended for production |

### Internal Boundaries

| Boundary | Communication | Notes |
|----------|---------------|-------|
| core/ <-> modules/ | Direct import | modules depends on core, not vice versa |
| api/ <-> core/ | Direct import | api uses core classes directly |
| cli/ <-> api/ | Direct import | cli is a thin wrapper around api |
| validation/ <-> core/ | Direct import | validation uses core to compare models |

## Build Order (Component Dependencies)

For phased development, components should be built in this order:

1. **Phase 1: Core quantization math** (`core/schemes.py`, `core/mappers.py`)
   - No dependencies on other components
   - Can be unit tested with pure tensors

2. **Phase 2: Observers** (`core/observers.py`)
   - Depends on: schemes (for qparams calculation)
   - Independent of PyTorch modules

3. **Phase 3: Quantizers** (`core/quantizers.py`)
   - Depends on: observers, mappers, schemes
   - Pure tensor transformations

4. **Phase 4: Quantized Modules** (`modules/linear.py`)
   - Depends on: quantizers, PyTorch
   - First component that actually replaces nn.Module

5. **Phase 5: Calibration** (`calibration/runner.py`)
   - Depends on: observers, modules
   - Orchestrates forward pass with observers

6. **Phase 6: Serialization** (`serialization/state_dict.py`)
   - Depends on: modules (to extract qparams)
   - Independent of other phases

7. **Phase 7: Python API** (`api/`)
   - Depends on: all previous components
   - Public interface

8. **Phase 8: CLI** (`cli/main.py`)
   - Depends on: api
   - User-facing interface

9. **Phase 9: Validation** (`validation/metrics.py`)
   - Depends on: api, serialization
   - Optional but recommended early

## CLI and Python API Relationship

Both interfaces share the same core quantization logic:

```
                    ┌─────────────────────┐
                    │   Core Quantization │
                    │   (core/, modules/) │
                    └──────────┬──────────┘
                               │
                ┌──────────────┴──────────────┐
                ▼                             ▼
        ┌───────────────┐             ┌───────────────┐
        │  Python API   │             │      CLI      │
        │  (@quantize)  │             │  (typer/click)│
        └───────────────┘             └───────────────┘
                │                             │
        ┌───────┴───────┐           ┌───────┴───────┐
        ▼               ▼           ▼               ▼
   @quantize     quantize()    mono-quant      mono-quant
   on function   on model      quantize        quantize
                                              --config
                                              --output
```

**Key principle:** CLI calls Python API functions. No quantization logic exists in CLI that isn't exposed via Python API. This ensures:
- CLI is a thin wrapper
- API is fully programmable
- Both interfaces stay in sync

## Sources

- [Practical Quantization in PyTorch - PyTorch Blog](https://pytorch.org/blog/quantization-in-practice/) - Observer patterns, calibration workflow, QConfig
- [Static Quantization - torchao documentation](https://docs.pytorch.org/ao/stable/static_quantization.html) - Calibration phase, quantization phase patterns
- [Architecture and Components - ExecuTorch documentation](https://docs.pytorch.org/executorch/stable/getting-started-architecture) - Program preparation, AOT compilation concepts
- [Quantization - Hugging Face Optimum documentation](https://huggingface.co/docs/optimum/en/concept_guides/quantization) - Calibration techniques, quantization schemes
- [Model Quantization: Concepts, Methods, and Why It Matters - NVIDIA Blog](https://developer.nvidia.com/blog/model-quantization-concepts-methods-and-why-it-matters/) - Affine vs symmetric quantization, AbsMax algorithm, quantization granularity
- [optimum-quanto GitHub repository](https://github.com/huggingface/optimum-quanto) - Design overview, quantization workflow, tensor subclassing approach
- [bitsandbytes GitHub repository](https://github.com/bitsandbytes-foundation/bitsandbytes) - Lightweight design, minimal dependencies pattern
- [A Survey of Quantization Methods for Efficient Neural Networks](https://arxiv.org/pdf/2103.13630) - Calibration methods, quantization approaches

---
*Architecture research for: Model Quantization Tools*
*Researched: 2026-02-03*
