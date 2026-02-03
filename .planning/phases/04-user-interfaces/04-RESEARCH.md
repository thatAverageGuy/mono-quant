# Phase 4: User Interfaces - Research

**Researched:** 2026-02-03
**Domain:** Python CLI Development, Progress Reporting, API Design
**Confidence:** HIGH

## Summary

Phase 4 requires implementing two user interfaces for the existing mono-quant quantization functionality: a simplified Python API and a command-line interface. The research reveals established patterns and libraries for both domains with clear best practices.

For the CLI, **Click** is the industry standard with proven patterns for subcommands, groups, and entry point integration. **Typer** provides a modern type-hint-based alternative built on Click, but adds an extra dependency for Rich. For progress reporting, **tqdm** is the de facto standard with minimal overhead (~60ns/iter), while **Rich** provides beautiful output at the cost of additional dependencies.

For the unified Python API, the decision point is between explicit parameters vs `**kwargs` for advanced options. The Result object pattern (from Pydantic and similar libraries) provides a clean way to return both success/failure status and results.

**Primary recommendation:** Use Click for CLI (mature, stable, composable), tqdm for progress bars (minimal deps, fast, CI-friendly), explicit common parameters with `**kwargs` for advanced options in the Python API, and a `QuantizationResult` dataclass with `.success` flag and convenience methods.

## Standard Stack

### Core

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| **Click** | 8.1+ | CLI framework | Industry standard, composable, arbitrary nesting of commands, automatic help generation |
| **tqdm** | 4.66+ | Progress bars | De facto standard, 60ns/iter overhead, works in any console including CI/CD |
| **dataclasses** | Built-in (3.7+) | Result objects | Native Python, type-safe, zero dependencies |

### Supporting

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| **Rich** | 13.7+ | Enhanced output (optional) | When pretty progress bars and colored terminal output are desired |
| **typer** | 0.9+ | Type-hint-based CLI (optional) | When you want Typer's type hint approach instead of Click decorators |

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| **Click** | Typer | Typer is built on Click but adds Rich dependency. Simpler for beginners but less explicit control. |
| **Click** | argparse | Built-in but more verbose for subcommands, less composable, manual help generation. |
| **tqdm** | Rich progress | Rich has nicer output but heavier dependency. tqdm is minimal and ubiquitous. |
| **tqdm** | Custom progress bar | Hand-rolled implementation creates maintenance burden. tqdm handles edge cases. |

**Installation:**
```bash
# Core CLI and progress
pip install click tqdm

# With enhanced terminal output
pip install click tqdm rich

# With Typer instead of Click
pip install typer  # Includes rich by default
```

## Architecture Patterns

### Recommended Project Structure (Phase 4 additions)

```
src/
└── mono_quant/
    ├── __init__.py           # Public API exports (add quantize())
    ├── api/                  # NEW: High-level Python API
    │   ├── __init__.py
    │   ├── quantize.py       # Unified quantize() function
    │   └── result.py         # QuantizationResult dataclass
    ├── cli/                  # NEW: Command-line interface
    │   ├── __init__.py
    │   ├── main.py           # Click app, command groups
    │   ├── quantize.py       # quantize subcommand
    │   ├── validate.py       # validate subcommand
    │   ├── info.py           # info subcommand
    │   ├── compare.py        # compare subcommand
    │   └── calibrate.py      # calibrate subcommand
    ├── core/                 # Existing: Core quantization (no changes)
    ├── modules/              # Existing: Quantized modules
    ├── io/                   # Existing: Model I/O
    ├── calibration/          # Existing: Calibration utilities
    └── config/               # Existing: Configuration handling
```

### Pattern 1: Click Subcommands with Groups

**What:** Use `@click.group()` to create nested command structures.

**When to use:** For CLI tools with multiple related commands (git-style).

**Example:**
```python
# Source: Click 8.1.x official documentation
import click

@click.group()
@click.version_option(version="0.1.0")
def cli():
    """Mono Quant - Simple, reliable model quantization."""
    pass

@cli.command()
@click.option('--model', '-m', required=True, type=click.Path(exists=True),
              help='Path to model file (.pt, .pth, .safetensors)')
@click.option('--bits', '-b', default=8, type=int,
              help='Target quantization bits (4, 8, or 16 for FP16)')
@click.option('--output', '-o', type=click.Path(),
              help='Output path (default: model_quantized.pt)')
@click.option('--scheme', '-s', type=click.Choice(['symmetric', 'asymmetric']),
              default='symmetric', help='Quantization scheme')
@click.option('--dynamic', is_flag=True,
              help='Use dynamic quantization (no calibration data)')
def quantize(model, bits, output, scheme, dynamic):
    """Quantize a model to specified precision."""
    click.echo(f"Quantizing {model} to {bits} bits...")
    # Implementation

@cli.command()
@click.argument('model_path', type=click.Path(exists=True))
def validate(model_path):
    """Validate a quantized model."""
    click.echo(f"Validating {model_path}...")
    # Implementation

if __name__ == '__main__':
    cli()
```

**Usage:**
```bash
$ monoquant quantize --model model.pt --bits 8
$ monoquant validate model_quantized.pt
$ monoquant --help
```

### Pattern 2: Unified Python API with Result Object

**What:** Single `quantize()` function that dispatches based on parameters, returns a Result object.

**When to use:** For simplified programmatic API that wraps existing `dynamic_quantize()` and `static_quantize()`.

**Example:**
```python
# Source: Standard Python dataclass pattern + API design best practices
from dataclasses import dataclass, field
from typing import Optional, List, Union, Any, Callable
from pathlib import Path
import torch
import torch.nn as nn

@dataclass
class QuantizationResult:
    """
    Result of quantization operation.

    Provides both direct access to results and convenience methods.
    Hybrid approach: exceptions raise on critical errors, but
    result also has success flag for conditional handling.
    """
    model: nn.Module
    info: "QuantizationInfo"
    success: bool = True
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def save(self, path: Union[str, Path]) -> None:
        """Save quantized model to path."""
        from mono_quant.io import save_model
        save_model(self.model, str(path),
                   quantization_info=self.info,
                   sqnr_db=self.info.sqnr_db,
                   compression_ratio=self.info.compression_ratio)

    def validate(self, on_failure: str = "error") -> "ValidationResult":
        """Run validation on quantized model."""
        from mono_quant.io.validation import validate_quantization
        # Note: original model needed for validation, may need to store reference
        raise NotImplementedError("Use validate_quantization directly for now")

    def __bool__(self) -> bool:
        """Allow truthiness check: if result: ..."""
        return self.success


def quantize(
    model: Union[nn.Module, str, Path],
    bits: int = 8,
    dynamic: bool = False,
    scheme: str = "symmetric",
    calibration_data: Optional["CalibrationData"] = None,
    **kwargs
) -> QuantizationResult:
    """
    Quantize a PyTorch model to specified precision.

    Unified API that dispatches to dynamic_quantize() or static_quantize()
    based on parameters.

    Args:
        model: Model to quantize (nn.Module, state_dict, or file path)
        bits: Target bits (4, 8, or 16 for FP16). Default: 8
        dynamic: If True, use dynamic quantization (no calibration needed)
        scheme: Quantization scheme ("symmetric" or "asymmetric")
        calibration_data: Required for static quantization (List[Tensor] or DataLoader)
        **kwargs: Advanced options (group_size, observer_type, etc.)

    Returns:
        QuantizationResult with model, info, and convenience methods

    Raises:
        QuantizationError: On critical quantization failures
        ValueError: On invalid input parameters

    Examples:
        >>> from mono_quant import quantize
        >>> # Dynamic quantization (simple)
        >>> result = quantize(model, bits=8, dynamic=True)
        >>> result.save("quantized.pt")
        >>> # Static quantization
        >>> result = quantize(model, bits=8, dynamic=False,
        ...                   calibration_data=calib_data)
    """
    # Import from existing modules
    from mono_quant.core import dynamic_quantize, static_quantize
    from mono_quant.io.handlers import _prepare_model, _detect_input_format

    # Normalize input
    model_input = model
    if isinstance(model, (str, Path)):
        from mono_quant.io import load_model
        state_dict = load_model(str(model))
        model = state_dict  # Will be handled by _prepare_model

    # Map bits to dtype
    dtype_map = {4: torch.qint8, 8: torch.qint8, 16: torch.float16}
    dtype = dtype_map.get(bits, torch.qint8)

    # Dispatch to appropriate function
    if dynamic:
        q_model, skipped = dynamic_quantize(
            model,
            dtype=dtype,
            symmetric=(scheme == "symmetric"),
            **kwargs
        )
        # Build info
        from mono_quant.core.quantizers import QuantizationInfo
        info = QuantizationInfo(
            selected_layers=[],  # TODO: populate from actual
            skipped_layers=skipped,
            calibration_samples_used=0,
            dtype=dtype,
            symmetric=(scheme == "symmetric"),
        )
    else:
        if calibration_data is None:
            raise ValueError(
                "calibration_data required for static quantization. "
                "Use dynamic=True for quantization without calibration data."
            )
        q_model, info = static_quantize(
            model,
            calibration_data=calibration_data,
            dtype=dtype,
            symmetric=(scheme == "symmetric"),
            **kwargs
        )

    return QuantizationResult(model=q_model, info=info)
```

### Pattern 3: Progress Bars with tqdm

**What:** Wrap iterables with tqdm for automatic progress tracking.

**When to use:** For any long-running operation (calibration, layer quantization).

**Example:**
```python
# Source: tqdm official documentation
from tqdm import tqdm
import time

def process_layers(layers: list):
    """Process layers with progress bar."""
    results = []
    for layer in tqdm(layers, desc="Quantizing layers"):
        result = quantize_layer(layer)
        results.append(result)
    return results

# With custom progress bar for manual updates
def run_calibration(model, data):
    """Run calibration with progress tracking."""
    from tqdm import trange

    n_samples = len(data)
    pbar = trange(n_samples, desc="Calibrating")

    for i, batch in enumerate(pbar):
        model(batch)
        # Update description with progress percentage
        pbar.set_postfix({"sample": f"{i+1}/{n_samples}"})

    pbar.close()
```

**CI/CD compatibility:**
```python
# Disable progress bars in non-interactive environments
import os
from tqdm import tqdm

def smart_tqdm(iterable, **kwargs):
    """tqdm that disables itself in CI/CD environments."""
    # Check if running in CI or no TTY
    if os.getenv('CI') or not os.isatty(1):
        # Return iterable without progress bar
        return iterable
    return tqdm(iterable, **kwargs)
```

### Pattern 4: Entry Point Configuration

**What:** Register CLI command via `console_scripts` entry point.

**When to use:** To make CLI command available after `pip install`.

**Example:**
```python
# pyproject.toml configuration
[project.scripts]
monoquant = "mono_quant.cli.main:cli"
mq = "mono_quant.cli.main:cli"  # Optional shorthand alias

# Alternative in setup.py
setup(
    name="mono-quant",
    entry_points={
        'console_scripts': [
            'monoquant=mono_quant.cli.main:cli',
            'mq=mono_quant.cli.main:cli',
        ],
    },
)
```

After installation:
```bash
$ pip install mono-quant
$ monoquant --help
# or
$ mq --help
```

### Pattern 5: Custom Exception Hierarchy

**What:** Domain-specific exceptions for quantization errors.

**When to use:** For error handling that distinguishes quantization failures from input errors.

**Example:**
```python
# Source: Python exception hierarchy best practices
class MonoQuantError(Exception):
    """Base exception for all mono-quant errors."""

    def __init__(self, message: str, suggestion: Optional[str] = None):
        self.message = message
        self.suggestion = suggestion
        full_message = message
        if suggestion:
            full_message += f"\n{suggestion}"
        super().__init__(full_message)


class QuantizationError(MonoQuantError):
    """Raised when quantization fails."""


class ValidationError(MonoQuantError):
    """Raised when model validation fails."""


class ConfigurationError(MonoQuantError):
    """Raised when configuration is invalid."""


# Usage with actionable messages
def quantize_with_checks(model, bits, group_size):
    if group_size > 64 and model.layers < 64:
        raise ConfigurationError(
            f"group_size={group_size} too large for layer with 64 channels",
            suggestion="Use --group-size 64 or smaller"
        )
```

### Anti-Patterns to Avoid

- **Using `sys.exit()` directly in Click commands:** Use `click.Context.exit()` or raise `click.ClickException` instead for proper CLI behavior.
- **Creating custom progress bars:** tqdm handles edge cases (terminal resizing, non-TTY output, IPython notebooks).
- **Modifying input model in-place:** Always copy before quantizing (already enforced in existing code).
- **Silent progress in Python API:** Provide `show_progress` parameter, default behavior can be silent for library use.

## Don't Hand-Roll

Problems with existing solutions:

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| CLI argument parsing | Custom argparse code | Click decorators | Handles subcommands, help generation, validation, composable groups |
| Progress bars | Custom print loops | tqdm | Handles terminal control codes, ETA calculation, non-TTY detection |
| CLI entry points | Manual script installation | `[project.scripts]` in pyproject.toml | Standard Python packaging, works cross-platform |
| Type validation | Custom type checking | Click types or Typer | Built-in conversion, error messages, path validation |
| Result objects | Dict returns or tuples | dataclass | Type-safe, IDE autocomplete, explicit fields |

**Key insight:** Python's CLI ecosystem is mature with Click as the clear standard. tqdm is the de facto standard for progress. Building custom implementations creates maintenance burden and misses edge cases.

## Common Pitfalls

### Pitfall 1: Progress Bar Breaking CI/CD Output

**What goes wrong:** Progress bars with carriage returns (`\r`) create messy output in CI logs that don't support terminal control codes.

**Why it happens:** tqdm and similar libraries use terminal control codes for in-place updates. CI environments typically don't support these.

**How to avoid:**
```python
import os
import sys

def should_show_progress():
    """Check if progress bars should be shown."""
    # Disable in CI environments
    if os.getenv('CI'):
        return False
    # Disable if no TTY
    return hasattr(sys.stdout, 'isatty') and sys.stdout.isatty()

# Use tqdm with file disable
from tqdm import tqdm
iterable = tqdm(data, disable=not should_show_progress())
```

**Warning signs:**
- CI logs show garbled characters or overlapping text
- Progress characters (`█`, `▓`) appear in log files

**Severity:** MEDIUM - Annoying for users, but doesn't break functionality

### Pitfall 2: CLI Entry Point Not Found After Install

**What goes wrong:** User runs `pip install` but command is not available.

**Why it happens:** Entry point misconfigured or `bin` directory not in PATH.

**How to avoid:**
```toml
# Correct configuration in pyproject.toml
[project.scripts]
monoquant = "mono_quant.cli.main:cli"
```

And verify:
```bash
$ pip show -f mono-quant | grep bin
# Should show entry point location
$ echo $PATH  # Verify user's PATH includes pip bin
```

**Warning signs:**
- `command not found` after install
- Need to run `python -m mono_quant.cli` instead of command name

**Severity:** HIGH - Breaks user experience

### Pitfall 3: Mixing Click and Argparse

**What goes wrong:** Attempting to use Click decorators alongside argparse.ArgumentParser.

**Why it happens:** Incremental migration from argparse to Click, or confusion about which library to use.

**How to avoid:**
Choose one framework per CLI application. Click and argparse don't mix well:
```python
# WRONG - Don't mix
import click
import argparse

@click.command()  # Click decorator
def cli():
    parser = argparse.ArgumentParser()  # Argparse - incompatible!
    # This doesn't work
```

**Warning signs:**
- Both `import click` and `import argparse` in same file
- Using `@click.command()` and `ArgumentParser()` together

**Severity:** HIGH - Causes runtime errors

### Pitfall 4: Blocking Main Thread with Progress Updates

**What goes wrong:** Progress bar updates blocking quantization on large models.

**Why it happens:** tqdm with `miniters=1` updates on every iteration.

**How to avoid:**
```python
# Use reasonable mininterval (default 0.1s is fine)
from tqdm import tqdm
for item in tqdm(data, mininterval=0.5):  # Update max 2x/sec
    process(item)
```

**Warning signs:**
- Progress bar updates dominate runtime
- Quantization significantly slower with progress bar

**Severity:** LOW - Usually not noticeable with default settings

### Pitfall 5: Result Object Lacks Failure Information

**What goes wrong:** Caller can't distinguish successful vs failed quantization without try/except.

**Why it happens:** Returning only model or raising exception without context.

**How to avoid:**
```python
@dataclass
class QuantizationResult:
    model: nn.Module
    info: "QuantizationInfo"
    success: bool  # ALWAYS include
    errors: List[str]  # Populate on failure
```

**Warning signs:**
- No way to check success without try/except
- Function returns different types on success vs failure

**Severity:** MEDIUM - Makes API harder to use correctly

## Code Examples

Verified patterns from official sources:

### Click CLI with Multiple Subcommands

```python
# Source: Click 8.1.x official documentation
import click

@click.group()
@click.version_option(version="0.1.0", prog_name="monoquant")
def cli():
    """Mono Quant - Simple, reliable model quantization."""
    pass

@cli.command(name="quantize")
@click.option('--model', '-m', 'model_path', required=True,
              type=click.Path(exists=True), help='Path to input model')
@click.option('--bits', '-b', default=8, type=click.IntRange(4, 16),
              help='Quantization bits (4, 8, or 16 for FP16)')
@click.option('--output', '-o', 'output_path', type=click.Path(),
              help='Output path (default: <input>_quantized.<ext>)')
@click.option('--scheme', '-s', type=click.Choice(['symmetric', 'asymmetric']),
              default='symmetric', help='Quantization scheme')
@click.option('--dynamic/--static', 'dynamic', default=False,
              help='Use dynamic quantization (no calibration needed)')
@click.option('--calibration', '-c', type=click.Path(exists=True),
              help='Calibration data file (for static quantization)')
@click.option('--strict', is_flag=True,
              help='Exit immediately on any error')
def quantize_cmd(model_path, bits, output_path, scheme, dynamic, calibration, strict):
    """Quantize a PyTorch model to specified precision."""
    from mono_quant.api import quantize
    from mono_quant.io import load_model

    click.echo(f"Loading model: {model_path}")

    try:
        state_dict = load_model(model_path)
        # ... state_dict to model conversion ...

        if dynamic:
            result = quantize(state_dict, bits=bits, dynamic=True, scheme=scheme)
        else:
            if not calibration:
                raise click.ClickException(
                    "Calibration data required for static quantization. "
                    "Use --dynamic flag or provide --calibration file."
                )
            # Load calibration data
            result = quantize(state_dict, bits=bits, dynamic=False,
                             scheme=scheme, calibration_data=calib_data)

        if not result.success:
            for error in result.errors:
                click.echo(f"Error: {error}", err=True)
            if strict:
                raise click.ClickException("Quantization failed")

        # Save output
        output = output_path or f"{model_path}_quantized.pt"
        result.save(output)
        click.echo(f"Saved quantized model to: {output}")
        click.echo(f"SQNR: {result.info.sqnr_db:.2f} dB")
        click.echo(f"Compression: {result.info.compression_ratio:.2f}x")

    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        raise click.ClickException(str(e))

@cli.command()
@click.argument('model_path', type=click.Path(exists=True))
def validate(model_path):
    """Validate a quantized model."""
    click.echo(f"Validating {model_path}...")
    # Implementation

@cli.command()
@click.argument('model_path', type=click.Path(exists=True))
def info(model_path):
    """Display information about a quantized model."""
    click.echo(f"Model info for: {model_path}")
    # Implementation

if __name__ == '__main__':
    cli()
```

### tqdm with Click Integration

```python
# Source: tqdm + Click integration patterns
import click
from tqdm import tqdm

# Disable tqdm in non-TTY environments (CI/CD)
import sys
TQDM_DISABLE = not hasattr(sys.stdout, 'isatty') or not sys.stdout.isatty()

@click.command()
@click.option('--count', default=100)
def process(count):
    """Process items with progress bar."""
    items = range(count)

    # Use tqdm as context manager for cleanup
    with tqdm(items, disable=TQDM_DISABLE,
              desc="Processing") as pbar:
        for item in pbar:
            # Update description dynamically
            pbar.set_postfix({"item": item})
            # Do work
            result = heavy_computation(item)

    click.echo("Done!")
```

### Custom Exception with Actionable Messages

```python
# Source: Python exception best practices + Click error handling
class QuantizationError(Exception):
    """Base exception for quantization errors."""

    def __init__(self, message: str, suggestion: str = None):
        self.message = message
        self.suggestion = suggestion
        parts = [message]
        if suggestion:
            parts.append(f"Suggestion: {suggestion}")
        super().__init__("\n".join(parts))


# In CLI command
try:
    result = quantize(model, bits=8, group_size=128)
except QuantizationError as e:
    click.echo(f"Quantization failed: {e}", err=True)
    if e.suggestion:
        click.echo(str(e.suggestion), err=True)
    raise click.ClickException(str(e))
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| argparse manual | Click decorators | ~2015 | Click is now standard for CLI |
| sys.stdout progress | tqdm/rich | ~2016 | tqdm is standard for progress |
| Manual script setup | console_scripts entry points | ~2018 | pyproject.toml standardizes this |
| Typer emerging | Typer over Click | 2020+ | Typer simplifies but adds Rich dep |

**Current state (2026):**
- **Click 8.1.x** is the mature, stable choice for CLIs
- **tqdm** remains the standard for progress bars (minimal deps)
- **Rich** is gaining popularity for pretty terminals but heavier
- **Typer** provides modern type-hint approach but less control than Click

## Open Questions

### 1: CLI Name: `monoquant` vs `mq` vs `mono-quant`

**What we know:**
- Full name is clearer but longer to type
- Short name is quicker but may conflict
- Can provide both via entry points

**Recommendation:**
Use `monoquant` as primary, `mq` as alias (documented but secondary). Package name is `mono-quant` but CLI command uses no hyphens for easier typing.

### 2: Python API Progress Bar Behavior

**What we know:**
- CLI should always show progress
- Library use should be silent by default
- Need a way to enable progress in Python API

**Recommendation:**
Add `show_progress: bool = False` parameter to `quantize()`. Default is silent for library use. Users can opt-in.

### 3: Error Codes for CLI Exit

**What we know:**
- Standard: 0 = success, non-zero = failure
- Specific codes help with scripting
- Click uses `sys.exit()` with exit codes

**Recommendation:**
```
0: Success
1: General error
2: Invalid input/configuration
3: Quantization failed
4: Validation failed
5: File I/O error
```

## Sources

### Primary (HIGH confidence)

- [Click 8.1.x Documentation](https://click.palletsprojects.com/) - Official Click docs, command groups, parameters
- [tqdm Documentation](https://tqdm.github.io/) - Official tqdm docs, usage patterns, overhead stats
- [Python Entry Points Specification](https://packaging.python.org/specifications/entry-points/) - Official Python packaging spec for console_scripts
- [Creating and Packaging Command-Line Tools](https://packaging.python.org/en/latest/guides/creating-command-line-tools/) - Official guide for CLI packaging
- [Rich Documentation](https://rich.readthedocs.io/) - Rich progress display and formatting
- [Typer Documentation](https://typer.tiangolo.com/) - Type-hint-based CLI built on Click

### Secondary (MEDIUM confidence)

- [How to Build CLI Applications with Click in Python (2026)](https://oneuptime.com/blog/post/2026-01-30-python-click-cli-applications/view) - Recent Click tutorial with 2026 content
- [Develop python CLI with subcommands using Click](https://dev.to/dr-cloudycoder/develop-python-cli-with-subcommands-using-click-4892) - Subcommand examples
- [Real Python: Click and Python](https://realpython.com/python-click/) - Comprehensive Click tutorial
- [KDNuggets: Building Command Line Apps with Click](https://www.kdnuggets.com/building-command-line-apps-in-python-with-click) - CLI group patterns
- [Best Practices for Python Exceptions](https://medium.com/@ghoshsiddharth25/exception-handling-in-python-writing-safer-production-ready-code-ea6a78dd1b44) - 2026 article on exception patterns
- [Python Error Handling Best Practices](https://dev.to/amaresh_adak/python-error-handling-best-practices-3ba) - Exception hierarchy guidance

### Tertiary (LOW confidence)

- [Best Python Libraries for CLI](https://levelup.gitconnected.com/best-python-libraries-for-command-line-interface-cli-development-31f7894a85aa) - Comparison posts (verify with official docs)
- Various blog posts on tqdm vs rich - Use for trends, verify specifics

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - Click and tqdm are industry standards, verified with official docs
- Architecture: HIGH - Based on Click 8.1.x official patterns and Python packaging standards
- Pitfalls: HIGH - Verified from multiple sources, common issues well-documented

**Research date:** 2026-02-03
**Valid until:** 2026-03-03 (30 days - CLI ecosystem stable, but Typer/Rich evolving)

**Key recommendations:**
1. **Click** for CLI (mature, stable, standard)
2. **tqdm** for progress bars (minimal deps, CI-friendly)
3. **dataclass** for Result object (built-in, type-safe)
4. **console_scripts** entry point for CLI installation
5. Custom exception hierarchy with actionable messages

---

*Phase: 04-user-interfaces*
*Research completed: 2026-02-03*
