# CLI Reference

Mono Quant includes a command-line interface `monoquant` for quick quantization and CI/CD integration.

## Installation

The CLI is installed automatically with Mono Quant:

```bash
pip install mono-quant
```

Verify installation:

```bash
monoquant --version
```

## Overview

```bash
monoquant [OPTIONS] COMMAND [ARGS]...
```

### Global Options

| Option | Short | Description |
|--------|-------|-------------|
| `--version` | | Show version and exit |
| `--verbose` | `-v` | Enable verbose output |
| `--help` | `-h` | Show help message |

## Commands

### `quantize`

Quantize a model.

```bash
monoquant quantize --model MODEL.pt [OPTIONS]
```

#### Options

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--model` | `-m` | *required* | Path to input model file |
| `--output` | `-o` | auto | Output file path (auto: `{input}_quantized.{ext}`) |
| `--bits` | `-b` | `8` | Quantization bits: 4, 8, or 16 |
| `--dynamic` | `-d` | `false` | Use dynamic quantization (no calibration) |
| `--scheme` | `-s` | `symmetric` | Quantization scheme: `symmetric` or `asymmetric` |
| `--observer` | | `minmax` | Calibration observer: `minmax`, `movingaverage`, `histogram` |
| `--strict` | | `false` | Exit immediately on errors (CI/CD mode) |
| `--help` | | | Show command help |

#### Examples

**Dynamic INT8 quantization:**

```bash
monoquant quantize --model model.pt --bits 8 --dynamic
```

**Static INT4 quantization:**

```bash
# Note: Calibration data must be prepared via Python API first
monoquant quantize --model model.pt --bits 4 --observer histogram
```

**Custom output path:**

```bash
monoquant quantize --model model.pt --output models/quantized/model_int8.pt
```

**CI/CD mode with strict error handling:**

```bash
monoquant quantize --model model.pt --bits 8 --dynamic --strict
```

### `validate`

Validate an existing quantized model.

```bash
monoquant validate --model MODEL.pt [OPTIONS]
```

#### Options

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--model` | `-m` | *required* | Path to quantized model file |
| `--reference` | `-r` | | Path to original model (for comparison) |
| `--help` | | | Show command help |

#### Examples

```bash
monoquant validate --model model_quantized.pt

# With original model for comparison
monoquant validate --model model_quantized.pt --reference model.pt
```

### `info`

Display information about a model file.

```bash
monoquant info --model MODEL.pt
```

#### Examples

```bash
monoquant info --model model_quantized.pt
```

Output includes:
- Model size
- Quantization parameters (bits, scheme)
- Format (PyTorch/Safetensors)
- Metadata (if available)

### `compare`

Compare two model files.

```bash
monoquant compare --original ORIGINAL.pt --quantized QUANTIZED.pt
```

#### Examples

```bash
monoquant compare --original model.pt --quantized model_quantized.pt
```

Output includes:
- Size comparison
- Compression ratio
- Format differences

### `calibrate`

Prepare calibration data for static quantization.

```bash
monoquant calibrate --model MODEL.pt --data DATA_PATH [OPTIONS]
```

#### Options

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--model` | `-m` | *required* | Path to model file |
| `--data` | `-d` | *required* | Path to calibration data |
| `--output` | `-o` | `calibration_data.pt` | Output calibration data file |
| `--samples` | `-n` | `150` | Number of calibration samples |
| `--help` | | | Show command help |

**Note:** Calibration data preparation typically requires Python API. This command provides basic functionality.

## Exit Codes

| Code | Meaning |
|------|---------|
| `0` | Success |
| `1` | General error |
| `2` | Configuration error |
| `3` | Quantization error |
| `4` | Validation error |
| `5` | I/O error |

## CI/CD Integration

### GitHub Actions Example

```yaml
name: Quantize Model

on: [push]

jobs:
  quantize:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python: '3.10'
      - name: Install Mono Quant
        run: pip install mono-quant
      - name: Quantize model
        run: monoquant quantize --model model.pt --bits 8 --dynamic --strict
```

### GitLab CI Example

```yaml
quantize:
  image: python:3.10
  script:
    - pip install mono-quant
    - monoquant quantize --model model.pt --bits 8 --dynamic --strict
  artifacts:
    paths:
      - model_quantized.pt
```

## Progress Bars

The CLI automatically shows progress bars for:
- Calibration processing
- Quantization progress
- Model loading/saving

Progress bars are **disabled** in CI environments (detected via `CI` environment variable).

To manually disable:

```bash
# Set CI environment variable
CI=1 monoquant quantize --model model.pt --bits 8

# Or redirect output
monoquant quantize --model model.pt --bits 8 | cat
```

## Aliases

The `mq` command is an alias for `monoquant`:

```bash
mq quantize --model model.pt --bits 8
mq --version
```

## Next Steps

- [**Commands**](commands.md) - Detailed command reference
- [**Examples: CI/CD**](../examples/cicd.md) - CI/CD pipeline examples
