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

### `export`

Export a model to ONNX, GPTQ, or GGUF format.

```bash
monoquant export --model MODEL.pt --output OUTPUT [OPTIONS]
```

Format is **auto-detected** from the output path extension when `--format` is omitted:

| Extension / path type | Auto-detected format |
|-----------------------|----------------------|
| `.onnx` | ONNX |
| `.gguf` | GGUF |
| directory (no extension) | GPTQ |

#### Options

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--model` | `-m` | *required* | Path to model file (torch.save format) |
| `--output` | `-o` | *required* | Output path or directory |
| `--format` | `-f` | auto | Format: `onnx`, `gptq`, `gguf` |
| `--list-formats` | | | Print available formats and exit |
| `--opset` | | `14` | ONNX opset version (ONNX only) |
| `--validate` | | `none` | Post-export validation: `none`, `load`, `full` (ONNX only) |
| `--group-size` | | `128` | Quantization group size (GPTQ only) |
| `--sym` | | `false` | Symmetric quantization (GPTQ only) |
| `--architecture` | | auto | GGUF architecture string (GGUF only) |
| `--config` | | | Path to HuggingFace config.json (GGUF only) |
| `--model-param` | | | `KEY=VALUE` override, repeatable (GGUF only) |

#### Examples

**List available formats:**

```bash
monoquant export --list-formats
```

**ONNX — auto-detected from extension:**

```bash
monoquant export -m q_model.pt -o model.onnx
```

**ONNX — explicit format + opset:**

```bash
monoquant export -m q_model.pt -o model.onnx --format onnx --opset 17 --validate load
```

**GPTQ:**

```bash
monoquant export -m q_model.pt -o ./gptq_dir/ --format gptq --group-size 64 --sym
```

**GGUF — auto-detected from extension:**

```bash
monoquant export -m q_model.pt -o model.gguf --config ./config.json
```

**GGUF — explicit architecture:**

```bash
monoquant export -m q_model.pt -o ./gguf_dir/ --format gguf --architecture llama \
    --model-param num_hidden_layers=32 --model-param hidden_size=4096
```

---

### `convert`

Convert a model to a different quantization bit-width via dynamic re-quantization.

```bash
monoquant convert INPUT OUTPUT --bits BITS
```

!!! note
    This uses dynamic re-quantization (dequantize → re-quantize). No calibration data
    required, but accuracy is lower than re-quantizing from the original FP32 model.

#### Arguments

| Argument | Description |
|----------|-------------|
| `INPUT` | Path to model file (.pt, torch.save format) |
| `OUTPUT` | Destination path for converted model |

#### Options

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--bits` | `-b` | *required* | Target bit-width: 4, 8, or 16 |

#### Examples

```bash
monoquant convert q_model_int8.pt q_model_int4.pt --bits 4
monoquant convert q_model_int8.pt q_model_fp16.pt --bits 16
```

---

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
