# Installation

Install Mono Quant via pip:

```bash
pip install mono-quant
```

## Requirements

- **Python**: 3.8 or higher
- **PyTorch**: 2.0 or higher

Mono Quant only requires `torch` as a dependency. All other features are optional or use standard library.

## Optional Dependencies

For additional functionality, you can install optional dependencies:

```bash
# Safetensors format support
pip install mono-quant[safetensors]

# Development dependencies (testing, building)
pip install mono-quant[dev]
```

### Optional Dependencies Breakdown

| Feature | Package | Install |
|---------|---------|---------|
| Safetensors format | `safetensors` | `pip install safetensors` |
| CLI progress bars | `tqdm` | `pip install tqdm` |
| Click framework | `click>=8.1` | `pip install click>=8.1` |

## Verify Installation

```bash
# Check package version
python -c "import mono_quant; print(mono_quant.__version__)"

# Test CLI
monoquant --version

# Run quick test
python -c "from mono_quant import quantize; print('Mono Quant ready!')"
```

## Development Installation

If you want to contribute or run from source:

```bash
# Clone repository
git clone https://github.com/thatAverageGuy/mono-quant.git
cd mono-quant

# Install in editable mode
pip install -e ".[dev]"

# Run tests
pytest
```

## Next Steps

- [**Quick Start**](quickstart.md) - Your first quantization
- [**Basic Usage**](basic-usage.md) - Common patterns and examples
