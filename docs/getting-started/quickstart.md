# Quick Start

Get started with Mono Quant in 5 minutes.

## Step 1: Install Mono Quant

```bash
pip install mono-quant
```

## Step 2: Prepare Your Model

Mono Quant works with any PyTorch model. Load it however you prefer:

```python
import torch.nn as nn
from mono_quant import quantize

# Option A: Your own model
model = MyModel()

# Option B: HuggingFace model
# from transformers import AutoModel
# model = AutoModel.from_pretrained("bert-base-uncased")

# Option C: Any PyTorch model from anywhere
```

## Step 3: Quantize

Choose your quantization mode:

### Dynamic INT8 (Fastest, No Data)

```python
from mono_quant import quantize

result = quantize(
    model,
    bits=8,
    dynamic=True  # No calibration data needed
)

# Access the quantized model
quantized_model = result.model

# Save it
result.save("my_model_quantized.pt")
```

### Static INT8 (Best Accuracy, Requires Data)

```python
# Prepare calibration data (list of tensors)
calibration_data = [torch.randn(1, 3, 224, 224) for _ in range(100)]

result = quantize(
    model,
    bits=8,
    dynamic=False,  # Static quantization
    calibration_data=calibration_data
)
```

### INT4 (Maximum Compression)

```python
result = quantize(
    model,
    bits=4,
    dynamic=False,
    calibration_data=calibration_data
)
```

## Step 4: Inspect Results

Mono Quant provides detailed metrics:

```python
print(f"Original size: {result.info.original_size_mb:.2f} MB")
print(f"Quantized size: {result.info.quantized_size_mb:.2f} MB")
print(f"Compression: {result.info.compression_ratio:.2f}x")
print(f"SQNR: {result.info.sqnr_db:.2f} dB")
print(f"Skipped layers: {len(result.info.skipped_layers)}")
```

Example output:

```
Original size: 438.56 MB
Quantized size: 11.75 MB
Compression: 37.32x
SQNR: 28.45 dB
Skipped layers: 2
```

## Step 5: Use the Quantized Model

The quantized model works just like a regular PyTorch model:

```python
# Load it later
from mono_quant import load_model

loaded_model = load_model("my_model_quantized.pt")

# Run inference
output = loaded_model(input_tensor)
```

## CLI Usage

Prefer command-line? Mono Quant includes a CLI:

```bash
# Dynamic quantization
monoquant quantize --model model.pt --bits 8 --dynamic

# Static quantization with calibration data
# (calibration data must be prepared in Python first)

# With custom output path
monoquant quantize --model model.pt --bits 8 --output model_quantized.pt

# Show progress
monoquant quantize --model model.pt --bits 8 --dynamic --verbose
```

## Common Patterns

### Quantize and Validate

```python
result = quantize(model, bits=8, dynamic=True)

# Validate the quantized model
validation = result.validate()
print(f"Load test passed: {validation.load_test_passed}")
```

### Suppress Warnings

```python
result = quantize(
    model,
    bits=4,
    calibration_data=data,
    accuracy_warning="ignore"  # Don't warn about aggressive quantization
)
```

### Skip Specific Layers

```python
result = quantize(
    model,
    bits=4,
    calibration_data=data,
    modules_to_skip=["embeddings", "layer_norm"]
)
```

## Next Steps

- [**Basic Usage**](basic-usage.md) - More detailed examples
- [**Quantization Modes**](../user-guide/modes.md) - Deep dive into modes
- [**CLI Reference**](../cli/commands.md) - Command-line tool guide
