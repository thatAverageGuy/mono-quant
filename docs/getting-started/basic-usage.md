# Basic Usage

Common usage patterns for Mono Quant.

## Loading Models

Mono Quant accepts models in three formats:

### Option 1: nn.Module (Recommended)

```python
from mono_quant import quantize

result = quantize(model, bits=8, dynamic=True)
```

### Option 2: state_dict

```python
# Provide architecture when using state_dict
result = quantize(
    "model_state_dict.pt",
    bits=8,
    dynamic=True,
    architecture=MyModelClass  # Required for state_dict
)
```

### Option 3: File Path

```python
# From PyTorch file
result = quantize("model.pt", bits=8, dynamic=True)

# From Safetensors file
result = quantize("model.safetensors", bits=8, dynamic=True)
```

## Quantization Modes

### Dynamic Quantization

**Use when:** You need fast quantization without data

```python
result = quantize(model, bits=8, dynamic=True)
```

**Characteristics:**
- No calibration data required
- Fastest quantization
- Good for inference speedup
- Lower accuracy than static

### Static Quantization

**Use when:** You need best accuracy

```python
calibration_data = [torch.randn(1, 3, 224, 224) for _ in range(150)]

result = quantize(
    model,
    bits=8,
    dynamic=False,
    calibration_data=calibration_data
)
```

**Characteristics:**
- Requires calibration data (100-200 samples)
- Slower quantization
- Best accuracy
- Better compression

## Choosing Bits

### INT8 (Default)

```python
result = quantize(model, bits=8, dynamic=True)
```

**Use when:** Balanced compression and accuracy

**Characteristics:**
- ~4x compression
- Minimal accuracy loss
- Fast inference
- Universal compatibility

### INT4

```python
result = quantize(model, bits=4, calibration_data=data)
```

**Use when:** Maximum compression needed

**Characteristics:**
- ~8x compression
- Potential accuracy loss
- Uses group-wise scaling
- Sensitive layers skipped automatically

### FP16

```python
result = quantize(model, bits=16, dynamic=True)
```

**Use when:** Memory reduction without precision loss

**Characteristics:**
- 2x compression
- Minimal accuracy impact
- Faster on GPUs
- Larger model size

## Saving and Loading

### Save Quantized Model

```python
result = quantize(model, bits=8, dynamic=True)

# Save to PyTorch format
result.save("model_quantized.pt")

# Save to Safetensors format
result.save("model_quantized.safetensors")
```

### Load Quantized Model

```python
from mono_quant import load_model

# Auto-detects format
loaded_model = load_model("model_quantized.pt")
```

## Error Handling

```python
from mono_quant import quantize, MonoQuantError

try:
    result = quantize(model, bits=5, dynamic=True)
except MonoQuantError as e:
    print(f"Quantization failed: {e}")
    print(f"Suggestion: {e.suggestion}")
```

### Common Errors

| Error | Cause | Solution |
|-------|-------|----------|
| `ValueError: Unsupported bits` | bits not in {4, 8, 16} | Use 4, 8, or 16 |
| `ValueError: architecture required` | state_dict without architecture | Provide architecture parameter |
| `MonoQuantError: Calibration data required` | Static quantization without data | Provide calibration_data or use dynamic=True |

## Working with Warnings

```python
result = quantize(model, bits=4, calibration_data=data)

# Check warnings
for warning in result.info.warnings:
    print(f"Warning: {warning}")
```

### Warning Types

- **Low SQNR** - Quantization quality may be poor
- **All layers quantized** - Consider skipping sensitive layers
- **Low calibration count** - Use more calibration samples

## Validation

```python
result = quantize(model, bits=8, dynamic=True)

# Validate automatically
validation = result.validate()

print(f"SQNR: {validation.sqnr_db:.2f} dB")
print(f"Compression: {validation.compression_ratio:.2f}x")
print(f"Load test: {validation.load_test_passed}")
```

## Next Steps

- [**Quantization Modes**](../user-guide/modes.md) - Deep dive
- [**Calibration**](../user-guide/calibration.md) - Calibration guide
- [**Examples**](../examples/dynamic-int8.md) - Real-world examples
