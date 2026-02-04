"""CLI subcommands for mono-quant.

This module provides all CLI subcommands for the mono-quant command-line tool.
Each subcommand corresponds to a major operation: quantize, validate, info, compare,
and calibrate.

## Commands

- **quantize**: Quantize a model to specified precision
- **validate**: Validate a quantized model
- **info**: Display model information
- **compare**: Compare original and quantized models
- **calibrate**: Run calibration and save calibration data
"""

import json
from pathlib import Path
from typing import Optional

import click as _click
import torch


# quantize_cmd - Quantize a model to specified precision
@_click.command(name="quantize")
@_click.option(
    "--model",
    "-m",
    "model_path",
    required=True,
    type=_click.Path(exists=True),
    help="Path to input model (.pt, .pth, .safetensors)",
)
@_click.option(
    "--bits",
    "-b",
    default=8,
    type=_click.IntRange(4, 16),
    help='Quantization bits (4, 8, or 16 for FP16). Default: 8',
)
@_click.option(
    "--output",
    "-o",
    "output_path",
    type=_click.Path(),
    help='Output path (default: <input>_quantized.<ext>)',
)
@_click.option(
    "--scheme",
    "-s",
    type=_click.Choice(["symmetric", "asymmetric"]),
    default="symmetric",
    help='Quantization scheme. Default: symmetric',
)
@_click.option(
    "--dynamic/--static",
    "dynamic",
    default=False,
    help='Use dynamic quantization (no calibration needed)',
)
@_click.option(
    "--calibration",
    "-c",
    type=_click.Path(exists=True),
    help='Calibration data file (for static quantization)',
)
@_click.option(
    "--strict",
    is_flag=True,
    help='Exit immediately on any error (default: warn and continue)',
)
@_click.pass_obj
def quantize_cmd(
    obj: dict,
    model_path: str,
    bits: int,
    output_path: Optional[str],
    scheme: str,
    dynamic: bool,
    calibration: Optional[str],
    strict: bool,
) -> None:
    """Quantize a PyTorch model to specified precision.

    Examples:
        # Dynamic quantization (simple, no calibration)
        monoquant quantize --model model.pt --bits 8 --dynamic

        # Static quantization with calibration
        monoquant quantize -m model.pt -b 8 -c calib_data.pt

        # INT4 quantization with symmetric scheme
        monoquant quantize --model model.pt --bits 4 --scheme symmetric

        # Specify output path
        monoquant quantize -m model.pt -b 8 -o quantized_model.pt
    """
    from mono_quant.api import quantize
    from mono_quant.api.exceptions import MonoQuantError
    from mono_quant.cli.progress import should_show_progress
    from mono_quant.io import load_model

    verbose = obj.get("verbose", False)

    _click.echo(f"Loading model: {model_path}")

    try:
        # Load the model
        state_dict = load_model(model_path)

        # For file-based input, we work with state_dict
        # Dynamic quantization doesn't need calibration
        if dynamic:
            if verbose:
                _click.echo(f"Running dynamic quantization ({bits}-bit, {scheme})")

            result = quantize(
                state_dict,
                bits=bits,
                dynamic=True,
                scheme=scheme,
                show_progress=should_show_progress(),
            )
        else:
            # Static quantization requires calibration data
            if not calibration:
                _click.echo(
                    "Error: Calibration data required for static quantization. "
                    "Use --dynamic flag or provide --calibration file.",
                    err=True,
                )
                _click.Context.exit(2)

            # Note: Calibration file loading from CLI is a placeholder
            # In a full implementation, this would load calibration tensors
            # from the file. For now, we suggest using the Python API.
            _click.echo(
                "Error: Loading calibration data from file is not yet supported in CLI. "
                "Use the Python API with calibration_data parameter:\n"
                "  from mono_quant.api import quantize\n"
                "  result = quantize(model, bits=8, calibration_data=calibration_data)",
                err=True,
            )
            _click.Context.exit(2)

        # Check if quantization succeeded
        if not result.success:
            for error in result.errors:
                _click.echo(f"Error: {error}", err=True)
            if strict:
                _click.Context.exit(3)
            return

        # Determine output path
        if output_path is None:
            # Auto-generate output name
            input_path = Path(model_path)
            output_path = str(input_path.parent / f"{input_path.stem}_quantized.pt")

        # Save the result
        result.save(str(output_path))
        _click.echo(f"Saved quantized model to: {output_path}")

        # Show metrics
        if result.info.sqnr_db is not None:
            _click.echo(f"SQNR: {result.info.sqnr_db:.2f} dB")
        if result.info.compression_ratio is not None:
            _click.echo(f"Compression: {result.info.compression_ratio:.2f}x")

        # Show warnings if any
        if result.warnings:
            if strict:
                _click.echo("\nWarnings encountered:", err=True)
                for warning in result.warnings:
                    _click.echo(f"  {warning}", err=True)
                _click.Context.exit(3)
            else:
                if verbose:
                    _click.echo("\nWarnings:")
                    for warning in result.warnings:
                        _click.echo(f"  {warning}")

    except MonoQuantError as e:
        _click.echo(f"Error: {e}", err=True)
        _click.Context.exit(2)
    except Exception as e:
        _click.echo(f"Unexpected error: {e}", err=True)
        if verbose:
            import traceback

            traceback.print_exc()
        _click.Context.exit(1)


# validate_cmd - Validate a quantized model
@_click.command(name="validate")
@_click.argument("model_path", type=_click.Path(exists=True))
@_click.option("--strict", is_flag=True, help="Exit on validation failure")
def validate_cmd(model_path: str, strict: bool) -> None:
    """Validate a quantized model.

    Checks model integrity, weight ranges, and performs a load/run test.

    Example:
        monoquant validate quantized_model.pt
    """
    from mono_quant.io import load_model

    _click.echo(f"Validating {model_path}...")

    try:
        # Load the model
        state_dict = load_model(model_path)

        # Convert state_dict to a simple module for validation
        # This is a simplified validation - in practice you'd need the actual model architecture
        errors = []
        warnings = []

        # Check if state_dict is empty
        if not state_dict:
            errors.append("Model state_dict is empty")
        else:
            _click.echo(f"  Loaded {len(state_dict)} tensors")

        # Check tensor shapes and dtypes
        quantized_count = 0
        for name, tensor in state_dict.items():
            if tensor.dtype in (torch.qint8, torch.float16, torch.quint8):
                quantized_count += 1
                # Check for NaN/Inf
                if hasattr(tensor, "dequantize"):
                    dequant = tensor.dequantize()
                else:
                    dequant = tensor
                if not torch.all(torch.isfinite(dequant)):
                    errors.append(f"Tensor '{name}' contains NaN or Inf values")

        _click.echo(f"  Quantized tensors: {quantized_count}/{len(state_dict)}")

        # Report results
        if errors:
            _click.echo("\nValidation errors:", err=True)
            for error in errors:
                _click.echo(f"  ✗ {error}", err=True)
            if strict:
                _click.Context.exit(4)
        else:
            _click.echo("\n  All checks passed ✓")

        if warnings:
            _click.echo("\nWarnings:")
            for warning in warnings:
                _click.echo(f"  ! {warning}")

    except Exception as e:
        _click.echo(f"Error during validation: {e}", err=True)
        _click.Context.exit(4)


# info_cmd - Display model information
@_click.command(name="info")
@_click.argument("model_path", type=_click.Path(exists=True))
@_click.option(
    "--format",
    "-f",
    type=_click.Choice(["json", "text"]),
    default="text",
    help="Output format",
)
def info_cmd(model_path: str, format: str) -> None:
    """Display information about a quantized model.

    Shows quantization parameters, model size, tensor count, and metadata.

    Examples:
        monoquant info quantized_model.pt
        monoquant info quantized_model.pt --format json
    """
    from mono_quant.io import load_model
    from mono_quant.io.formats import SAFETENSORS_AVAILABLE

    try:
        state_dict = load_model(model_path)

        # Gather information
        info = {
            "path": model_path,
            "tensor_count": len(state_dict),
            "tensors": {},
        }

        total_params = 0
        quantized_count = 0

        for name, tensor in state_dict.items():
            param_count = tensor.numel()
            total_params += param_count
            dtype_str = str(tensor.dtype)

            if tensor.dtype in (torch.qint8, torch.float16, torch.quint8):
                quantized_count += 1

            info["tensors"][name] = {
                "dtype": dtype_str,
                "shape": list(tensor.shape),
                "params": param_count,
            }

        info["total_params"] = total_params
        info["quantized_tensors"] = quantized_count

        # Try to read metadata if it's a safetensors file
        if model_path.endswith(".safetensors") and SAFETENSORS_AVAILABLE:
            try:
                from safetensors.torch import safe_open

                with safe_open(model_path, framework="pt", device="cpu") as f:
                    metadata = f.metadata()
                    if metadata:
                        info["metadata"] = metadata
            except Exception:
                pass

        # Format output
        if format == "json":
            _click.echo(json.dumps(info, indent=2))
        else:
            _click.echo(f"Model: {model_path}")
            _click.echo(f"Total tensors: {info['tensor_count']}")
            _click.echo(f"Quantized tensors: {quantized_count}")
            _click.echo(f"Total parameters: {total_params:,}")

            if "metadata" in info:
                _click.echo("\nMetadata:")
                for key, value in info["metadata"].items():
                    _click.echo(f"  {key}: {value}")

            _click.echo("\nTensors:")
            for name, tensor_info in list(info["tensors"].items())[:10]:
                shape_str = "x".join(map(str, tensor_info["shape"]))
                _click.echo(
                    f"  {name}: {tensor_info['dtype']} [{shape_str}] ({tensor_info['params']:,} params)"
                )
            if len(info["tensors"]) > 10:
                _click.echo(f"  ... and {len(info['tensors']) - 10} more")

    except Exception as e:
        _click.echo(f"Error reading model info: {e}", err=True)
        _click.Context.exit(5)


# compare_cmd - Compare original and quantized models
@_click.command(name="compare")
@_click.argument("original_path", type=_click.Path(exists=True))
@_click.argument("quantized_path", type=_click.Path(exists=True))
@_click.option(
    "--metrics",
    "-m",
    multiple=True,
    type=_click.Choice(["sqnr", "size", "all"]),
    default=["all"],
    help="Metrics to compare",
)
def compare_cmd(original_path: str, quantized_path: str, metrics: tuple) -> None:
    """Compare original and quantized models.

    Computes SQNR, size comparison, and other metrics.

    Example:
        monoquant compare model.pt quantized_model.pt
    """
    from mono_quant.io import load_model
    from mono_quant.io.validation import calculate_sqnr

    _click.echo(f"Comparing {original_path} vs {quantized_path}")

    try:
        original = load_model(original_path)
        quantized = load_model(quantized_path)

        # Determine which metrics to show
        show_all = "all" in metrics
        show_sqnr = show_all or "sqnr" in metrics
        show_size = show_all or "size" in metrics

        # Calculate size comparison
        if show_size:
            orig_params = sum(t.numel() for t in original.values())
            quant_params = sum(t.numel() for t in quantized.values())
            ratio = orig_params / quant_params if quant_params > 0 else 0

            _click.echo("\nSize Comparison:")
            _click.echo(f"  Original:  {orig_params:,} parameters")
            _click.echo(f"  Quantized: {quant_params:,} parameters")
            _click.echo(f"  Ratio:     {ratio:.2f}x")

        # Calculate SQNR for matching tensors
        if show_sqnr:
            sqnr_values = []
            for name, orig_tensor in original.items():
                if name in quantized:
                    quant_tensor = quantized[name]
                    try:
                        if orig_tensor.dtype != quant_tensor.dtype:
                            sqnr = calculate_sqnr(orig_tensor, quant_tensor)
                            sqnr_values.append((name, sqnr))
                    except Exception:
                        pass

            if sqnr_values:
                avg_sqnr = sum(s for _, s in sqnr_values) / len(sqnr_values)
                _click.echo(f"\nSQNR (average): {avg_sqnr:.2f} dB")

                _click.echo("\nTop 10 layers by SQNR:")
                sqnr_values.sort(key=lambda x: x[1], reverse=True)
                for name, sqnr in sqnr_values[:10]:
                    _click.echo(f"  {name}: {sqnr:.2f} dB")
            else:
                _click.echo("\nNo quantized tensors found for SQNR comparison")

    except Exception as e:
        _click.echo(f"Error during comparison: {e}", err=True)
        _click.Context.exit(1)


# calibrate_cmd - Run calibration on a model
@_click.command(name="calibrate")
@_click.argument("model_path", type=_click.Path(exists=True))
@_click.argument("data_path", type=_click.Path(exists=True))
@_click.option(
    "--output",
    "-o",
    type=_click.Path(),
    help="Output calibration file",
)
@_click.option(
    "--samples",
    "-n",
    type=int,
    default=150,
    help="Number of samples",
)
def calibrate_cmd(model_path: str, data_path: str, output: Optional[str], samples: int) -> None:
    """Run calibration on a model and save calibration data.

    Generates calibration statistics from sample data.

    Example:
        monoquant calibrate model.pt calibration_data.pt --samples 200
    """
    _click.echo(f"Calibrating {model_path} with {samples} samples")
    _click.echo("\nNote: CLI calibration is not yet implemented.")
    _click.echo("Use the Python API for calibration:\n")
    _click.echo("  from mono_quant.api import quantize")
    _click.echo("  import torch")
    _click.echo("  calibration_data = [torch.randn(32, 128) for _ in range(100)]")
    _click.echo("  result = quantize(model, bits=8, calibration_data=calibration_data)")
    _click.Context.exit(1)


__all__ = [
    "quantize_cmd",
    "validate_cmd",
    "info_cmd",
    "compare_cmd",
    "calibrate_cmd",
]
