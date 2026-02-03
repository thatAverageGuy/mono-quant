"""Click CLI main entry point.

This module provides the main Click group for the mono-quant CLI.
It serves as the entry point for both 'monoquant' and 'mq' commands.

## Usage

    $ monoquant --help
    $ monoquant quantize --model model.pt --bits 8 --dynamic
    $ monoquant validate quantized_model.pt
    $ monoquant info quantized_model.pt

## Commands

- **quantize**: Quantize a model to specified precision
- **validate**: Validate a quantized model
- **info**: Display model information
- **compare**: Compare original and quantized models
- **calibrate**: Run calibration and save calibration data

## Global Options

- **--version**: Show version and exit
- **--verbose, -v**: Enable verbose output
"""

import click as _click

# Import all subcommands
# These are imported here to avoid circular dependencies
# and to allow the main CLI to register them
from .commands import (
    calibrate_cmd,
    compare_cmd,
    info_cmd,
    quantize_cmd,
    validate_cmd,
)


@_click.group()
@_click.version_option(version="0.1.0", prog_name="monoquant")
@_click.option("--verbose", "-v", is_flag=True, help="Enable verbose output")
@_click.pass_context
def cli(ctx: _click.Context, verbose: bool) -> None:
    """Mono Quant - Simple, reliable model quantization.

    MonoQuant is a command-line tool for quantizing PyTorch models with minimal
    dependencies. It supports dynamic and static quantization, INT4/INT8/FP16
    precision, and provides validation metrics.

    For detailed help on each command, run:
        monoquant COMMAND --help

    Examples:
        # Dynamic quantization (simple, no calibration)
        monoquant quantize --model model.pt --bits 8 --dynamic

        # Static quantization with calibration data
        monoquant quantize -m model.pt -b 8 -c calibration_data.pt

        # Validate a quantized model
        monoquant validate quantized_model.pt

        # Show model information
        monoquant info quantized_model.pt
    """
    # Ensure ctx.obj exists and is a dict
    ctx.ensure_object(dict)

    # Store verbose flag for use in subcommands
    ctx.obj["verbose"] = verbose


# Register all subcommands
cli.add_command(quantize_cmd)
cli.add_command(validate_cmd)
cli.add_command(info_cmd)
cli.add_command(compare_cmd)
cli.add_command(calibrate_cmd)


__all__ = ["cli"]


if __name__ == "__main__":
    # Allow direct execution for development/testing
    cli()
