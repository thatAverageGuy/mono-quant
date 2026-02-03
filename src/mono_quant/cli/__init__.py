"""Command-line interface for mono-quant.

This package provides the CLI commands for model quantization. The main entry
point is the `cli` Click group which is registered as a console script.

## Usage

From the command line:
    $ monoquant --help
    $ monoquant quantize --model model.pt --bits 8 --dynamic
    $ monoquant validate quantized_model.pt

As a Python API:
    >>> from mono_quant.cli import cli
    >>> cli(['quantize', '--model', 'model.pt', '--bits', '8', '--dynamic'])

## Commands

- **quantize**: Quantize a model to specified precision
- **validate**: Validate a quantized model
- **info**: Display model information
- **compare**: Compare original and quantized models
- **calibrate**: Run calibration and save calibration data
"""

from .main import cli

__all__ = ["cli"]
