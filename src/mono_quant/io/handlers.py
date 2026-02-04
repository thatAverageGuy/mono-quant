"""Model input/output handlers for model-agnostic quantization.

This module provides functions to detect, prepare, and validate models for
quantization, supporting both nn.Module and state_dict inputs.

Per CONTEXT.md requirement: Always copy the user's model - never modify
the original.
"""

from copy import deepcopy
from typing import Dict, List, Optional, Tuple, Type, Union

import torch
import torch.nn as nn


def _detect_input_format(
    model: Union[nn.Module, Dict[str, torch.Tensor]],
) -> str:
    """
    Detect whether input is an nn.Module or state_dict.

    Args:
        model: Either a PyTorch nn.Module or a state_dict.

    Returns:
        "module" if input is nn.Module, "state_dict" if input is a dict.

    Raises:
        TypeError: If input is neither nn.Module nor dict.

    Examples:
        >>> import torch.nn as nn
        >>> m = nn.Linear(10, 10)
        >>> _detect_input_format(m)
        'module'
        >>> _detect_input_format(m.state_dict())
        'state_dict'
    """
    if isinstance(model, nn.Module):
        return "module"
    elif isinstance(model, dict):
        # Additional check: looks like a state_dict (has tensor values)
        if model and all(isinstance(v, torch.Tensor) for v in model.values()):
            return "state_dict"
        else:
            raise TypeError(
                "Input is a dict but doesn't appear to be a state_dict. "
                "Expected dict with torch.Tensor values."
            )
    else:
        raise TypeError(
            f"Input must be nn.Module or state_dict (dict), got {type(model).__name__}. "
            f"Pass your model directly or use model.state_dict()."
        )


def _prepare_model(
    model: Union[nn.Module, Dict[str, torch.Tensor]],
    architecture: Optional[Type[nn.Module]] = None,
) -> nn.Module:
    """
    Prepare model for quantization by copying and/or loading state_dict.

    IMPORTANT: This function ALWAYS returns a copy of the model to preserve
    the original, as required by CONTEXT.md. Quantization operations will
    modify the model in-place, so the copy protects the user's original.

    Args:
        model: Either a PyTorch nn.Module or a state_dict.
        architecture: Optional nn.Module class. Required when model is a
            state_dict. Ignored when model is an nn.Module.

    Returns:
        A COPY of the model as nn.Module, ready for quantization.

    Raises:
        ValueError: If model is state_dict but architecture is not provided.

    Examples:
        >>> import torch.nn as nn
        >>> original = nn.Linear(10, 10)
        >>> prepared = _prepare_model(original)
        >>> id(original) != id(prepared)  # Always a copy!
        True

        With state_dict:
        >>> state = original.state_dict()
        >>> prepared = _prepare_model(state, architecture=nn.Linear, architecture_kwargs={'in_features': 10, 'out_features': 10})
    """
    input_format = _detect_input_format(model)

    if input_format == "module":
        # Always copy to preserve original (CONTEXT.md requirement)
        model_copy = deepcopy(model)
        return model_copy
    else:  # state_dict
        if architecture is None:
            raise ValueError(
                "State_dict input requires 'architecture' parameter. "
                "Provide the model class (e.g., nn.Linear) so we can instantiate "
                "and load the state_dict. "
                "\nExample: _prepare_model(state_dict, architecture=nn.Linear, architecture_kwargs={'in_features': 10, 'out_features': 10})"
            )

        # Try to instantiate with no args first, then load state_dict
        # If architecture needs args, user must provide them via architecture_kwargs
        try:
            model_instance = architecture()
        except TypeError as e:
            raise ValueError(
                f"Could not instantiate {architecture.__name__}() without arguments. "
                f"Please provide architecture_kwargs with required parameters."
            ) from e

        # Load the state_dict
        model_instance.load_state_dict(model)
        return model_instance


def _validate_model(model: nn.Module) -> List[Tuple[str, str]]:
    """
    Validate model and identify quantizable layers.

    Scans the model for layers that support quantization (Linear, Conv2d).
    Returns a list of (layer_name, layer_type) tuples for the quantization
    pipeline.

    Args:
        model: PyTorch nn.Module to validate.

    Returns:
        List of tuples (layer_name, layer_type) for quantizable layers.

    Examples:
        >>> import torch.nn as nn
        >>> m = nn.Sequential(
        ...     nn.Linear(10, 20),
        ...     nn.ReLU(),
        ...     nn.Conv2d(3, 16, 3)
        ... )
        >>> layers = _validate_model(m)
        >>> len(layers) >= 2  # At least Linear and Conv2d
        True
    """
    quantizable_layers: List[Tuple[str, str]] = []

    # Supported layer types for quantization
    QUANTIZABLE_TYPES = (
        nn.Linear,
        nn.Conv2d,
    )

    for name, module in model.named_modules():
        # Skip the root module
        if name == "":
            continue

        if isinstance(module, QUANTIZABLE_TYPES):
            layer_type = type(module).__name__
            quantizable_layers.append((name, layer_type))

    return quantizable_layers
