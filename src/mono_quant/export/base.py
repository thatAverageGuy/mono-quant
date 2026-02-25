"""Base exporter abstract class."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Union

import torch.nn as nn


class BaseExporter(ABC):
    """Abstract base class for model exporters."""

    @abstractmethod
    def export(
        self,
        model: nn.Module,
        path: Union[str, Path],
        **kwargs: Any,
    ) -> None:
        """Export a model to the target format."""
        ...

    @abstractmethod
    def validate_compatibility(self, model: nn.Module) -> None:
        """Check that the model is compatible with this exporter."""
        ...

    @abstractmethod
    def build_metadata(self, model: nn.Module, **kwargs: Any) -> Dict[str, Any]:
        """Build metadata dictionary for the exported model."""
        ...
