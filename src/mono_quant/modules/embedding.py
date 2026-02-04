
"""Quantized Embedding module implementation."""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================================================
# Quantized Embedding Module
# ============================================================================

class QuantizedEmbedding(nn.Module):
    """
    An Embedding layer with quantized weights for inference.

    This module wraps quantized weights (typically INT8) and handles
    dequantization during the forward pass. The weights are quantized
    once during initialization and cached for efficient inference.

    Note: INT4 quantization is NOT supported for embeddings due to
    accuracy concerns. Embeddings map discrete tokens to continuous
    spaces, and INT4's aggressive quantization causes significant
    semantic degradation. Use INT8 or FP16 instead.

    Args:
        num_embeddings: Size of the dictionary of embeddings.
        embedding_dim: The size of each embedding vector.
        padding_idx: If specified, the entries at padding_idx do not
                    contribute to the gradient.
        max_norm: If given, each embedding vector with norm larger than
                 max_norm is renormalized to have norm max_norm.
        norm_type: Type of the used p-norm (default: 2.0).
        scale_grad_by_freq: If True, gradient will be scaled by inverse of
                           frequency of the tokens in the mini-batch.
        sparse: If True, gradient w.r.t. weight_matrix will be a sparse tensor.
        dtype: Quantization dtype (torch.qint8 or torch.float16).
        symmetric: If True, use symmetric quantization.

    Attributes:
        num_embeddings: Number of embeddings in the dictionary.
        embedding_dim: Dimension of each embedding vector.
        _quantized_weight: Cached quantized weight tensor.
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: Optional[int] = None,
        max_norm: Optional[float] = None,
        norm_type: float = 2.0,
        scale_grad_by_freq: bool = False,
        sparse: bool = False,
        dtype: torch.dtype = torch.qint8,
        symmetric: bool = False,
    ):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.scale_grad_by_freq = scale_grad_by_freq
        self.sparse = sparse
        self.dtype = dtype
        self.symmetric = symmetric

        # Store quantized weight (cached after quantization)
        self._quantized_weight: Optional[torch.Tensor] = None

        # Placeholder for the original weight shape
        self._weight_shape = (num_embeddings, embedding_dim)

    @classmethod
    def from_embedding(
        cls,
        module: nn.Embedding,
        symmetric: bool = False,
    ) -> "QuantizedEmbedding":
        """
        Create a QuantizedEmbedding from an existing nn.Embedding module.

        This factory method quantizes the weights from the source Embedding
        module and creates a new QuantizedEmbedding instance.

        Args:
            module: Source nn.Embedding module to quantize.
            symmetric: If True, use symmetric quantization. Default is False.

        Returns:
            A new QuantizedEmbedding instance with quantized weights.

        Raises:
            TypeError: If module is not an nn.Embedding instance.
        """
        if not isinstance(module, nn.Embedding):
            raise TypeError(
                f"Expected nn.Embedding, got {type(module).__name__}. "
                f"QuantizedEmbedding.from_embedding only supports nn.Embedding layers."
            )

        # Create new QuantizedEmbedding with same configuration
        q_emb = cls(
            num_embeddings=module.num_embeddings,
            embedding_dim=module.embedding_dim,
            padding_idx=module.padding_idx,
            max_norm=module.max_norm,
            norm_type=module.norm_type,
            scale_grad_by_freq=module.scale_grad_by_freq,
            sparse=module.sparse,
            symmetric=symmetric,
        )

        # Quantize the weight using per-tensor quantization (not per-channel)
        # Note: Embeddings use per-tensor quantization because each embedding
        # vector is independent
        from mono_quant.core.quantizers import quantize_weight_int8
        q_emb._quantized_weight = quantize_weight_int8(
            module.weight.data, symmetric=symmetric, axis=0
        )

        return q_emb

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with dequantized weights.

        Args:
            input: Input tensor of indices (any shape).

        Returns:
            Output tensor with shape (*input.shape, embedding_dim).
        """
        # Lazy quantization check
        if self._quantized_weight is None:
            raise RuntimeError(
                "QuantizedEmbedding has no quantized weights. "
                "Use from_embedding() or set _quantized_weight directly."
            )

        # Dequantize weight for computation
        from mono_quant.core.quantizers import dequantize_weight
        weight = dequantize_weight(self._quantized_weight)

        # Standard embedding lookup
        return F.embedding(
            input, weight,
            padding_idx=self.padding_idx,
            max_norm=self.max_norm,
            norm_type=self.norm_type,
            scale_grad_by_freq=self.scale_grad_by_freq,
            sparse=self.sparse,
        )

    @property
    def weight(self) -> Optional[torch.Tensor]:
        """
        Return the dequantized weight (for compatibility).

        This property provides compatibility with code that expects
        a weight attribute on Embedding modules.

        Returns:
            Dequantized weight tensor, or None if not quantized yet.
        """
        if self._quantized_weight is None:
            return None
        from mono_quant.core.quantizers import dequantize_weight
        return dequantize_weight(self._quantized_weight)

    def extra_repr(self) -> str:
        """Return extra representation string."""
        return (
            f"num_embeddings={self.num_embeddings}, "
            f"embedding_dim={self.embedding_dim}, "
            f"dtype={self.dtype}, "
            f"symmetric={self.symmetric}"
        )


def quantize_embedding_module(
    module: nn.Embedding,
    dtype: torch.dtype = torch.qint8,
    symmetric: bool = False,
) -> QuantizedEmbedding:
    """
    Quantize an nn.Embedding module to a QuantizedEmbedding.

    This function creates a new QuantizedEmbedding module with quantized
    weights copied from the source Embedding module. The original module
    is not modified.

    Note: INT4 quantization is NOT supported for embeddings. Use INT8 or FP16.

    Args:
        module: Source nn.Embedding module to quantize.
        dtype: Target quantization dtype. Only torch.qint8 is supported.
        symmetric: If True, use symmetric quantization (zero_point = 0).
                   If False, use asymmetric quantization. Default is False.

    Returns:
        A new QuantizedEmbedding instance with quantized weights.

    Raises:
        TypeError: If module is not an nn.Embedding instance.

    Examples:
        >>> import torch.nn as nn
        >>> from mono_quant.modules import quantize_embedding_module
        >>> emb = nn.Embedding(1000, 128)
        >>> q_emb = quantize_embedding_module(emb, symmetric=False)
        >>> assert isinstance(q_emb, QuantizedEmbedding)
    """
    if not isinstance(module, nn.Embedding):
        raise TypeError(
            f"Expected nn.Embedding, got {type(module).__name__}. "
            f"quantize_embedding_module only supports nn.Embedding layers."
        )

    # Use the factory method to create QuantizedEmbedding
    return QuantizedEmbedding.from_embedding(module, symmetric=symmetric)
