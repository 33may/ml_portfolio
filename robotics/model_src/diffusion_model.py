import torch
from torch import nn


class SinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, dim):
        super(SinusoidalPositionalEmbedding, self).__init__()
        self.dim = dim

    def forward(self,
                pos: int | torch.Tensor,
                dim: int,
                base: float = 10000.0,
        ) -> torch.Tensor:
        """
        Compute a sinusoidal positional embedding for a single position.

        Args:
        pos   : int or 0-D tensor – position index.
        dim   : int – embedding dimension (must be even).
        base  : float – base for the exponential frequency scaling.

        Returns:
        Tensor of shape (dim,) containing the positional embedding.
        """
        pos = torch.as_tensor(pos, dtype=torch.float)        # if pos was int
        if pos.dim() == 0:                                   # scalar → (1,)
            pos = pos.unsqueeze(0)

        # Half of the dimensions will be sine, half cosine
        half_dim = dim // 2

        # Exponent term:  base^{2i/dim}  for i = 0 .. half_dim-1
        exponent = torch.arange(half_dim, dtype=torch.float)
        div_term = base ** (2 * exponent / dim)  # (half_dim,)

        # Compute the value: pos / base^{2i/dim}
        value = pos.unsqueeze(-1) / div_term                    # (num_pos, half_dim)

        emb = torch.empty(pos.size(0), dim, dtype=torch.float)
        emb[:, 0::2] = torch.sin(value)              # even indices  -> sin
        emb[:, 1::2] = torch.cos(value)              # odd  indices -> cos
        return emb
