import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class CatMultiAttn(nn.Module):
    def __init__(
        self,
        input_time: int,       # T1 + T2
        num_heads: int,
        hidden_dim: int,       # E_h
        use_activation: bool
    ):
        """
        Args:
            input_time (int): Combined time dimension after concatenation (T1 + T2)
            num_heads (int): Number of attention heads
            hidden_dim (int): Hidden output dimension (E_h)
            use_activation (bool): Whether to apply GELU activation
        """
        super().__init__()
        self.use_activation = use_activation

        self.attn = nn.MultiheadAttention(embed_dim=input_time, num_heads=num_heads)

        self.proj = nn.Sequential(
            nn.Linear(input_time, hidden_dim),
            nn.GELU() if use_activation else nn.Identity(),
            nn.Linear(hidden_dim, 256)
        )

    def forward(self, h: Tensor, h_prime: Tensor) -> Tensor:
        """
        Args:
            h (Tensor): [N, T1]
            h_prime (Tensor): [N, T2]

        Returns:
            Tensor: [N, hidden_dim] — per-series representation
        """
        assert h.shape[0] == h_prime.shape[0], "Number of time series (N) must match."
        x = torch.cat([h, h_prime], dim=1)              # [N, T1 + T2]

        x = x.unsqueeze(1)                              # [N, 1, T]
        x = x.transpose(0, 1)                           # [1, N, T] → N is "sequence"

        x, _ = self.attn(x, x, x)                       # [1, N, T]
        x = x.squeeze(0)                                # [N, T]
        x = self.proj(x)                                # [N, 256]

        return x
