import torch
import torch.nn as nn
from torch import Tensor


class CatMultiAttn(nn.Module):
    def __init__(
        self,
        input_time: int,       # T1 + T2
        num_heads: int,
        hidden_dim: int,
        output_dim: int,
        use_activation: bool
    ):
        """
        Args:
            input_time (int): Combined time dimension after concatenation (T1 + T2)
            num_heads (int): Number of attention heads
            hidden_dim (int): Hidden output dimension (E_h)
            output_dim (int): Final projection dimension
            use_activation (bool): Whether to apply GELU activation
        """
        super().__init__()
        self.use_activation = use_activation

        # Multi-head attention: treating each series as a token
        # Sequence length L = number of series (N), embed_dim = input_time
        self.attn = nn.MultiheadAttention(embed_dim=input_time, num_heads=num_heads)
        self.norm = nn.LayerNorm(input_time)

        # Projection network
        self.proj = nn.Sequential(
            nn.Linear(input_time, hidden_dim),
            nn.GELU() if use_activation else nn.Identity(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, h: Tensor, h_prime: Tensor) -> Tensor:
        """
        Args:
            h (Tensor): shape [N, T1]  — features for each of N series over T1 time
            h_prime (Tensor): shape [N, T2]  — additional features over T2 time

        Returns:
            Tensor: shape [N, output_dim] — per-series representation after attention + projection
        """
        # Ensure same batch of series
        assert h.shape[0] == h_prime.shape[0], "Number of series (N) must match."

        # Concatenate along time dimension => [N, T1+T2]
        x = torch.cat([h, h_prime], dim=1)

        # Treat each series as a token: reshape to (L=N, batch=1, E=input_time)
        x = x.unsqueeze(1)  # [N, 1, input_time]

        # Multi-head attention across series dimension
        attn_out, _ = self.attn(x, x, x)  # [L=N, 1, E]
        attn_out = self.norm(attn_out)    # LayerNorm over last dim

        # Remove batch dimension => [N, E]
        x = attn_out.squeeze(1)

        # Project to output dimension
        x = self.proj(x)  # [N, output_dim]
        return x
