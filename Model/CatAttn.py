import torch
import torch.nn as nn
import torch.nn.functional as F

class CatMultiAttn(nn.Module):
    def __init__(self, embed_dim, num_heads, output, active, timestamp):
        super().__init__()
        self.active = active
        self.attn     = nn.MultiheadAttention(embed_dim, num_heads)
        self.linear   = nn.Linear(embed_dim * timestamp, output * timestamp)
        self.norm     = nn.LayerNorm(output * timestamp)

    def forward(self, h: Tensor, h_prime: Tensor) -> Tensor:
        """
        h, h_prime: [B, timestamp, embedding]
        returns:    [B, output * timestamp]
        """
        B = h.shape[0]

        # 1) concat features
        x = torch.cat([h, h_prime], dim=-1)        # [B, timestamp, 2*embedding]
        # 2) Multi-Head Attn expects (seq, batch, embed)
        x = x.permute(1, 0, 2)                     # [timestamp, B, 2*embedding]
        x, _ = self.attn(x, x, x)
        # 3) back to batch-first and flatten
        x = x.permute(1, 0, 2).reshape(B, -1)      # [B, timestamp * 2*embedding]
        # 4) project, norm, act
        x = self.linear(x)                         # [B, output * timestamp]
        x = self.norm(x)
        if self.active:
            x = F.gelu(x)
        return x

