import torch
import torch.nn as nn


class CatMultiAttn(torch.nn.Module):
    def __init__(self, embedding, num_heads, output):
        super(CatMultiAttn, self).__init__()
        self.attn_layer = nn.MultiheadAttention(embedding, num_heads)
        self.linear_layer = nn.Linear(embedding, output)
        self.activation = nn.LeakyReLU()

    def forward(self, z, h):
        h = torch.cat((z, h), dim=1).unsqueeze(0)

        h, _ = self.attn_layer(h, h, h)

        h = self.activation(self.linear_layer(h.squeeze(0)))

        return h