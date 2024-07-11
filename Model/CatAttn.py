import torch
import torch.nn as nn


class CatMultiAttn(torch.nn.Module):
    def __init__(self, embedding, num_heads, output, active, timestamp):
        super(CatMultiAttn, self).__init__()
        self.attn_layer = nn.MultiheadAttention(embedding, num_heads)
        self.linear_layer = nn.Linear(embedding*timestamp, output*timestamp)
        self.activation = nn.PReLU()
        self.active = active
        self.timestamp = timestamp

    def forward(self, h, h_prime):
        h = torch.cat((h, h_prime), dim=1).view(h.shape[0], self.timestamp, -1)
        h, _ = self.attn_layer(h, h, h)
        h = h.reshape(h.shape[0], -1)
        if self.active:
            h = self.activation(self.linear_layer(h))
        else:
          h = self.linear_layer(h)

        return h
