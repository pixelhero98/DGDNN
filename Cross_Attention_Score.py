import torch.nn as nn
import math
import torch


class cross_attention_score(nn.Module):

    def __init__(self, input_dim, dim_k, dim_v):
        super(cross_attention_score, self).__init__()
        self.dim_q = dim_k
        self.dim_k = dim_k
        self.dim_v = dim_v

        self.linear_q = nn.Linear(input_dim, dim_k, bias=False)
        self.linear_k = nn.Linear(input_dim, dim_k, bias=False)
        self.linear_v = nn.Linear(input_dim, dim_v, bias=False)
        self._norm_fact = 1 / math.sqrt(dim_k)

    def forward(self, x, x1):
        q = self.linear_q(x)
        k = self.linear_k(x1)
        v = self.linear_v(x)

        dist = (q * k) * v * self._norm_fact
        dist = torch.nn.functional.softmax(dist, dim=0)

        return dist