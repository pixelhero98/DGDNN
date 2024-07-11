import torch
import torch.nn as nn
import torch.nn.functional as F
from GGD import GeneralizedGraphDiffusion
from CatAttn import CatMultiAttn

class DGDNN(nn.Module):
    def __init__(self, diffusion_size, embedding_size, classes, layers, num_nodes, expansion_step, num_heads, active, timestamp):
        super(DGDNN, self).__init__()

        # Initialize transition matrices and weight coefficients for all layers
        self.T = nn.Parameter(torch.randn(layers, expansion_step, num_nodes, num_nodes))
        self.theta = nn.Parameter(torch.randn(layers, expansion_step))

        # Initialize different module layers at all levels
        self.diffusion_layers = nn.ModuleList(
            [GeneralizedGraphDiffusion(diffusion_size[i], diffusion_size[i + 1], active[i]) for i in range(len(diffusion_size) - 1)])
        self.cat_attn_layers = nn.ModuleList(
            [CatMultiAttn(embedding_size[2 * i], num_heads, embedding_size[2 * i + 1], active[i], timestamp) for i in range(len(embedding_size) // 2)])
        self.linear = nn.Linear(embedding_size[-1]*timestamp, classes)

    def forward(self, X, A):
        z, h = X, X

        for l in range(self.T.shape[0]):
            z = self.diffusion_layers[l](self.theta[l], self.T[l], z, A)
            h = self.cat_attn_layers[l](z, h)

        h = self.linear(h)
        return h
