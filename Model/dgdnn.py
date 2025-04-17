import torch
import torch.nn as nn
import torch.nn.functional as F
from GGD import GeneralizedGraphDiffusion
from CatAttn import CatMultiAttn

class DGDNN(nn.Module):
    def __init__(self, diffusion_size, embedding_size, classes,
                 layers, num_nodes, expansion_step, num_heads, active, timestamp):
        super().__init__()

        # Allocate transition params
        self.T     = nn.Parameter(torch.empty(layers,
                                              expansion_step,
                                              num_nodes,
                                              num_nodes))
        self.theta = nn.Parameter(torch.empty(layers, expansion_step))

        # Initialize other modules
        self.diffusion_layers = nn.ModuleList(
            [GeneralizedGraphDiffusion(diffusion_size[i],
                                       diffusion_size[i + 1],
                                       active[i])
             for i in range(len(diffusion_size) - 1)]
        )
        self.cat_attn_layers = nn.ModuleList(
            [CatMultiAttn(embedding_size[2 * i],
                          num_heads,
                          embedding_size[2 * i + 1],
                          active[i],
                          timestamp)
             for i in range(len(embedding_size) // 2)]
        )
        self.linear = nn.Linear(embedding_size[-1] * timestamp, classes)

        # Perform smart initialization
        self._init_transition_params()

    def _init_transition_params(self):
        # Xavier init for transition matrices
        nn.init.xavier_uniform_(self.T)
        # Uniform or constant for theta
        nn.init.constant_(self.theta, 1.0 / self.theta.size(-1))

    def forward(self, X, A):
        z, h = X, X
        # Optionally constrain theta to be non-negative or sum-to-one:
        theta_pos  = F.softplus(self.theta)             # >= 0
        theta_prob = F.softmax(self.theta, dim=-1)      # sum-to-1

        for l in range(self.T.shape[0]):
            # pick whichever makes sense for your diffusion layer:
            z = self.diffusion_layers[l](theta_pos[l], self.T[l], z, A)
            h = self.cat_attn_layers[l](z, h)

        return self.linear(h)

# For those who use fast implementation version.
#class DGDNN(nn.Module):
    #def __init__(self, diffusion_size, embedding_size, classes, num_heads, active, timestamp, layers):
        #super(DGDNN, self).__init__()

        # Initialize different module layers at all levels
        #self.diffusion_layers = nn.ModuleList(
            #[GeneralizedGraphDiffusion(diffusion_size[i], diffusion_size[i + 1], active[i]) for i in range(len(diffusion_size) - 1)])
        #self.cat_attn_layers = nn.ModuleList(
            #[CatMultiAttn(embedding_size[2 * i], num_heads, embedding_size[2 * i + 1], active[i], timestamp) for i in range(len(embedding_size) // 2)])
        #self.linear = nn.Linear(embedding_size[-1]*timestamp, classes)

    #def forward(self, X, A, W):
        #z, h = X, X

        #for l in range(layers):
            #z = self.diffusion_layers[l](z, A, W)
            #h = self.cat_attn_layers[l](z, h)

        #h = self.linear(h)

        #return h
