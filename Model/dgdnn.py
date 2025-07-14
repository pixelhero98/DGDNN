import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from ggd import GeneralizedGraphDiffusion
from catattn import CatMultiAttn


class DGDNN(nn.Module):
    def __init__(
        self,
        diffusion_size: list,     # e.g., [F0, F1, F2]
        embedding_size: list,     # e.g., [F1+F1, E1, E1+F2, E2, ...]
        embedding_hidden_size: int,
        embedding_output_size: int,
        raw_feature_size: int,
        classes: int,
        layers: int,
        num_nodes: int,
        expansion_step: int,      # number of diffusion basis per layer
        num_heads: int,
        active: list              # bool per layer
    ):
        super().__init__()
        assert len(diffusion_size) - 1 == layers, "diffusion_size length must equal layers + 1"
        assert len(embedding_size) == layers, "embedding_size length must equal layers"

        # Transition matrices and weights
        self.T = nn.Parameter(torch.empty(layers, expansion_step, num_nodes, num_nodes))
        self.theta = nn.Parameter(torch.empty(layers, expansion_step))

        # Graph diffusion layers
        self.diffusion_layers = nn.ModuleList([
            GeneralizedGraphDiffusion(
                input_dim=diffusion_size[i],
                output_dim=diffusion_size[i+1],
                active=active[i]
            ) for i in range(layers)
        ])

        # Self-attention layers over concatenated feature matrices
        self.cat_attn_layers = nn.ModuleList([
            CatMultiAttn(
                input_time=embedding_size[i],
                num_heads=num_heads,
                hidden_dim=embedding_hidden_size,
                output_dim=embedding_output_size,
                use_activation=active[i]
            ) for i in range(layers)
        ])

        # Transform raw features to match embedding_output_size
        self.raw_h_prime = nn.Linear(diffusion_size[0], raw_feature_size)

        # Final classifier
        self.linear = nn.Linear(embedding_output_size, classes)

        # Initialize transition parameters
        self._init_transition_params()

    def _init_transition_params(self):
        # Xavier init for transition matrices
        nn.init.xavier_uniform_(self.T)
        # Uniform init for theta so coefficients sum to one
        nn.init.constant_(self.theta, 1.0 / self.theta.size(-1))

    def forward(self, X: Tensor, A: Tensor) -> Tensor:
        """
        Args:
            X: [N, F0]  node features
            A: [N, N]   adjacency matrix
        Returns:
            logits: [N, classes]
        """
        N = X.size(0)
        theta_soft = F.softmax(self.theta, dim=-1)  # [layers, expansion_step]

        # Initial representations
        h = X.clone()
        h_prime = X.clone()

        for l in range(len(self.diffusion_layers)):
            # Diffusion step
            t_l = theta_soft[l]            # [expansion_step]
            T_l = self.T[l]               # [expansion_step, N, N]
            h = self.diffusion_layers[l](t_l, T_l, h, A)  # [N, F_{l+1}]

            # Attention fusion
            if l == 0:
                # project raw input features for first fusion
                h_prime = self.cat_attn_layers[l](h, self.raw_h_prime(X))  # [N, E]
            else:
                h_prime = h_prime + self.cat_attn_layers[l](h, h_prime)

        # Classification
        logits = self.linear(h_prime)
        return logits



## For those who use fast implementation version.

# class DGDNN(nn.Module):
#     def __init__(
#         self,
#         diffusion_size: list,
#         embedding_size: list,
#         embedding_hidden_size: int,
#         embedding_output_size: int,
#         raw_feature_size: int,
#         classes: int,
#         layers: int,
#         num_heads: int,
#         active: list
#     ):
#         super().__init__()
#         assert len(diffusion_size) - 1 == layers, "Mismatch in diffusion layers"
#         assert len(embedding_size) == layers, "Mismatch in attention layers"

#         self.layers = layers

#         self.diffusion_layers = nn.ModuleList([
#             GeneralizedGraphDiffusion(diffusion_size[i], diffusion_size[i + 1], active[i])
#             for i in range(layers)
#         ])

#         self.cat_attn_layers = nn.ModuleList([
#             CatMultiAttn(
#                 input_time=embedding_size[i],        # e.g., input = concat[h, h_prime] dim
#                 num_heads=num_heads,
#                 hidden_dim=embedding_hidden_size,      
#                 output_dim=embedding_output_size,
#                 use_activation=active[i]             
#             )
#             for i in range(len(embedding_size))
#         ])
#         # Transform raw features to be divisible by num_heads
#         self.raw_h = nn.Linear(diffusion_size[0], raw_feature_size)
        
#         self.linear = nn.Linear(embedding_output_size, classes)

#     def forward(self, X: torch.Tensor, A: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
#         """
#         Args:
#             X: [N, F_in]         - node features
#             A: [2, E]            - adjacency (sparse index)
#             W: [E]               - edge weights (if using sparse edge_index)

#         Returns:
#             logits: [N, classes]
#         """
#         z = X
#         h = X

#         for l in range(self.layers):
#             z = self.diffusion_layers[l](z, A, W)  # GeneralizedGraphDiffusion (e.g. GCNConv)
#             if l == 0:
#                 h = self.cat_attn_layers[l](z, self.raw_h(h))
#             else:
#                 h = h + self.cat_attn_layers[l](z, h)

#         return self.linear(h)  # [N, classes]
