import torch
import torch.nn as nn
import torch.nn.functional as F
from GGD import GeneralizedGraphDiffusion
from CATATTN import CatMultiAttn


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

        # Transition matrices and weights
        self.T = nn.Parameter(torch.empty(layers, expansion_step, num_nodes, num_nodes))
        self.theta = nn.Parameter(torch.empty(layers, expansion_step))

        # Graph diffusion layers
        self.diffusion_layers = nn.ModuleList([
            GeneralizedGraphDiffusion(diffusion_size[i], diffusion_size[i + 1], active[i])
            for i in range(len(diffusion_size) - 1)
        ])

        # Self-attention layers over concatenated feature matrices
        self.cat_attn_layers = nn.ModuleList([
            CatMultiAttn(
                input_time=embedding_size[i],        # e.g., input = concat[h, h_prime] dim
                num_heads=num_heads,
                hidden_dim=embedding_hidden_size,      
                output_dim=embedding_output_size,
                use_activation=active[i]             
            )
            for i in range(len(embedding_size))
        ])
        # Transform raw features to be divisible by num_heads
        self.raw_h_prime = nn.Linear(diffusion_size[0], raw_feature_size)
        # Final classifier
        self.linear = nn.Linear(embedding_output_size, classes)

        # Init transition weights
        self._init_transition_params()

    def _init_transition_params(self):
        nn.init.xavier_uniform_(self.T)
        nn.init.constant_(self.theta, 1.0 / self.theta.size(-1))  # normalize

    def forward(self, X: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        """
        Args:
            X: [N, F_in]  - node features
            A: [N, N]     - adjacency matrix
        Returns:
            logits: [N, classes]
        """
        h = X              # diffused features
        h_prime = X              # original features for attention fusion
        theta_soft = F.softmax(self.theta, dim=-1)  # normalize theta to summation of 1 per layer as the regularization

        for l in range(len(self.diffusion_layers) - 1):
            # Diffuse using learned linear combination of T_slices
            h = self.diffusion_layers[l](theta_soft[l], self.T[l], h, A)  # [N, diffusion_size]

            # Combine with prior representation using CatMultiAttn
            if l == 0:
                h_prime = self.cat_attn_layers[l](h, self.raw_h_prime(h_prime))  # [N, embedding_output_size]
            
            else:
                h_prime = h_prime + self.cat_attn_layers[l](h, h_prime)

        # Final projection to class logits
        out = self.linear(h_prime)  # [N, classes]
        
        return out


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
