import torch
import torch.nn as nn
from torch import Tensor

class GeneralizedGraphDiffusion(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, active: bool):
        super().__init__()
        # Linear projection from input features to output features
        self.fc = nn.Linear(input_dim, output_dim)
        # Pre-activation: PReLU with one parameter per channel if active
        self.activation = nn.PReLU(num_parameters=input_dim) if active else nn.Identity()

    def forward(
        self,
        theta: Tensor,        # [S] diffusion coefficients
        T_slices: Tensor,     # [S, N, N] diffusion bases
        x: Tensor,            # [N, F_in] node features
        a: Tensor             # [N, N] adjacency weights
    ) -> Tensor:             # [N, F_out]
        # Combine diffusion bases via learned coefficients
        q = torch.einsum('s,sij->ij', theta, T_slices)  # [N, N]
        # Mask by adjacency
        q = q * a                                      # [N, N]

        # Convert to sparse and perform sparse-dense multiplication
        q_sparse = q.to_sparse().coalesce()
        out = torch.sparse.mm(q_sparse, x)            # [N, F_in]

        # Apply activation and final linear layer
        out = self.activation(out)
        out = self.fc(out)                            # [N, F_out]
        return out

# Fast approximation version using GCNConv for efficiency
# from torch_geometric.nn import GCNConv
# import torch.nn as nn
# import torch

# class GeneralizedGraphDiffusion(nn.Module):
#     def __init__(self, input_dim, output_dim, active: bool):
#         super().__init__()
#         self.gconv = GCNConv(input_dim, output_dim)
#         self.activation = nn.PReLU(num_parameters=output_dim) if active else nn.Identity()

#     def forward(
#         self,
#         x: torch.Tensor,                  # [N, F_in]
#         edge_index: torch.Tensor,        # [2, E] COO format
#         edge_weight: torch.Tensor        # [E] edge weights (like q_ij values)
#     ) -> torch.Tensor:
#         x = self.gconv(x, edge_index, edge_weight)  # [N, F_out]
#         x = self.activation(x)
#         return x

