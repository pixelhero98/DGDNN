import torch.nn as nn
import torch

class GeneralizedGraphDiffusion(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, active: bool):
        super().__init__()
        self.fc = nn.Linear(input_dim, output_dim)
        self.activation = nn.PReLU(num_parameters=input_dim) if active else nn.Identity()

    def forward(
        self,
        theta: Tensor,        # [S]
        T_slices: Tensor,     # [S, N, N]
        x: Tensor,            # [N, F_in]
        a: Tensor             # [N, N]
    ) -> Tensor:              # [N, F_out]

        q = torch.einsum('s,sij->ij', theta, T_slices)  # [N, N]
        q = q * a                                       # [N, N]

        q = q.to_sparse()
        out = torch.sparse.mm(q, x)                    # [N, F_in]

        out = self.activation(out)
        out = self.fc(out)                             # [N, F_out]
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

