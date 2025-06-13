import torch
import torch.nn as nn

class GeneralizedGraphDiffusion(nn.Module):
    def __init__(self, input_dim, output_dim, active):
        super().__init__()
        self.fc         = nn.Linear(input_dim, output_dim)
        self.activation = nn.PReLU(num_parameters=output_dim)
        self.active     = active

    def forward(self,
                theta: torch.Tensor,    # [S]
                T_slices: torch.Tensor, # [S, N, N]
                x: torch.Tensor,        # [N, F_in]
                a: torch.Tensor         # [N, N]
               ) -> torch.Tensor:       # [N, F_out]

        # 1) build the weighted diffusion matrix q (N×N)
        #    q[i,j] = sum_s theta[s] * T_slices[s, i, j]
        q = torch.einsum('s,sij->ij', theta, T_slices)

        # 2) elementwise combine with adjacency, then propagate:
        #    (q * a) is N×N, multiply into x: (N×N) @ (N×F_in) → N×F_in
        out = (q * a) @ x
        
        # 3) optional nonlinearity
        if self.active:
            out = self.activation(out)
            
        # 4) project to output features
        out = self.fc(out)
                   
        return out

# you can use the below for fast implementation if computational resources are limited.
#class GeneralizedGraphDiffusion(torch.nn.Module):
    #def __init__(self, input_dim, output_dim, active):
        #super(GeneralizedGraphDiffusion, self).__init__()
        #self.output = output_dim
        #self.gconv = GCNConv(input_dim, output_dim)
        #self.activation0 = torch.nn.PReLU()
        #self.active = active

    #def forward(self, x, a, w):
        #x = self.gconv(x, a, w)
        #if self.active:
            #x = self.activation0(x)

        #return x
