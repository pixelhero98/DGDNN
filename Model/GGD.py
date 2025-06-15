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
