import torch
import torch.nn as nn

class GeneralizedGraphDiffusion(torch.nn.Module):
    def __init__(self, input_dim, output_dim, active):
        super(GeneralizedGraphDiffusion, self).__init__()
        self.output = output_dim
        self.fc = nn.Linear(input_dim, output_dim)
        self.activation0 = torch.nn.PReLU()
        self.active = active

    def forward(self, theta, t, x, a):
        q = 0
        for i in range(theta.shape[0]):
            q += theta[i] * t[i]
        
        x = self.fc((q * a) @ x)
        if self.active:
            x = self.activation0(x)

        return x

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
