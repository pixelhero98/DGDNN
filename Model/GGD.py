import torch
import torch.nn as nn

class GeneralizedGraphDiffusion(torch.nn.Module):

    def __init__(self, input_dim, output_dim):
        super(GeneralizedGraphDiffusion, self).__init__()
        self.output = output_dim
        self.fc_layer = nn.Linear(input_dim, output_dim)
        self.activation0 = torch.nn.PReLU()

    def forward(self, theta, t, x, a):

        q = torch.zeros_like(a)
        h = x

        for i in range(theta.shape[0]):
            q += theta[i] * t[i]

        h = self.fc_layer((q * a) @ h)
        h = self.activation0(h)

        return h