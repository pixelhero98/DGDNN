import torch; torch.manual_seed(0)
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions
import numpy as np


class VariationalEncoder(nn.Module):
    def __init__(self, input_dims, hidden_dims, latent_dims):
        super(VariationalEncoder, self).__init__()
        self.linear1 = nn.Linear(input_dims, hidden_dims)
        self.linear2 = nn.Linear(hidden_dims, latent_dims)
        self.linear3 = nn.Linear(hidden_dims, latent_dims)

        self.N = torch.distributions.Normal(0, 1)
        self.N.loc = self.N.loc.cuda() # hack to get sampling on the GPU
        self.N.scale = self.N.scale.cuda()
        self.kl = 0

    def forward(self, x):
        x = F.relu(self.linear1(x))
        mu =  self.linear2(x).cuda()
        sigma = torch.exp(self.linear3(x)).cuda()
        z = mu + sigma*self.N.sample(mu.shape)
        self.kl = (sigma**2 + mu**2 - torch.log(sigma) - 1/2).sum()
        return z




class Decoder(nn.Module):
    def __init__(self, latent_dims, hidden_dims, output_dims):
        super(Decoder, self).__init__()
        self.linear1 = nn.Linear(latent_dims, hidden_dims).cuda()
        self.linear2 = nn.Linear(hidden_dims, output_dims).cuda()

    def forward(self, x):
        z = F.relu(self.linear1(x)).cuda()
        z = self.linear2(z).cuda()
        return z




class Vae(nn.Module):
    def __init__(self, input_dims, hidden_dims0, latent_dims, hidden_dims1, output_dims):
        super(Vae, self).__init__()
        self.encoder = VariationalEncoder(input_dims, hidden_dims0, latent_dims)
        self.decoder = Decoder(latent_dims, hidden_dims1, output_dims)

    def forward(self, x):
        z = self.encoder(x)
        z = self.decoder(z)
        return z
    def reset_parameters(self):
        self.encoder.reset_parameters()
        self.decoder.reset_parameters()