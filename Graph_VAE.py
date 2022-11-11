import torch; torch.manual_seed(0)
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions
import numpy as np
import torch_geometric
import torch_geometric.nn as tnn


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


class my_edge_conv(nn.Module):

    def __init__(self, input_dims, hidden_dims, output_dims):
        super(my_edge_conv, self).__init__()
        self.in_layer = nn.Linear(input_dims, hidden_dims)
        self.out_layer = nn.Linear(hidden_dims, output_dims)

    def forward(self, x):
        z = F.relu(self.in_layer(x))
        z = self.out_layer(z)

        return z


class my_gat_conv(nn.Module):

    def __init__(self, ins, hids, outs):
        super(my_gat_conv, self).__init__()
        self.gat_conv_1 = tnn.GATConv(ins, hids, heads=1)
        self.gat_conv_2 = tnn.GATConv(hids, outs)

    def forward(self, x, edge_index, edge_attr):
        x = F.dropout(x, p=0.4, training=self.training).cuda()
        x = self.gat_conv_1(x, edge_index, edge_attr).cuda()
        x = F.relu(x)
        x = F.dropout(x, p=0.4, training=self.training)
        x = self.gat_conv_2(x, edge_index, edge_attr)

        return x


def get_edge_feature(x, sparse_heat_matrix):
    residual_matrix = []
    for index_i, i in enumerate(x):
        box = torch.zeros(x.shape[1]).cuda()
        for index_j, j in enumerate(x):
            if sparse_heat_matrix[index_i][index_j] > 0:
                box = box.cuda() + sparse_heat_matrix[index_i][index_j] * (i - j).cuda()
            else:
                box = box.cuda()
        residual_matrix.append(box/x.shape[0])
    res_mat = torch.nn.utils.rnn.pad_sequence(residual_matrix, batch_first=True, padding_value=0)

    return torch.cat((res_mat, x), dim=1)



