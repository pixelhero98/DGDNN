import torch_geometric
import torch_geometric.nn as tnn
import torch.nn as nn
import torch.nn.functional as F
import torch
import math

## to generate another halved-feature 1026x1854  concate(sqrt(tnn.EdgeConv() * tnn.GATConv()), N_G)


def get_edge_feature(x, sparse_heat_matrix):
    residual_matrix = []
    for index_i, i in enumerate(x):
        box = torch.zeros(x.shape[1])
        for index_j, j in enumerate(x):
            if sparse_heat_matrix[index_i][index_j] > 0:
                box = box + sparse_heat_matrix[index_i][index_j] * (i - j)
            else:
                box = box
        residual_matrix.append(box / x.shape[0])

    return torch.cat((residual_matrix, x), dim=1)


class my_edge_conv(nn.Module):
    def __init__(self, input_dims, hidden_dims, output_dims):
        super(my_edge_conv, self).__init__()
        self.in_layer = nn.Linear(input_dims, hidden_dims)
        self.out_layer = nn.Linear(hidden_dims, output_dims)
        self.relu = F.relu()

    def forward(self, x):
        z = self.in_layer(x)
        z = self.relu(z)
        z = self.out_layer(z)

        return z

    def reset_parameters(self):
        self.in_layer.reset_parameters()
        self.out_layer.reset_parameters()


class my_gat_conv(nn.Module):
    def __init__(self, ins, hids, outs):
        super(my_gat_conv, self).__init__()
        self.gat_conv_1 = tnn.GATConv(ins, hids, heads=1)
        self.gat_conv_2 = tnn.GATConv(hids, outs)
        self.relu = F.relu()
        self.dropout = F.dropout(p=0.4)

    def forward(self, x, edge_index, edge_attr):
        x = self.dropout(x, p=0.4, training=self.training)
        x = self.gat_conv_1(x, edge_index, edge_attr)
        x = self.relu(x)
        x = self.dropout(x, p=0.4, training=self.training)
        x = self.gat_conv_2(x, edge_index, edge_attr)

        return x

    def reset_parameters(self):
        self.gat_conv_1.reset_parameters()
        self.gat_conv_1.reset_parameters()