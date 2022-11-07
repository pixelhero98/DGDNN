import torch
import torch.nn as nn
import math
from Graph_VAE import Vae
from My_edge_gat_conv import my_gat_conv, my_edge_conv


class AU_Net(nn.Module):
    def __init__(self, vae_ins, vae_hids0, vae_lats, vae_hids1, vae_outs,
                 edgeconv_ins, edgeconv_hids, edgeconv_outs, gat_ins, gat_hids, gat_outs):

        super(AU_Net, self).__init__()
        self.vae_layer = Vae(vae_ins, vae_hids0, vae_lats, vae_hids1, vae_outs)
        self.edgeconv_layer = my_edge_conv(edgeconv_ins, edgeconv_hids, edgeconv_outs)
        self.gatconv_layer = my_gat_conv(gat_ins, gat_hids, gat_outs)


    def forward(self, x, edge_index, edge_attr, label, concat_x):

        z0 = self.vae_layer(x)
        z11 = self.edgeconv_layer(concat_x)
        z12 = self.gatconv_layer(x, edge_index, edge_attr)
        z1 = math.sqrt(z11 * z12)
        z = torch.cat((z0, z1), dim=1)

        return z

    def reset_parameters(self):

        self.vae_layer.reset_parameters()
        self.edgeconv_layer.reset_parameters()
        self.gatconv_layer.reset_parameters()


