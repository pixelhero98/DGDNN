import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from Graph_VAE import Vae, my_gat_conv, my_edge_conv


class AU_Net(nn.Module):
    def __init__(self, vae_ins, vae_hids0, vae_lats, vae_hids1, vae_outs,
                 edgeconv_ins, edgeconv_hids, edgeconv_outs, gat_ins, gat_hids, gat_outs,
                 num_labels):

        super(AU_Net, self).__init__()
        self.vae_layer = Vae(vae_ins, vae_hids0, vae_lats, vae_hids1, vae_outs)
        self.edgeconv_layer = my_edge_conv(edgeconv_ins, edgeconv_hids, edgeconv_outs)
        self.gatconv_layer = my_gat_conv(gat_ins, gat_hids, gat_outs)
        self.get_classes_layer = nn.Linear(vae_ins, num_labels)


    def forward(self, x, edge_index, edge_attr, concat_x):

        z0 = self.vae_layer(x)
        z11 = F.relu(self.edgeconv_layer(concat_x))
        z12 = F.relu(self.gatconv_layer(x, edge_index, edge_attr))
        z1 = torch.sqrt(z11 * z12).cuda()
        zx = torch.cat((z0, z1), dim=1).cuda()
        z = self.get_classes_layer(zx).cuda()


        return z


