import torch
import torch.nn as nn
import torch_geometric.nn as tnn
import torch.nn.functional as F


class AU_Net(nn.Module):

  def __init__(self, ins, hids0, hids1, hids2, outs, num_labels):

    super(AU_Net, self).__init__()

    self.edgeconv1 = nn.Linear(ins, hids0)
    self.dimreduc = nn.Linear(hids0, hids2)
    self.gcnconv1 = tnn.GCNConv(hids0, hids1)
    self.gcnconv2 = tnn.GCNConv(hids1, hids2)
    self.edgeconv2 = nn.Linear(hids2 + hids1 + hids0, hids2)
    self.edgeconv3 = nn.Linear(hids2, outs)
    self.out = nn.Linear(outs, num_labels)


  def forward(self, x, edge_index, gx):

    z = F.relu(self.edgeconv1(torch.concat((x, gx), dim=1)))
    z0 = self.dimreduc(z)
    z1 = F.relu(self.gcnconv1(z + gx, edge_index))
    z2 = F.relu(self.gcnconv2(z1, edge_index))
    z = F.relu(self.edgeconv2(torch.concat((z, z1, z2), dim=1)))
    z = F.relu(self.edgeconv3(z + z0))
    z = self.out(z)

    return z

  def reset_parameters(self):

    self.edgeconv1.reset_parameters()
    self.edgeonv2.reset_parameters()
    self.gcnconv1.reset_parameters()
    self.gcnconv2.reset_parameters()
    self.edgeconv3.reset_parameters()
    self.dimreduc.reset_parameters()




