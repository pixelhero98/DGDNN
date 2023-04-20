import torch
import torch.nn as nn
import torch.nn.functional as F


class joint_gemb(nn.Module):

    def __init__(self, ins, hid, out):
        super(joint_gemb, self).__init__()

        self.fc1 = nn.Linear(ins, hid)

        self.fc2 = nn.Linear(hid, out)

    def forward(self, x, gx):
        z = F.leaky_relu(self.fc1(torch.concat((x, gx), dim=1)))

        z = self.fc2(z)

        return z
