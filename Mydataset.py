import torch
import csv
import os
from torch_geometric.data import Data
from torch_geometric.data import InMemoryDataset, download_url
import numpy as np
import torch.nn as nn
import math
import torch.nn.functional as f
from torch import Tensor
from typing import Optional, Tuple
import random
from Cross_Attention_Score import cross_attention_score
from path_config import path, target_comps_path



def node_feature_generation(X1):
    path0 = path()
    target_path = target_comps_path()
    comlist = []
    with open(target_comps_path) as k:
        ks = csv.reader(k)
        for line in ks:
            comlist.append(line[0])

    for h in comlist:
        x = []
        for items in os.listdir(path0):
            d_path = path0 + '/' + items
            if 'NASDAQ_' + h + '_30Y' == items[:-4]:
                with open(d_path) as f:
                    file = csv.reader(f)
                    for line in file:
                        if '2013' == line[0][:4] or '2014' == line[0][:4] or '2015' == line[0][:4]:
                            for ele in line[1:]:
                                if ele == line[-1]:
                                    x.append(float(ele) * 0.00001)
                                else:
                                    x.append(float(ele))

                X1.append(torch.Tensor(x))
                break

    X1 = torch.nn.utils.rnn.pad_sequence(X1, batch_first=True, padding_value=0)

    return X1


def edge_info_generation(X1):
    edge_index_0 = []
    edge_index_1 = []
    edge_attr_0 = []
    for com_index, com in enumerate(X1):
        sum = 0
        for com_name_index, com_name in enumerate(X1):
            if com_name_index > com_index:
                s_atten = cross_attention_score(X1.shape[1], 64, 64)
                att_op_weight = s_atten(X1[com_name_index], com)
                for j in att_op_weight:
                    sum = sum + j * j
                if sum > 0.99 and rand_seed_generation(sum) > 0.33:
                    edge_index_0.append(int(com_index))
                    edge_index_1.append(int(com_name_index))
                    edge_index_0.append(int(com_name_index))
                    edge_index_1.append(int(com_index))
                    edge_attr_0.append(int(sum))

    Edge_index = []
    Edge_index.append(torch.Tensor(edge_index_0))
    Edge_index.append(torch.Tensor(edge_index_1))
    Edge_index = torch.nn.utils.rnn.pad_sequence(Edge_index, batch_first=True, padding_value=0)

    return Edge_index, torch.Tensor(edge_attr_0)


def rand_seed_generation(sum):
    if sum > 0.99:
        np.random.seed(random.sample(range(0, 9000), 1))
        p0 = np.random.rand(1)
    return p0


class MyOwnDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['some_file_1', 'some_file_2', ...]

    @property
    def processed_file_names(self):
        return ['data.pt']

    def process(self):

        X1 = node_feature_generation(X1=[])

        Edge_index, Edge_attr = edge_info_generation(X1, path, )

        Edge_attr = Edge_attr.reshape(Edge_attr.shape[0], 1)

        # Y = torch.tensor([0,1,0],dtype=torch.float)

        data = Data(x=X1.type(torch.float), edge_index=Edge_index.type(torch.long), edge_attr=Edge_attr)

        data_list = [data]

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

