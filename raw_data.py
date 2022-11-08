import torch
import csv
import os
import numpy as np
import torch.nn as nn
import torch.nn.functional as f
import random
from Cross_Attention_Score import cross_attention_score
from path_config import path, target_comps_path


def check_years(line, years):
    for year in years:
        if line == year:
            return 1


def check_label(x):
    if x[0] == 0:
        return x[-2] / x[5] - 1
    else:
        return x[-2] / x[0] - 1


def label_generation(labels):
    label = []
    for i in labels:
        if i >= -0.01 and i <= 0.01:
            label.append(list([0,0,0,0]))
        elif i >= -0.15 and i < -0.01:
            label.append(list([0,0,0,1]))
        elif i >= -0.29 and i < -0.15:
            label.append(list([0,0,1,0]))
        elif i >= -0.43 and i < -0.29:
            label.append(list([0,0,1,1]))
        elif i >= -0.57 and i < -0.43:
            label.append(list([0,1,0,0]))
        elif i >= -0.71 and i < -0.57:
            label.append(list([0,1,0,1]))
        elif i >= -0.85 and i < -0.71:
            label.append(list([0,1,1,0]))
        elif i < -0.85:
            label.append(list([0,1,1,1]))
        elif i > 0.01 and i <= 0.15:
            label.append(list([1,0,0,0]))
        elif i > 0.15 and i <= 0.29:
            label.append(list([1,0,0,1]))
        elif i > 0.29 and i <= 0.43:
            label.append(list([1,0,1,0]))
        elif i > 0.43 and i <= 0.57:
            label.append(list([1,0,1,1]))
        elif i > 0.57 and i <= 0.71:
            label.append(list([1,1,0,0]))
        elif i > 0.71 and i <= 0.85:
            label.append(list([1,1,0,1]))
        elif i > 0.85 and i <= 0.99:
            label.append(list([1,1,1,0]))
        else:
            label.append(list([1,1,1,1]))

    return torch.Tensor(label)


def node_feature_label_generation(X1, start, end):
    data_path = path()
    target_comps_path_0 = target_comps_path()
    comlist = []
    years = []
    labels = []

    for i in range(int(start), int(end) + 1):
        years.append(str(i))

    with open(target_comps_path_0) as k:
        ks = csv.reader(k)
        for line in ks:
            comlist.append(line[0])

    for h in comlist:
        x = []
        for items in os.listdir(data_path):
            d_path = data_path + '/' + items
            if 'NASDAQ_' + h + '_30Y' == items[:-4]:
                with open(d_path) as f:
                    file = csv.reader(f)
                    for line in file:
                        if check_years(line[0][:4], years):
                            for ele in line[1:]:
                                if ele == line[-1]:
                                    x.append(float(ele) * 0.00001)
                                else:
                                    x.append(float(ele))

                X1.append(torch.Tensor(x))
                labels.append(check_label(torch.Tensor(x)))
                break

    X1 = torch.nn.utils.rnn.pad_sequence(X1, batch_first=True, padding_value=0)
    labels_one_hot = label_generation(labels)

    return X1, torch.Tensor(labels_one_hot)


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


def train_mask_generation():
