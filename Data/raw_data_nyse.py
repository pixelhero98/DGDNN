from scipy.linalg import expm
import torch
import csv
import os
import numpy as np
import random
from path_config import NYSE_path, NYSE_target_comps_path



def check_years(line, years):

    for year in years:
        if line == year:
            return 1


def check_label(x):

    if x[0] == 0 or x[-2] == 0:

        count0 = 0
        count1 = 0

        if x[0] == 0:
            while x[0 + 5 * count0] == 0:
                count0 += 1

        if x[-2] == 0:
            while x[-2 - 5 * count1] == 0:
                count1 += 1

        return (x[-2 - 5 * count1] / x[0 + 5 * count0]) - 1

    else:
        return (x[-2] / x[0]) - 1


def node_feature_label_generation(X1, start, end):

    data_path = NYSE_path()
    target_comps_path_0 = NYSE_target_comps_path()
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
        x_lab = []
        for items in os.listdir(data_path):
            d_path = data_path + '/' + items
            if 'NYSE_' + h + '_30Y' == items[:-4]:

                with open(d_path) as f:
                    file = csv.reader(f)
                    for line in file:
                        if check_years(line[0][:4], years):
                            for ele in line[1:]:
                                x_lab.append(float(ele))

                for pp in range(5):
                    with open(d_path) as f:
                        file = csv.reader(f)
                        for line in file:
                            if check_years(line[0][:4], years):
                                if pp == 0:
                                    x.append(float(line[1]))
                                elif pp == 1:
                                    x.append(float(line[2]))
                                elif pp == 2:
                                    x.append(float(line[3]))
                                elif pp == 3:
                                    x.append(float(line[4]))
                                else:
                                    x.append(float(line[5]) * 1e-6)

                X1.append(torch.Tensor(x))
                labels.append(check_label(torch.Tensor(x_lab)))
                break

    X1 = torch.nn.utils.rnn.pad_sequence(X1, batch_first=True, padding_value=0)
    label_new = []

    q = torch.tensor([0.1, 0.2, 0.3, 0.6, 0.8])
    y = torch.quantile(torch.Tensor(labels), q, dim=0, keepdim=True)

    for i in labels:
        for index, j in enumerate(y):
            if i <= j:
                label_new.append(index)
                break
            elif i > j and j == y[-1]:
                label_new.append(index + 1)
                break

    return X1, torch.Tensor(label_new)


def edge_info_generation(X1, method0, method1):

    Edge = []
    edge_index_0 = []
    edge_index_1 = []
    edge_attr = []
    R = discrete_conv(X1)
    R = torch.Tensor.numpy(R)
    D_T = np.diag(1 / np.sqrt(R.sum(axis=1)))
    R = D_T @ R @ D_T

    if method0 == 'heat':
        R = expm(-5 * (np.eye(X1.shape[0]) - R))
    elif method0 == 'ppr':
        0.1 * np.linalg.inv(np.eye(X1.shape[0]) - (1 - 0.1) * R)
    else:
        R = expm(-5 * (np.eye(X1.shape[0]) - R))

    if method1 == 'threshold':
        R[R < 0.003] = 0
    elif method1 == 'top_k':
        row_idx = np.arange(X1.shape[0])
        R[R.argsort(axis=0)[:X1.shape[0] - 128], row_idx] = 0
    else:
        R[R < 0.003] = 0

    norm = R.sum(axis=0)
    norm[norm <= 0] = 1
    R = R / norm
    R = torch.Tensor(R)

    for index_i, i in enumerate(R):
        for index_j, j in enumerate(i):
            if j > 0:
                edge_index_0.append(index_i)
                edge_index_1.append(index_j)
                edge_attr.append(j)

    Edge = [torch.Tensor(edge_index_0), torch.Tensor(edge_index_1)]
    Edge = torch.nn.utils.rnn.pad_sequence(Edge, batch_first=True, padding_value=0)
    edge_attr = torch.Tensor(edge_attr)

    return Edge, edge_attr.reshape(edge_attr.shape[0], 1)


def discrete_conv(X1):
    dc = []
    for index, i in enumerate(X1):
        r = []
        for j in X1:
            box = sum(np.convolve(i, j, mode='full')) / sum(np.convolve(i, i, mode='full'))
            if box >= 1 / 1.1 and box <= 1.1:
                r.append(box)
            else:
                r.append(0)

        dc.append(torch.Tensor(r))

    return torch.nn.utils.rnn.pad_sequence(dc, batch_first=True, padding_value=0)


def mask_generation(X1):
    train_mask = []
    val_mask = []
    test_mask = []
    train = random.sample(range(0, 408), 408)
    val = random.sample(range(408 + 309, X1.shape[0]), 309)
    test = random.sample(range(408, 408 + 309), 309)

    for i in range(X1.shape[0]):
        train_mask.append(False)
        val_mask.append(False)
        test_mask.append(False)

    for i in train:
        train_mask[i] = True

    for i in val:
        val_mask[i] = True

    for i in test:
        test_mask[i] = True

    return torch.Tensor(train_mask).type(torch.bool), torch.Tensor(val_mask).type(torch.bool), torch.Tensor(
        test_mask).type(torch.bool)


def gx_generation(gx, traindata):

    box = torch.Tensor(np.zeros(traindata.x.shape[1]))

    for index_i, i in enumerate(traindata.edge_index[0]):

        if index_i == 0:
            box = box + (traindata.x[i] - traindata.x[traindata.edge_index[1][index_i]]) * traindata.edge_attr[index_i]

        elif traindata.edge_index[0][index_i - 1] != i:
            gx.append(box / traindata.x.shape[0])
            box = torch.Tensor(np.zeros(traindata.x.shape[1]))
            box = box + (traindata.x[i] - traindata.x[traindata.edge_index[1][index_i]]) * traindata.edge_attr[index_i]

        elif index_i == traindata.edge_index[0].shape[0] - 1:
            gx.append(box)

    gx = torch.nn.utils.rnn.pad_sequence(gx, batch_first=True, padding_value=0)

    return gx
