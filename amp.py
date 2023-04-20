import torch
import torch.nn.functional as F
from raw_data import gx_generation, node_feature_label_generation
from path_config import dir_path
from Mydataset import TrDataset_0, TrDataset_1, TrDataset_2
from AU_GaphNet import AU_Net
from torch_geometric.logging import log
import torch.distributions
import numpy as np
import matplotlib.pyplot as plt
import random
from matplotlib import cm
from matplotlib import axes
import seaborn as sns
import sklearn.preprocessing as skp

# configure the device for running the model on GPU or CPU

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# configure the default parameters

for ii in ['TrDataset_0', 'TrDataset_1', 'TrDataset_2']:

    print('generating Inmemory Dataset, please wait...')

    for jj in ['gx', 'weighted']:

        for kk in ['0.1', '0.2', '0.3', '0.5', '0.7', '0.9', '1.0']:
            pp_list = []
            for vv in range(10):
                results_list = []

                # load training dataset

                if ii == 'TrDataset_0':
                    traindata = TrDataset_0(root=dir_path() + 'train').data
                elif ii == 'TrDataset_1':
                    traindata = TrDataset_1(root=dir_path() + 'val').data
                else:
                    traindata = TrDataset_2(root=dir_path() + 'test').data


                # generate gx term
                def get_adj_matrix(data):
                    num_nodes = data.x.shape[0]
                    adj_matrix = np.zeros(shape=(num_nodes, num_nodes))
                    for i, j in zip(data.edge_index[0], data.edge_index[1]):
                        adj_matrix[i, j] = 1.

                    return adj_matrix


                if jj == 'gx':
                    gx = []
                    gx = gx_generation(gx, traindata)
                else:
                    gx = []
                    gx = torch.Tensor(get_adj_matrix(traindata))
                    key = 0
                    for index_j, j in enumerate(gx):
                        for index_k, k in enumerate(j):
                            if k != 0:
                                gx[index_j][index_k] = traindata.edge_attr[key]
                                key += 1

                # gx term to device
                gx = gx.to(device)


                # randomly generating mask
                def mask_generation(X1):

                    train_mask = []
                    val_mask = []
                    test_mask = []
                    train = random.sample(range(0, 408), int(408 * float(kk)))
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

                    return torch.Tensor(train_mask).type(torch.bool), torch.Tensor(val_mask).type(
                        torch.bool), torch.Tensor(test_mask).type(torch.bool)


                traindata.train_mask, traindata.val_mask, traindata.test_mask = mask_generation(traindata.x)

                # init model
                if jj == 'gx':
                    model = AU_Net(ins=traindata.x.shape[1] + traindata.x.shape[1], hids0=traindata.x.shape[1],
                                   hids1=1024, hids2=512,
                                   outs=256, num_labels=6, adj_dim=traindata.x.shape[0])
                else:
                    model = AU_Net(ins=traindata.x.shape[1] + traindata.x.shape[0], hids0=traindata.x.shape[0],
                                   hids1=1024, hids2=512,
                                   outs=256, num_labels=6, adj_dim=traindata.x.shape[0])

                # model and dataset to GPU
                model, traindata = model.to(device), traindata.to(device)

                # define optimizer
                optimizer = torch.optim.Adam(params=model.parameters(), lr=2e-4)
                scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.997)


                # define training process & testing process

                def train():
                    model.train()
                    optimizer.zero_grad()
                    out = model(traindata.x, traindata.edge_index, gx)
                    loss = F.cross_entropy(out[traindata.train_mask], traindata.y[traindata.train_mask])
                    loss.backward()
                    optimizer.step()

                    return float(loss.item())


                @torch.no_grad()
                def test():
                    model.eval()
                    out = model(traindata.x, traindata.edge_index, gx)
                    pred = out.argmax(dim=1)
                    accs = []
                    for mask in [traindata.train_mask, traindata.val_mask, traindata.test_mask]:
                        accs.append(int((pred[mask] == traindata.y[mask]).sum()) / int(mask.sum()))

                    return accs


                # record accuracy change with respect to training process

                btacc = 0
                bvalacc = 0
                btestacc = 0
                epoch_l = []
                loss_l = []
                train_l = []
                val_l = []
                test_l = []
                btrain_l = []
                bval_l = []
                btest_l = []

                for epoch in range(0, 2000):

                    loss = train()
                    tacc, valacc, testacc = test()

                    if tacc > btacc:
                        btacc = tacc
                    if valacc > bvalacc:
                        bvalacc = valacc
                    if testacc > btestacc:
                        btestacc = testacc

                    epoch_l.append(epoch)
                    loss_l.append(loss)
                    train_l.append(tacc)
                    val_l.append(valacc)
                    test_l.append(testacc)
                    btrain_l.append(btacc)
                    bval_l.append(bvalacc)
                    btest_l.append(btestacc)

                    log(Epoch=epoch, Loss=loss, BestTrainAcc=btacc, BestValAcc=bvalacc, BestTestAcc=btestacc)

                    if epoch < 750:
                        scheduler.step()

                # plot the training & validation & test process
                plt.plot(epoch_l, loss_l, label='loss')
                plt.plot(epoch_l, train_l, label='train_acc')
                plt.plot(epoch_l, val_l, label='val_acc')
                plt.plot(epoch_l, test_l, label='test_acc')
                # plt.plot(epoch_l, btrain_l, label = 'best_train_acc')
                # plt.plot(epoch_l, bval_l, label = 'best_val_acc')
                # plt.plot(epoch_l, btest_l, label = 'best_test_acc')
                plt.legend()
                plt.xlabel('number of epochs')
                plt.ylabel('values')
                plt.show()

                results_list.append(btacc)
                results_list.append(bvalacc)
                results_list.append(btestacc)
                # save model to the directory
                torch.save(model, dir_path() + 'NASDAQ/model' + '_' + str(ii) + '_' + str(jj) + '_' + str(kk))

                model = torch.load(dir_path() + 'NASDAQ/model' + '_' + str(ii) + '_' + str(jj) + '_' + str(kk))

                pp_list.append(results_list)

            f = open(dir_path() + 'NASDAQ/model' + '_' + str(ii) + '_' + str(jj) + '_' + str(kk) + '.txt', 'w')
            for content in pp_list:
                f.write(str(content) + '\n')
            f.close()
