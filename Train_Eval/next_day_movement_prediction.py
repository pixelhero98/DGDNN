import torch
import torch.nn.functional as F
from raw_data import gx_generation, node_feature_label_generation
from path_config import dir_path
from Mydataset import TrDataset_0, TrDataset_1, TrDataset_2
from dgdnn import DGDNN
from torch_geometric.logging import log
import torch.distributions
import numpy as np
import matplotlib.pyplot as plt
import random
from matplotlib import cm
from matplotlib import axes
import seaborn as sns
import sklearn.preprocessing as skp
from sklearn import metrics


#

def f1_score_clc(t_label, pre_label):
    score = 0
    for index, i in enumerate(t_label):
        score += metrics.f1_score(i, pre_label[index])

    return score / t_label.shape[0]


def mcc_coeff(t_label, pre_label):
    score = 0
    for index, i in enumerate(t_label):
        score += metrics.matthews_corrcoef(i, pre_label[index])

    return score / t_label.shape[0]


# configure the device for running the model on GPU or CPU

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# configure the default parameters

for ii in ['TrDataset_2']:

    print('generating Inmemory Dataset, please wait...')

    for jj in ['gx']:

        for kk in ['1.0']:
            pp_list = []
            for vv in range(10):
                results_list = []

                # load training dataset

                if ii == 'TrDataset_2':
                    traindata = TrDataset_2(root=dir_path() + 'test').data

                D1 = []
                D1, new_label = node_feature_label_generation(D1, 2015, 2015)

                new_tmp_seq = []
                new_tmp_seq1 = []
                days = int(D1.shape[1] / 5)

                D2 = []
                D2, new_label = node_feature_label_generation(D2, 2016, 2016)
                testdata = TrDataset_0(root=dir_path() + 'train').data

                # generate up & down sequence
                for index_pr, ppp in enumerate(D1):
                    tmp = D1[index_pr][-2 * days + 1:-days] - D1[index_pr][:days - 1]

                    new_tmp_seq.append(torch.Tensor(tmp))

                new_tmp_seq = torch.nn.utils.rnn.pad_sequence(new_tmp_seq, batch_first=True, padding_value=0)

                new_tmp_seq = torch.Tensor(skp.normalize(new_tmp_seq)).type(torch.float)

                for index_pr, ppp in enumerate(D2):
                    tmp = D2[index_pr][-2 * days + 1:-days] - D2[index_pr][:days - 1]

                    new_tmp_seq1.append(torch.Tensor(tmp))

                new_tmp_seq1 = torch.nn.utils.rnn.pad_sequence(new_tmp_seq1, batch_first=True, padding_value=0)

                new_tmp_seq1 = torch.Tensor(skp.normalize(new_tmp_seq1)).type(torch.float)


                # traindata.x = torch.Tensor(skp.normalize(D1, axis=0)).type(torch.float)

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

                # init model
                if jj == 'gx':
                    model = DGDNN(ins=traindata.x.shape[1] + traindata.x.shape[1], hids0=traindata.x.shape[1],
                                   hids1=1024, hids2=512,
                                   outs=256, num_labels=new_tmp_seq.shape[1], adj_dim=traindata.x.shape[0])
                else:
                    model = DGDNN(ins=traindata.x.shape[1] + traindata.x.shape[0], hids0=traindata.x.shape[0],
                                   hids1=2048, hids2=1024,
                                   outs=512, num_labels=6, adj_dim=traindata.x.shape[0])

                # model and dataset to GPU
                model, traindata = model.to(device), traindata.to(device)

                # define optimizer
                optimizer = torch.optim.Adam(params=model.parameters(), lr=2e-4)
                scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.997)
                # define training process & testing process

                gx1 = []
                gx1 = gx_generation(gx1, testdata)

                gx1, testdata = gx1.to(device), testdata.to(device)


                def train():
                    model.train()
                    optimizer.zero_grad()
                    out = model(traindata.x, traindata.edge_index, gx)

                    loss = F.cross_entropy(out, new_tmp_seq.to(device))
                    loss.backward()
                    optimizer.step()

                    return float(loss.item())


                @torch.no_grad()
                def test():
                    model.eval()
                    accs = []
                    out = model(traindata.x, traindata.edge_index, gx)
                    tmp_pre = out

                    tmp_pre[tmp_pre > 0] = 1
                    tmp_pre[tmp_pre < 0] = 0

                    tmp_new = new_tmp_seq

                    tmp_new[tmp_new > 0] = 1
                    tmp_new[tmp_new < 0] = 0

                    accs.append(int((tmp_pre == tmp_new.to(device)).sum()) / int(out.shape[0] * out.shape[1]))

                    out = model(testdata.x, testdata.edge_index, gx1)
                    tmp_pre = out

                    tmp_pre[tmp_pre > 0] = 1
                    tmp_pre[tmp_pre < 0] = 0

                    tmp_new1 = new_tmp_seq1

                    tmp_new1[tmp_new1 > 0] = 1
                    tmp_new1[tmp_new1 < 0] = 0

                    accs.append(int((tmp_pre == tmp_new1.to(device)).sum()) / int(out.shape[0] * out.shape[1]))

                    score = f1_score_clc(tmp_new1.cpu(), tmp_pre.cpu())

                    mcc = mcc_coeff(tmp_new1.cpu(), tmp_pre.cpu())

                    return accs, score, mcc


                # record accuracy change with respect to training process

                loss_0 = 0
                testacc_0 = 0
                score_0 = 0
                mcc_0 = 0
                epoch_l = []
                loss_l = []
                train_l = []
                val_l = []
                test_l = []
                btrain_l = []
                bval_l = []
                btest_l = []

                for epoch in range(0, 5000):

                    loss = train()
                    [tacc, testacc], score, mcc = test()

                    epoch_l.append(epoch)
                    loss_l.append(loss)
                    train_l.append(tacc)

                    btrain_l.append(tacc)

                    loss_0 = loss
                    testacc_0 = testacc
                    score_0 = score
                    mcc_0 = mcc

                    log(Epoch=epoch, Loss=loss, TrainAcc=tacc, TestAcc=testacc, F1_score=score, MCC=mcc)

                    if epoch < 805:
                        scheduler.step()

                torch.save(model, dir_path() + 'model_next_day' + '_' + str(ii) + '_' + str(jj) + '_' + str(kk))

                # plot the training & validation & test process
                # plt.plot(epoch_l, loss_l, label='loss')
                # plt.plot(epoch_l, train_l, label='train_acc')
                # plt.plot(epoch_l, btrain_l, label = 'best_train_acc')
                # plt.plot(epoch_l, bval_l, label = 'best_val_acc')
                # plt.plot(epoch_l, btest_l, label = 'best_test_acc')
                # plt.legend()
                # plt.xlabel('number of epochs')
                # plt.ylabel('values')
                # plt.show()

                results_list.append(loss_0)
                results_list.append(testacc_0)
                results_list.append(score_0)
                results_list.append(mcc_0)
                # save model to the directory
                torch.save(model, dir_path() + 'model_next_day' + '_' + str(ii) + '_' + str(jj) + '_' + str(kk))

                # model = torch.load(dir_path() + 'NASDAQ/model' + '_' + str(ii) + '_' + str(jj) + '_' + str(kk))

                pp_list.append(results_list)

                f = open(dir_path() + 'model_next_day' + '_' + str(ii) + '_' + str(jj) + '_' + str(kk) + '.txt', 'w')
                for content in pp_list:
                    f.write(str(content) + '\n')
                f.close()
